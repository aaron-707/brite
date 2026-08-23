"""Pipeline orchestrator: wires Parser → Retriever → Gate → Synthesizer → Validator.

Thin wiring layer — no business logic.  Each component is called in sequence
and can be replaced independently.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import re

from .amendment_parser import parse_amendment, AmendmentRecord
from .citation_validator import CitationValidator, ValidationResult
from .gate import Gate, GateDecision
from .parser import Clause, build_clause_index, parse_corpus, parse_part_headings
from .retriever import HybridRetriever, RetrievalResult
from .synthesizer import Synthesizer, SynthesizerOutput
from .temporal import resolve_clause, ClauseVersion


# ── Month name → number ───────────────────────────────────────────────────────
_MONTH_NAMES = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}

# ISO date: 2026-04-15
_ISO_DATE_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
# "15 April 2026" or "April 15, 2026" or "April 2026"
_NAMED_DATE_RE = re.compile(
    r"\b(?:(\d{1,2})\s+)?(january|february|march|april|may|june|july|august"
    r"|september|october|november|december)(?:\s+(\d{4}))?\b",
    re.IGNORECASE,
)

# Phrases that signal the event_date context
_EVENT_PHRASES = re.compile(
    r"\b(?:change(?:d)?\s+(?:of\s+)?circumstances?\s+(?:in|on|from|during)"
    r"|reported?\s+(?:in|on)\s+"
    r"|change\s+occurred\s+(?:in|on)"
    r"|event\s+(?:in|on)"
    r"|claim\s+from"
    r"|happened\s+in)\b",
    re.IGNORECASE,
)

# Phrases that signal the determination_date context
_DET_PHRASES = re.compile(
    r"\b(?:determination\s+(?:made\s+)?(?:in|on|today)"
    r"|decided\s+(?:in|on|today)"
    r"|assessed\s+(?:in|on|today)"
    r"|as\s+of\s+today"
    r"|made\s+today)\b",
    re.IGNORECASE,
)


def _parse_named_date(match: re.Match) -> date | None:
    """Convert a named-month regex match to a date.  Day defaults to 1."""
    day_str, month_str, year_str = match.group(1), match.group(2), match.group(3)
    month = _MONTH_NAMES.get(month_str.lower())
    if month is None:
        return None
    year = int(year_str) if year_str else date.today().year
    day = int(day_str) if day_str else 1
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _extract_dates(question: str) -> tuple[date, date | None]:
    """Extract determination_date and event_date from a question string.

    Strategy (first match wins within each context):
      1. ISO dates (YYYY-MM-DD)
      2. Named-month dates ("in February 2026", "15 March 2026")
      3. "today" → today's date for determination_date

    Heuristic for event vs determination context: if an event-phrase
    precedes a date, it's taken as event_date; otherwise it contributes
    to determination_date.  With only one date found, it's always
    determination_date and event_date stays None.

    LIMITATION (Deliberate): For queries with 3+ dates (or multiple dates of the
    same type), the parser assigns the first candidate not preceded by an event
    phrase as determination_date and ignores subsequent ones. For example:
      "If a change occurred in February 2026, and I was deciding it in March 2026,
       what is the disregard for a claim spanning April 2026?"
    Here, "February 2026" (event date) is parsed. If "March 2026" is the determination date,
    but "spanning April 2026" appears later, the first non-event date (March) is chosen,
    and subsequent dates are ignored. In a query like:
      "Comparing April 2026 and February 2026, what is the disregard?"
    "April 2026" would be incorrectly selected as determination_date simply because
    it appears first, showing how the "first mentioned wins" heuristic has limitations.

    Returns:
        (determination_date, event_date)
        determination_date defaults to today if nothing is found.
        event_date defaults to None if not mentioned.
    """
    text = question

    # Collect all date candidates with their position and kind
    candidates: list[tuple[int, date, str]] = []  # (pos, date, kind_hint)

    for m in _ISO_DATE_RE.finditer(text):
        try:
            d = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            candidates.append((m.start(), d, "iso"))
        except ValueError:
            pass

    for m in _NAMED_DATE_RE.finditer(text):
        d = _parse_named_date(m)
        if d:
            candidates.append((m.start(), d, "named"))

    # Check for "today"
    if re.search(r"\btoday\b", text, re.IGNORECASE):
        candidates.append((0, date.today(), "today"))

    if not candidates:
        return date.today(), None

    # Sort by position
    candidates.sort(key=lambda x: x[0])

    if len(candidates) == 1:
        return candidates[0][1], None

    # With 2+ dates: check for event-phrase before each candidate
    event_date: date | None = None
    det_date: date | None = None

    for pos, d, kind in candidates:
        # Look at the 100 chars before this date for event/det phrase
        window = text[max(0, pos - 100): pos]
        if _EVENT_PHRASES.search(window) and event_date is None:
            event_date = d
        elif det_date is None:
            det_date = d

    # If we have both, fine; if only event_date found, det_date = today
    determination_date = det_date if det_date else date.today()
    return determination_date, event_date


def _dedup_conflicts(conflicts: list[str]) -> list[str]:
    seen_clauses = set()
    deduped = []
    for entry in conflicts:
        clause_ids = re.findall(r"§?(\d+\.\d+(?:\.\d+[A-Za-z]?)?)", entry)
        key = frozenset(clause_ids)
        if key not in seen_clauses:
            seen_clauses.add(key)
            deduped.append(entry)
    return deduped


@dataclass
class PipelineResult:
    """Full result from a pipeline run."""

    question: str
    decision: str  # "ANSWER" | "REFUSE" | "FLAG_CONFLICT"
    answer: str
    citations: list[str]
    gate_decision: GateDecision
    validation: ValidationResult | None = None
    retrieval_results: list[RetrievalResult] | None = None



class Pipeline:
    """Orchestrates the full RAG pipeline."""

    _RAW_CLAUSE_RE = re.compile(r"^§?(\d+\.\d+(?:\.\d+[A-Za-z]?)?)$")

    def __init__(
        self,
        corpus_path: str | Path | None = None,
        config_path: str | Path = "config/gate_thresholds.yaml",
    ) -> None:
        # Parse corpus once
        self.clauses = parse_corpus(corpus_path)
        self.clause_index = build_clause_index(self.clauses)
        self.part_headings = parse_part_headings(corpus_path)
        # Load amendment records once — used per-query to build resolved index
        self.amendments = parse_amendment()
        # Set of clause IDs touched by any amendment (fast membership test)
        self._amended_ids: frozenset[str] = frozenset(
            r.target_clause_id for r in self.amendments
        )

        # Inject inserted clauses (e.g. §10.5.3A) into the retriever index.
        # These clauses don't exist in the base corpus; they live only in the
        # amendment's insert_after records.  We add them here — using their
        # post-amendment text — so the retriever can find them for queries
        # about the new provision.  Per-query temporal patching in run() then
        # drops them from retrieved results for pre-effective determinations,
        # so Gate and Synthesizer never see them as if they were current law.
        for rec in self.amendments:
            if rec.operation == "insert_after" and rec.target_clause_id not in self.clause_index:
                synthetic = Clause(
                    clause_id=rec.target_clause_id,
                    text=rec.new_value,
                    cross_references=[],
                )
                self.clauses.append(synthetic)
                self.clause_index[rec.target_clause_id] = synthetic

        # Initialize components
        self.retriever = HybridRetriever(self.clauses, corpus_path=corpus_path, config_path=config_path)
        self.gate = Gate(config_path=config_path)
        self.synthesizer = Synthesizer(config_path=config_path)
        self.validator = CitationValidator(config_path=config_path)

        # Load max_retries from config
        import yaml

        cp = Path(config_path)
        if cp.exists():
            with open(cp, encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        else:
            cfg = {}
        self.max_retries: int = cfg.get("synthesizer", {}).get("max_retries", 1)

    def run(
        self,
        question: str,
        determination_date: date | None = None,
        event_date: date | None = None,
    ) -> PipelineResult:
        """Run the full pipeline for a question.

        Args:
            question: The user's question about the policy manual.
            determination_date: Override the determination date for temporal
                resolution.  If None, extracted from the question text or
                defaulted to today.
            event_date: Override the event date (change of circumstances) for
                §5.2 temporal resolution.  If None, extracted from the
                question or left as None (unresolved / ambiguous).

        Returns:
            PipelineResult with the decision, answer, and supporting data.
        """
        # ── Temporal date resolution ──────────────────────────────────────
        # Always extract from question text first; explicit overrides win.
        det_overridden = determination_date is not None
        extracted_det, extracted_evt = _extract_dates(question)
        if determination_date is None:
            determination_date = extracted_det
        if event_date is None:
            event_date = extracted_evt
        # (If the caller passed explicit dates, extraction results are discarded.)

        # Default event_date to determination_date if no overrides or extracted
        # event dates were found, AND the question contains no explicit date text,
        # so that generic queries reflect current reality while date-containing queries
        # remain properly constrained/ambiguous.
        has_extracted_dates = (
            _ISO_DATE_RE.search(question) is not None or
            _NAMED_DATE_RE.search(question) is not None or
            re.search(r"\btoday\b", question, re.IGNORECASE) is not None
        )
        if event_date is None and not det_overridden and not has_extracted_dates:
            event_date = determination_date

        # Fast path: raw clause reference lookup
        stripped = question.strip()
        match = self._RAW_CLAUSE_RE.match(stripped)
        if match:
            clause_id = match.group(1)
            clause = self.clause_index.get(clause_id)
            if clause:
                return PipelineResult(
                    question=question,
                    decision="ANSWER",
                    answer=clause.text,
                    citations=[clause_id],
                    gate_decision=GateDecision(
                        decision="ANSWER",
                        reason=f"Raw clause §{clause_id} lookup request.",
                    ),
                    validation=ValidationResult(valid=True, errors=[]),
                )
            else:
                return PipelineResult(
                    question=question,
                    decision="REFUSE",
                    answer=f"Clause §{clause_id} was not found in the manual.",
                    citations=[],
                    gate_decision=GateDecision(
                        decision="REFUSE",
                        reason=f"Clause §{clause_id} not found.",
                    ),
                )

        # Step 1: Retrieve
        expanded_question = self.retriever._expand_query(question)
        results = self.retriever.query(expanded_question)

        # Step 1b: Temporal patch — resolve amended clause texts, filter by date.
        # For each retrieval result whose clause_id is affected by an amendment,
        # call resolve_clause with the query's determination_date and event_date.
        # Results where exists=False (clause not yet in force) are DROPPED so
        # Gate and Synthesizer never see them as current law.
        # The patched clause_text flows through unchanged for everything else
        # — no modifications to Gate/Synthesizer internals required.
        patched_results: list[RetrievalResult] = []
        has_ambiguous = False
        has_apportionment = False

        # Set apportionment if the query dates span 1 March 2026, or if any two
        # dates extracted directly from the question text span that boundary.
        if determination_date is not None and event_date is not None:
            d1 = min(determination_date, event_date)
            d2 = max(determination_date, event_date)
            if d1 < date(2026, 3, 1) <= d2:
                has_apportionment = True

        all_candidates = []
        for m in _ISO_DATE_RE.finditer(question):
            try:
                all_candidates.append(date(int(m.group(1)), int(m.group(2)), int(m.group(3))))
            except ValueError:
                pass
        for m in _NAMED_DATE_RE.finditer(question):
            d = _parse_named_date(m)
            if d:
                all_candidates.append(d)

        if len(all_candidates) >= 2:
            d_before = any(d < date(2026, 3, 1) for d in all_candidates)
            d_after = any(d >= date(2026, 3, 1) for d in all_candidates)
            if d_before and d_after:
                has_apportionment = True

        for result in results:
            if result.clause_id in self._amended_ids:
                try:
                    cv = resolve_clause(
                        result.clause_id,
                        determination_date=determination_date,
                        event_date=event_date,
                        clause_index=self.clause_index,
                        amendments=self.amendments,
                    )
                    if not cv.exists:
                        # Clause not yet in force for this determination_date:
                        # drop it entirely — don't let pre-amendment synthetic
                        # text leak into Gate or Synthesizer.
                        continue
                    if cv.apportionment:
                        has_apportionment = True
                    if cv.ambiguous:
                        has_ambiguous = True
                        result.clause_text = (
                            f"[AMBIGUOUS VERSION - DEPENDS ON EVENT DATE]\n"
                            f"If the change of circumstances occurred before 1 March 2026:\n"
                            f"{cv.old_text}\n\n"
                            f"If the change of circumstances occurred on or after 1 March 2026:\n"
                            f"{cv.new_text}"
                        )
                    elif cv.apportionment or has_apportionment:
                        old_txt = cv.old_text if cv.old_text else self.clause_index[result.clause_id].text
                        post_effective_date = max(determination_date, date(2026, 3, 1))
                        try:
                            cv_post = resolve_clause(
                                result.clause_id,
                                determination_date=post_effective_date,
                                event_date=event_date if event_date else post_effective_date,
                                clause_index=self.clause_index,
                                amendments=self.amendments,
                            )
                            new_txt = cv_post.text if cv_post.text else cv_post.new_text
                        except KeyError:
                            new_txt = cv.text
                        if not new_txt:
                            new_txt = cv.text
                        result.clause_text = (
                            f"[APPORTIONED VERSION - CLAIM SPANS 1 MARCH 2026 BOUNDARY]\n"
                            f"For the portion of the claim period before 1 March 2026:\n"
                            f"{old_txt}\n\n"
                            f"For the portion of the claim period on or after 1 March 2026:\n"
                            f"{new_txt}"
                        )
                    elif cv.text is not None:
                        result.clause_text = cv.text
                except KeyError:
                    pass  # shouldn't happen; defensive
            patched_results.append(result)
        results = patched_results

        if has_apportionment:
            # Sourced dynamically from parsed corpus
            c743 = self.clause_index.get("7.4.3")
            if c743 is None:
                raise KeyError("Clause §7.4.3 was not found in the parsed corpus.")
            p743_text = c743.text

            # Inject §5.3 (from amendment) and §7.4.3 (from base manual) as retrieved documents
            # so the synthesizer and citation validator can verify references to them.
            p53_text = (
                "**5.3** Where a claim relates to a period spanning 1 March 2026, the applicable "
                "figures are those in force on each day of the period, and the award is apportioned "
                "accordingly under §7.4.3."
            )
            results.append(RetrievalResult(
                clause_id="5.3",
                clause_text=p53_text,
                score=1.0,
                bm25_rank=1,
                tfidf_rank=1
            ))
            results.append(RetrievalResult(
                clause_id="7.4.3",
                clause_text=p743_text,
                score=1.0,
                bm25_rank=1,
                tfidf_rank=1
            ))

        # Step 2: Gate
        gate_decision = self.gate.evaluate(expanded_question, results, self.clause_index)

        if gate_decision.decision == "REFUSE":
            return PipelineResult(
                question=question,
                decision="REFUSE",
                answer=gate_decision.reason,
                citations=[],
                gate_decision=gate_decision,
                retrieval_results=results,
            )

        # Deduplicate conflicts before passing to synthesizer
        if gate_decision.conflicts:
            gate_decision.conflicts = _dedup_conflicts(gate_decision.conflicts)

        # Step 3 & 4: Synthesize + Validate (with retry)
        instructions = []
        if has_ambiguous:
            instructions.append(
                "IMPORTANT: The retrieved context contains ambiguous clauses where the applicable version "
                "depends on when the change of circumstances occurred (before or on/after 1 March 2026). "
                "You must state plainly that the answer depends on when the change of circumstances occurred, "
                "provide both values with that condition clearly attached, and do not present either value "
                "as the sole answer."
            )
        if has_apportionment:
            instructions.append(
                "IMPORTANT: The claim/event period spans the 1 March 2026 amendment boundary. "
                "Under §5.3 and §7.4.3, you must explicitly state in the answer that the claim spans the amendment boundary, "
                "cite §5.3 and §7.4.3 immediately when mentioning the boundary or apportionment, and say that the award must be apportioned across the two rate periods. "
                "Make sure every sentence mentioning the date span, the boundary, or the apportionment contains an inline citation to (5.3) or (7.4.3). "
                "Do not attempt to calculate or present a specific prorated/apportioned/calculated figure, "
                "as that arithmetic is out of scope for this system."
            )

        initial_instruction = "\n\n".join(instructions) if instructions else None
        correction: str | None = initial_instruction
        last_output: SynthesizerOutput | None = None
        last_validation: ValidationResult | None = None

        for attempt in range(1 + self.max_retries):
            synth_output = self.synthesizer.generate(
                question, results, gate_decision, correction=correction
            )
            if isinstance(synth_output, dict):
                return PipelineResult(
                    question=question,
                    decision=synth_output.get("decision", "REFUSE"),
                    answer=synth_output.get("answer", ""),
                    citations=synth_output.get("citations", []),
                    gate_decision=gate_decision,
                    validation=ValidationResult(valid=False, errors=["API error"]),
                    retrieval_results=results,
                )
            last_output = synth_output

            validation = self.validator.validate(synth_output, results)
            last_validation = validation

            if validation.valid:
                return PipelineResult(
                    question=question,
                    decision=gate_decision.decision,
                    answer=synth_output.answer,
                    citations=synth_output.cited_clause_ids,
                    gate_decision=gate_decision,
                    validation=validation,
                    retrieval_results=results,
                )

            # Build correction instruction for retry
            error_text = "\n".join(f"- {e}" for e in validation.errors)
            correction = (
                f"Your previous answer failed citation validation. "
                f"Fix these issues:\n{error_text}\n\n"
                f"Remember: only cite clause ids from the provided clauses, "
                f"and every factual claim must be supported by a cited clause."
            )
            if initial_instruction:
                correction = initial_instruction + "\n\n" + correction

        # All retries exhausted — return a REFUSE-style output
        error_summary = "; ".join(last_validation.errors) if last_validation else "Unknown"
        return PipelineResult(
            question=question,
            decision="REFUSE",
            answer=(
                f"Unable to generate a fully verified answer after "
                f"{1 + self.max_retries} attempts. Validation errors: "
                f"{error_summary}. Please ask a supervisor for assistance."
            ),
            citations=[],
            gate_decision=gate_decision,
            validation=last_validation,
            retrieval_results=results,
        )


def _cid_sort_key(c: str) -> tuple:
    parts = []
    for p in c.split("."):
        m = re.match(r"^(\d+)([A-Za-z]?)$", p)
        if m:
            parts.append((int(m.group(1)), m.group(2)))
        else:
            parts.append((0, p))
    return tuple(parts)


def _parse_date_flag(value: str, flag: str) -> date:
    """Parse a YYYY-MM-DD date string from a CLI flag; exit on error."""
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        print(f"Error: {flag} must be a date in YYYY-MM-DD format (got {value!r})")
        sys.exit(1)


def main() -> None:
    """CLI entry point.

    Usage:
        python -m src.pipeline "<question>"
        python -m src.pipeline --source §4.3.2
        python -m src.pipeline --as-of 2026-02-15 "<question>"
        python -m src.pipeline --as-of 2026-04-15 --event-date 2026-02-10 "<question>"
    """
    args = sys.argv[1:]

    # ── Parse --as-of / --event-date overrides ────────────────────────────
    det_override: date | None = None
    evt_override: date | None = None
    remaining: list[str] = []
    i = 0
    while i < len(args):
        if args[i] == "--as-of" and i + 1 < len(args):
            det_override = _parse_date_flag(args[i + 1], "--as-of")
            i += 2
        elif args[i] == "--event-date" and i + 1 < len(args):
            evt_override = _parse_date_flag(args[i + 1], "--event-date")
            i += 2
        else:
            remaining.append(args[i])
            i += 1
    args = remaining

    # ── --source §X.Y.Z fast-path ─────────────────────────────────────────
    if args and args[0] == "--source":
        if len(args) < 2:
            print("Usage: python -m src.pipeline --source §X.Y.Z")
            sys.exit(1)
        raw_id = args[1].lstrip("§")
        pipeline = Pipeline()
        clause = pipeline.clause_index.get(raw_id)
        if not clause:
            print(f"Clause §{raw_id} not found in the policy manual.")
            sys.exit(1)
        part_heading = pipeline.part_headings.get(clause.part(), f"Part {clause.part()}")
        print(f"{part_heading}\n")
        print(f"§{clause.clause_id}\n")
        print(clause.text)
        return

    # ── Normal question path ───────────────────────────────────────────────
    if not args:
        print('Usage: python -m src.pipeline "<question>"')
        print("       python -m src.pipeline --source §X.Y.Z")
        print("       python -m src.pipeline --as-of YYYY-MM-DD [--event-date YYYY-MM-DD] \"<question>\"")
        sys.exit(1)

    question = " ".join(args)
    print(f"Question: {question}\n")

    pipeline = Pipeline()
    result = pipeline.run(
        question,
        determination_date=det_override,
        event_date=evt_override,
    )

    # Show which dates governed this query
    extracted_det, extracted_evt = _extract_dates(question)
    eff_det = det_override if det_override else extracted_det
    eff_evt = evt_override if evt_override is not None else extracted_evt
    print(f"Temporal context: determination_date={eff_det}"
          + (f", event_date={eff_evt}" if eff_evt else "") + "\n")

    print(f"Decision: {result.decision}")
    if result.gate_decision.conflicts:
        print(f"Conflicts: {result.gate_decision.conflicts}")
        cids = []
        for conflict in result.gate_decision.conflicts:
            cids.extend(re.findall(r"(\d+\.\d+(?:\.\d+[A-Za-z]?)?)", conflict))
        cids = sorted(list(set(cids)), key=_cid_sort_key)
        if cids:
            print("\nConflicting provisions:")
            for cid in cids:
                clause = pipeline.clause_index.get(cid)
                if clause:
                    print(f"\n  §{cid}: {clause.text}")
    print(f"\nAnswer:\n{result.answer}")
    if result.citations:
        print(f"\nCitations: {', '.join('§' + c for c in result.citations)}")
        # Build a lookup from resolved retrieval results for Sources block
        resolved_texts: dict[str, str] = {}
        if result.retrieval_results:
            for rr in result.retrieval_results:
                resolved_texts[rr.clause_id] = rr.clause_text
        print("\nSources:")
        for clause_id in result.citations:
            # Prefer resolved text from retrieval; fall back to base clause_index
            resolved_text = resolved_texts.get(clause_id)
            clause = pipeline.clause_index.get(clause_id)
            if resolved_text or clause:
                text_to_show = resolved_text or (clause.text if clause else "")
                part_num = clause.part() if clause else int(clause_id.split(".")[0])
                part_heading = pipeline.part_headings.get(part_num, f"Part {part_num}")
                print(f"\n  [{part_heading}]")
                print(f"  §{clause_id}:")
                for line in text_to_show.splitlines():
                    print(f"    {line}")
    if result.validation and not result.validation.valid:
        print(f"\nValidation errors: {result.validation.errors}")


if __name__ == "__main__":
    main()
