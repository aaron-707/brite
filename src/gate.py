"""Pre-LLM gate: decides ANSWER / REFUSE / FLAG_CONFLICT before any LLM call.

Signals:
  1. Top retrieval score (RRF)
  2. Term-overlap coverage between question and top clauses
  3. Cross-reference integrity (TF-IDF cosine similarity)
  4. Numeric contradiction detection across clauses sharing a cross-ref target

All thresholds read from config/gate_thresholds.yaml.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .parser import Clause
from .retriever import RetrievalResult, STOPWORDS

# Regex to extract numeric values with units from clause text
_NUM_UNIT_RE = re.compile(
    r"(\d+)\s*(calendar\s+days?|days?|weeks?|months?|per\s+cent|%)",
    re.IGNORECASE,
)

# Regex to find cross-reference targets
_XREF_RE = re.compile(r"§(\d+\.\d+(?:\.\d+)?)")


def _resolve_xref(ref: str, all_clauses: dict[str, Clause]) -> list[Clause]:
    """Resolve a cross-reference to one or more Clause objects.

    Handles both 3-digit refs (§4.3.2 → exact match) and 2-digit section
    refs (§4.3 → all clauses matching 4.3.*).
    """
    # Exact match (3-digit)
    if ref in all_clauses:
        return [all_clauses[ref]]

    # Section-level match (2-digit): find all clauses under that section
    prefix = ref + "."
    matches = [c for cid, c in all_clauses.items() if cid.startswith(prefix)]
    return matches


STRUCTURAL_CONNECTIVES = [
    "subject to the adjustments in",
    "subject to adjustments in",
    "except as provided in",
    "as described in",
    "as set out in",
    "as defined in",
    "under",
    "as specified in",
    "in accordance with",
    "referred to in",
    "where",
    "except where",
]


def _is_structural_reference(preceding_text: str) -> bool:
    """Return True if the citation is preceded by a structural connective
    phrase, meaning the reference is definitionally forward-pointing and
    term-overlap cannot assess it."""
    t = preceding_text.strip().lower()
    return any(t.endswith(phrase) for phrase in STRUCTURAL_CONNECTIVES)


def _dedup_conflicts(conflicts: list[str]) -> list[str]:
    seen_clauses = set()
    deduped = []
    for entry in conflicts:
        clause_ids = re.findall(r"§?(\d+\.\d+(?:\.\d+)?)", entry)
        key = frozenset(clause_ids)
        if key not in seen_clauses:
            seen_clauses.add(key)
            deduped.append(entry)
    return deduped



@dataclass
class GateDecision:
    """Result of the gate evaluation."""

    decision: str  # "ANSWER" | "REFUSE" | "FLAG_CONFLICT"
    reason: str
    conflicts: list[str] = field(default_factory=list)
    next_step: str = ""
    no_coverage: bool = False


# Minimum term-overlap coverage required before the gate passes a query to the
# synthesizer.  Chosen by empirical sweep across 16 queries (10 eval + 6
# borderline) at 0.15 / 0.25 / 0.35: 0.25 yields 0 false-refusals and 3
# false-answers (all handled correctly by the LLM or soft-fallback); 0.35 adds
# 2 false-refusals with only 1 marginal FA eliminated.  See DECISIONS.md §3
# "Why 0.25 as the coverage threshold" for the domain-asymmetry rationale.
MIN_TERM_COVERAGE: float = 0.25

_DEFAULT_CONFIG = {
    "gate": {
        "min_retrieval_score": 0.015,
        "min_term_coverage": MIN_TERM_COVERAGE,
        "xref_relevance_threshold": 0.3,
        "numeric_contradiction": True,
        "top_k_for_gate": 5,
    }
}


class Gate:
    """Pre-LLM gate that filters queries before synthesis."""

    def __init__(self, config_path: str | Path = "config/gate_thresholds.yaml") -> None:
        config_path = Path(config_path)
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        else:
            cfg = {}

        gate_cfg = cfg.get("gate", _DEFAULT_CONFIG["gate"])
        self.min_retrieval_score: float = gate_cfg.get("min_retrieval_score", 0.015)
        self.min_term_coverage: float = gate_cfg.get("min_term_coverage", MIN_TERM_COVERAGE)
        self.xref_relevance_threshold: float = gate_cfg.get("xref_relevance_threshold", 0.3)
        self.numeric_contradiction: bool = gate_cfg.get("numeric_contradiction", True)
        self.top_k_for_gate: int = gate_cfg.get("top_k_for_gate", 5)

    def _is_real_language_query(self, query: str) -> bool:
        """
        Returns True if the query contains at least 2 non-stopword
        tokens of length >= 3. Distinguishes real-language queries
        with vocabulary not in the manual from genuinely empty,
        single-character, or gibberish inputs.
        """
        tokens = [
            t for t in query.lower().split()
            if t not in STOPWORDS and len(t) >= 3
        ]
        return len(tokens) >= 2

    def evaluate(
        self,
        question: str,
        results: list[RetrievalResult],
        all_clauses: dict[str, Clause],
    ) -> GateDecision:
        """Evaluate whether to answer, refuse, or flag a conflict.

        Args:
            question: The user's question.
            results: Ranked retrieval results from the retriever.
            all_clauses: Full clause index for cross-reference lookup.

        Returns:
            A GateDecision with the decision, reason, and any conflicts.
        """
        if not hasattr(self, "_high_freq"):
            doc_counts: dict[str, int] = {}
            for cid, c in all_clauses.items():
                seen_tokens = set(re.findall(r"[a-z0-9]+", c.text.lower()))
                for token in seen_tokens:
                    doc_counts[token] = doc_counts.get(token, 0) + 1
            threshold = len(all_clauses) * 0.15
            self._high_freq = {t for t, count in doc_counts.items() if count > threshold}

        if not results:
            return GateDecision(
                decision="REFUSE",
                reason="No relevant clauses found in the policy manual.",
                next_step="Ask a supervisor or consult the governing statute.",
            )

        top_results = results[: self.top_k_for_gate]

        # ── Signal 1: Retrieval confidence ───────────────────────────────
        top_score = top_results[0].score
        if top_score < self.min_retrieval_score:
            return GateDecision(
                decision="REFUSE",
                reason=(
                    f"Top retrieval score ({top_score:.4f}) is below the "
                    f"minimum threshold ({self.min_retrieval_score}). "
                    "This question does not appear to be covered by the policy manual."
                ),
                next_step="Ask a supervisor or consult the governing statute.",
            )

        # ── Signal 2: Term-overlap coverage ──────────────────────────────
        q_tokens = {
            t for t in re.findall(r"[a-z0-9]+", question.lower())
            if t not in STOPWORDS and t not in self._high_freq
        }
        if q_tokens:
            clause_tokens: set[str] = set()
            for r in top_results:
                clause_tokens.update(
                    t for t in re.findall(r"[a-z0-9]+", r.clause_text.lower())
                    if t not in STOPWORDS and t not in self._high_freq
                )
            coverage = len(q_tokens & clause_tokens) / len(q_tokens)
        else:
            coverage = 0.0

        if coverage < self.min_term_coverage:
            if self._is_real_language_query(question):
                return GateDecision(
                    decision="ANSWER",
                    reason=(
                        f"Term-overlap coverage ({coverage:.2f}) is below "
                        f"minimum ({self.min_term_coverage}), but query has "
                        f"real language structure. Soft fallback triggered."
                    ),
                    no_coverage=True,
                )
            else:
                return GateDecision(
                    decision="REFUSE",
                    reason=(
                        f"Term-overlap coverage ({coverage:.2f}) is below "
                        f"minimum ({self.min_term_coverage}) and query lacks "
                        f"real language structure."
                    ),
                    no_coverage=False,
                )

        # ── Signal 3 & 4: Cross-reference checks ────────────────────────
        conflicts: list[str] = []

        # Collect all xrefs cited in the top results — only check the
        # highest-ranked results to avoid flagging structural cross-refs
        # from tangentially retrieved clauses.
        xref_check_depth = min(3, len(top_results))
        xref_contexts: list[tuple[str, str, int, str]] = []  # (source_clause_id, xref_target, match_start, source_text)
        for r in top_results[:xref_check_depth]:
            for m in _XREF_RE.finditer(r.clause_text):
                ref = m.group(1)
                if ref != r.clause_id:
                    xref_contexts.append((r.clause_id, ref, m.start(), r.clause_text))

        # Signal 3: Cross-reference relevance (Term overlap)
        for source_id, xref_target, match_start, source_text in xref_contexts:
            resolved = _resolve_xref(xref_target, all_clauses)
            if not resolved:
                conflicts.append(
                    f"§{source_id} references §{xref_target}, "
                    f"but §{xref_target} does not exist in the manual."
                )
                continue

            # Term overlap check: look at preceding ~6 words, excluding single chars & high-freq words
            window = source_text[max(0, match_start - 150):match_start]
            if _is_structural_reference(window):
                continue
            tokens = [t for t in re.findall(r"[a-z0-9]+", window.lower()) 
                      if t not in STOPWORDS and len(t) > 1]
            filtered_tokens = [t for t in tokens if t not in self._high_freq]
            key_terms = filtered_tokens[-6:] if len(filtered_tokens) >= 6 else filtered_tokens
            
            has_overlap = False
            if not key_terms:
                has_overlap = True
            else:
                target_text = " ".join(c.text.lower() for c in resolved)
                target_tokens = set(re.findall(r"[a-z0-9]+", target_text))
                for term in key_terms:
                    # Exact match
                    if term in target_tokens:
                        has_overlap = True
                        break
                    # prefix match for stemming (e.g. 'determined' vs 'determines')
                    if len(term) >= 5:
                        prefix = term[:5]
                        # Only allow prefix match if the target token is also not high-freq
                        for t in target_tokens:
                            if len(t) >= 5 and t.startswith(prefix):
                                if t not in self._high_freq:
                                    has_overlap = True
                                    break
                        if has_overlap:
                            break

            if not has_overlap:
                best_clause = resolved[0]
                conflicts.append(
                    f"§{source_id} cross-references §{xref_target}, but the "
                    f"referenced clause appears topically unrelated. "
                    f"§{best_clause.clause_id} discusses '{best_clause.text[:80].strip()}…' "
                    f"— this may be a dead or incorrect reference."
                )


        # Signal 4: Numeric contradiction detection
        if self.numeric_contradiction:
            # Build list of clauses (top retrieved clauses only)
            check_clauses = []
            seen_cids = set()
            for r in top_results:
                if r.clause_id not in seen_cids:
                    seen_cids.add(r.clause_id)
                    parent = ".".join(r.clause_id.split(".")[:2])
                    text = f"§{parent} {r.clause_text}"
                    check_clauses.append({"id": r.clause_id, "text": text})
            
            conflicts.extend(self._find_numeric_contradictions(check_clauses))

        if conflicts:
            return GateDecision(
                decision="FLAG_CONFLICT",
                reason="Potential inconsistencies detected in relevant clauses.",
                conflicts=conflicts,
                next_step="Review the flagged conflicts carefully.",
            )

        return GateDecision(
            decision="ANSWER",
            reason="Sufficient evidence found to answer the question.",
        )

    def _find_numeric_contradictions(self, clauses: list[dict]) -> list[str]:
        conflicts = []
        anchor_unit_map = {}

        for clause in clauses:
            text = clause["text"]
            clause_id = clause["id"]
            anchors = re.findall(r"§(\d+\.\d+)", text)
            num_unit_pairs = re.findall(
                r"(\d+(?:\.\d+)?)\s*(calendar days|days|per cent|percent|%|weeks|months|hours)",
                text, re.IGNORECASE
            )
            for anchor in anchors:
                for value, unit in num_unit_pairs:
                    unit_norm = unit.lower().replace("calendar ", "")
                    key = (anchor, unit_norm)
                    if key not in anchor_unit_map:
                        anchor_unit_map[key] = []
                    anchor_unit_map[key].append((clause_id, float(value)))

        for (anchor, unit), entries in anchor_unit_map.items():
            distinct_values = {}
            for clause_id, value in entries:
                if value not in distinct_values:
                    distinct_values[value] = clause_id
            if len(distinct_values) > 1:
                # Base-rule / exception pairs in the same section are NOT contradictions:
                prefixes = { ".".join(cid.split(".")[:2]) for cid in distinct_values.values() }
                if len(prefixes) <= 1:
                    continue
                parts = "; ".join(
                    f"§{cid} states {int(v) if v == int(v) else v}"
                    for v, cid in distinct_values.items()
                )
                conflicts.append(
                    f"Numeric contradiction referencing §{anchor}: "
                    f"{parts}. These values are inconsistent."
                )

        return _dedup_conflicts(conflicts)
