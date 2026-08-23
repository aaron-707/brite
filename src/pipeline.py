"""Pipeline orchestrator: wires Parser → Retriever → Gate → Synthesizer → Validator.

Thin wiring layer — no business logic.  Each component is called in sequence
and can be replaced independently.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import re

from .citation_validator import CitationValidator, ValidationResult
from .gate import Gate, GateDecision
from .parser import Clause, build_clause_index, parse_corpus
from .retriever import HybridRetriever, RetrievalResult
from .synthesizer import Synthesizer, SynthesizerOutput


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

    _RAW_CLAUSE_RE = re.compile(r"^§?(\d+\.\d+(?:\.\d+)?)$")

    def __init__(
        self,
        corpus_path: str | Path | None = None,
        config_path: str | Path = "config/gate_thresholds.yaml",
    ) -> None:
        # Parse corpus once
        self.clauses = parse_corpus(corpus_path)
        self.clause_index = build_clause_index(self.clauses)

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

    def run(self, question: str) -> PipelineResult:
        """Run the full pipeline for a question.

        Args:
            question: The user's question about the policy manual.

        Returns:
            PipelineResult with the decision, answer, and supporting data.
        """
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
        correction: str | None = None
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


def main() -> None:
    """CLI entry point."""
    if len(sys.argv) < 2:
        print("Usage: python -m src.pipeline \"<question>\"")
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    print(f"Question: {question}\n")

    pipeline = Pipeline()
    result = pipeline.run(question)

    print(f"Decision: {result.decision}")
    if result.gate_decision.conflicts:
        print(f"Conflicts: {result.gate_decision.conflicts}")
        cids = []
        for conflict in result.gate_decision.conflicts:
            cids.extend(re.findall(r"(\d+\.\d+(?:\.\d+)?)", conflict))
        cids = sorted(list(set(cids)))
        if cids:
            print("\nConflicting provisions:")
            for cid in cids:
                clause = pipeline.clause_index.get(cid)
                if clause:
                    print(f"\n  §{cid}: {clause.text}")
    print(f"\nAnswer:\n{result.answer}")
    if result.citations:
        print(f"\nCitations: {', '.join('§' + c for c in result.citations)}")
        print("\nSources:")
        for clause_id in result.citations:
            clause = pipeline.clause_index.get(clause_id)
            if clause:
                print(f"\n  §{clause_id}: {clause.text[:300]}"
                      f"{'...' if len(clause.text) > 300 else ''}")
    if result.validation and not result.validation.valid:
        print(f"\nValidation errors: {result.validation.errors}")


if __name__ == "__main__":
    main()
