"""Citation validator: post-checks the synthesizer's output.

Checks:
1. Every clause id in ``citations`` must appear in the retrieved set.
2. Every factual sentence in ``answer`` should be supported by at least one
   cited clause (simple keyword-overlap check, not a semantic model).
3. No hallucinated clause ids.

On failure, returns details so the pipeline can retry with a correction prompt.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .retriever import RetrievalResult, STOPWORDS
from .synthesizer import SynthesizerOutput

_CITE_RE = re.compile(r"§?(\d+\.\d+(?:\.\d+[A-Za-z]?)?)")

# Sentences that are purely structural / connective — not factual claims
_NON_FACTUAL_PREFIXES = (
    "in summary",
    "therefore",
    "thus",
    "in conclusion",
    "overall",
    "to summarize",
    "note that",
    "however",
    "additionally",
    "furthermore",
    "based on the above",
    "as noted",
    "as stated",
)


@dataclass
class ValidationResult:
    """Result of citation validation."""

    valid: bool
    errors: list[str] = field(default_factory=list)
    unverified_sentences: list[str] = field(default_factory=list)


_DEFAULT_CONFIG = {
    "citation_validator": {
        "min_support_overlap": 0.15,
    }
}


class CitationValidator:
    """Validates that synthesizer output is grounded in retrieved clauses."""

    def __init__(self, config_path: str | Path = "config/gate_thresholds.yaml") -> None:
        config_path = Path(config_path)
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        else:
            cfg = {}

        val_cfg = cfg.get("citation_validator", _DEFAULT_CONFIG["citation_validator"])
        self.min_support_overlap: float = val_cfg.get("min_support_overlap", 0.15)

    def validate(
        self,
        synth_output: SynthesizerOutput,
        retrieved: list[RetrievalResult],
        question: str | None = None,
    ) -> ValidationResult:
        """Validate the synthesizer's output against retrieved clauses.

        Args:
            synth_output: The synthesizer's answer and cited clause ids.
            retrieved: The retrieval results that were passed to the synthesizer.

        Returns:
            ValidationResult indicating pass/fail with error details.
        """
        errors: list[str] = []
        unverified: list[str] = []

        retrieved_ids = {r.clause_id for r in retrieved}
        clause_texts = {r.clause_id: r.clause_text for r in retrieved}

        # ── Check 1: All cited clause ids must be in the retrieved set ────
        for cid in synth_output.cited_clause_ids:
            if cid not in retrieved_ids:
                errors.append(
                    f"Citation §{cid} was not in the retrieved clause set — "
                    f"possible hallucinated clause id."
                )

        # ── Check 2 & 3: Sentence-level support ─────────────────────────
        answer = synth_output.answer
        body = answer
        for marker in ["conflicting provisions", "sources"]:
            if marker in body.lower():
                # Split case-insensitively
                parts = re.split(re.escape(marker), body, flags=re.IGNORECASE)
                if parts:
                    body = parts[0]
        sentences = self._split_sentences(body)

        # Extract question tokens to ignore in verification overlap
        question_tokens = self._tokenize(question) if question else set()

        for sentence in sentences:
            stripped = sentence.strip()
            if not stripped:
                continue

            # Skip non-factual connector sentences
            lower = stripped.lower()
            if any(lower.startswith(p) for p in _NON_FACTUAL_PREFIXES):
                continue

            # Skip very short sentences (likely fragments)
            words = stripped.split()
            if len(words) < 4:
                continue

            # Extract citations from this sentence
            cited_in_sentence = _CITE_RE.findall(stripped)
            valid_cited = [c for c in cited_in_sentence if c in retrieved_ids]

            if not valid_cited:
                # Check if it's a factual claim (contains numbers, specific terms)
                if self._is_factual(stripped):
                    unverified.append(stripped)
                continue

            # Check keyword overlap between sentence and cited clause text
            sent_tokens = self._tokenize(stripped)
            if not sent_tokens:
                continue

            # Exclude tokens derived from the query itself
            filtered_sent_tokens = sent_tokens - question_tokens
            if not filtered_sent_tokens:
                # Sentence contains only query terms or stopwords; it is verified by definition
                supported = True
            else:
                supported = False
                for cid in valid_cited:
                    if cid in clause_texts:
                        clause_tokens = self._tokenize(clause_texts[cid])
                        if not clause_tokens:
                            continue
                        overlap = len(filtered_sent_tokens & clause_tokens) / len(filtered_sent_tokens)
                        if overlap >= self.min_support_overlap:
                            supported = True
                            break

            if not supported:
                unverified.append(
                    f"{stripped} [cited: {', '.join('§' + c for c in valid_cited)}]"
                )

        # Unverified factual sentences are errors
        if unverified:
            for s in unverified:
                errors.append(f"Unverified claim: {s}")

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            unverified_sentences=unverified,
        )

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text into sentences on period, question mark, exclamation."""
        # Split on sentence-ending punctuation followed by whitespace or end
        parts = re.split(r"(?<=[.!?])\s+", text)
        return [p.strip() for p in parts if p.strip()]

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """Lowercase, extract alphanumeric tokens, remove stopwords."""
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        return {t for t in tokens if t not in STOPWORDS}

    @staticmethod
    def _is_factual(sentence: str) -> bool:
        """Heuristic: does this sentence make a factual claim?

        Factual if it contains a number, a dollar amount, a specific time
        period, a clause reference, or policy-specific language.
        """
        indicators = [
            r"\d",  # any digit
            r"\$",  # dollar amount
            r"§",  # clause reference
            r"\b(must|shall|may not|is required|is not eligible|is eligible)\b",
        ]
        for pattern in indicators:
            if re.search(pattern, sentence):
                return True
        return False
