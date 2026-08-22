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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as cos_sim

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


@dataclass
class GateDecision:
    """Result of the gate evaluation."""

    decision: str  # "ANSWER" | "REFUSE" | "FLAG_CONFLICT"
    reason: str
    conflicts: list[str] = field(default_factory=list)
    next_step: str = ""


_DEFAULT_CONFIG = {
    "gate": {
        "min_retrieval_score": 0.015,
        "min_term_coverage": 0.25,
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
        self.min_term_coverage: float = gate_cfg.get("min_term_coverage", 0.25)
        self.xref_relevance_threshold: float = gate_cfg.get("xref_relevance_threshold", 0.3)
        self.numeric_contradiction: bool = gate_cfg.get("numeric_contradiction", True)
        self.top_k_for_gate: int = gate_cfg.get("top_k_for_gate", 5)

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
            t for t in re.findall(r"[a-z0-9]+", question.lower()) if t not in STOPWORDS
        }
        if q_tokens:
            clause_tokens: set[str] = set()
            for r in top_results:
                clause_tokens.update(
                    t for t in re.findall(r"[a-z0-9]+", r.clause_text.lower())
                    if t not in STOPWORDS
                )
            coverage = len(q_tokens & clause_tokens) / len(q_tokens)
        else:
            coverage = 0.0

        if coverage < self.min_term_coverage:
            return GateDecision(
                decision="REFUSE",
                reason=(
                    f"Term-overlap coverage ({coverage:.2f}) is below the "
                    f"minimum threshold ({self.min_term_coverage}). "
                    "The question terms are not well matched by the manual."
                ),
                next_step="This topic may not be covered. Ask a supervisor.",
            )

        # ── Signal 3 & 4: Cross-reference checks ────────────────────────
        conflicts: list[str] = []

        # Collect all xrefs cited in the top results — only check the
        # highest-ranked results to avoid flagging structural cross-refs
        # from tangentially retrieved clauses.
        xref_check_depth = min(3, len(top_results))
        xref_contexts: list[tuple[str, str, str]] = []  # (source_clause_id, xref_target, source_text)
        for r in top_results[:xref_check_depth]:
            refs = _XREF_RE.findall(r.clause_text)
            for ref in refs:
                if ref != r.clause_id:
                    xref_contexts.append((r.clause_id, ref, r.clause_text))


        # Signal 3: Cross-reference relevance
        for source_id, xref_target, source_text in xref_contexts:
            resolved = _resolve_xref(xref_target, all_clauses)
            if not resolved:
                conflicts.append(
                    f"§{source_id} references §{xref_target}, "
                    f"but §{xref_target} does not exist in the manual."
                )
                continue

            # Check topical relevance: compute similarity between the
            # source text and each resolved target clause.  Use the best
            # match — if *none* are relevant the xref is dead.
            best_sim = 0.0
            best_clause = resolved[0]
            for target_clause in resolved:
                try:
                    vec = TfidfVectorizer(lowercase=True, stop_words="english")
                    matrix = vec.fit_transform([source_text, target_clause.text])
                    sim = cos_sim(matrix[0:1], matrix[1:2])[0, 0]
                except ValueError:
                    sim = 0.0
                if sim > best_sim:
                    best_sim = sim
                    best_clause = target_clause

            if best_sim < self.xref_relevance_threshold:
                # Build a human-readable explanation
                conflicts.append(
                    f"§{source_id} cross-references §{xref_target}, but the "
                    f"referenced clause appears topically unrelated "
                    f"(similarity: {best_sim:.2f}). §{best_clause.clause_id} discusses "
                    f"'{best_clause.text[:80].strip()}…' — this may be a "
                    f"dead or incorrect reference."
                )


        # Signal 4: Numeric contradiction detection
        if self.numeric_contradiction:
            conflicts.extend(self._check_numeric_contradictions(top_results, all_clauses))

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

    def _check_numeric_contradictions(
        self,
        results: list[RetrievalResult],
        all_clauses: dict[str, Clause],
    ) -> list[str]:
        """Detect numeric contradictions across clauses referencing the same target.

        Looks for cases where two clauses cite the same §X.Y but state different
        numeric values (e.g. "10 calendar days" vs "30 calendar days").
        """
        conflicts: list[str] = []

        # Group: xref_target -> list of (source_clause_id, extracted_numbers)
        xref_numbers: dict[str, list[tuple[str, list[tuple[str, str]]]]] = {}

        for r in results:
            refs = _XREF_RE.findall(r.clause_text)
            nums = _NUM_UNIT_RE.findall(r.clause_text)
            for ref in refs:
                if ref != r.clause_id and nums:
                    key = ref
                    if key not in xref_numbers:
                        xref_numbers[key] = []
                    xref_numbers[key].append((r.clause_id, nums))

        # Also check the target clauses themselves
        for xref_target, sources in list(xref_numbers.items()):
            resolved = _resolve_xref(xref_target, all_clauses)
            for target_clause in resolved:
                target_nums = _NUM_UNIT_RE.findall(target_clause.text)
                if target_nums:
                    sources.append((target_clause.clause_id, target_nums))


        # Check for disagreements
        for xref_target, sources in xref_numbers.items():
            if len(sources) < 2:
                continue

            # Normalize units and compare
            seen_values: dict[str, list[str]] = {}  # "10 calendar days" -> [clause_ids]
            for source_id, nums in sources:
                for value, unit in nums:
                    normalized = f"{value} {unit.lower().strip()}"
                    if normalized not in seen_values:
                        seen_values[normalized] = []
                    seen_values[normalized].append(source_id)

            # If multiple distinct values exist for similar units, flag it
            unit_groups: dict[str, list[tuple[str, str]]] = {}  # base_unit -> [(value, clause_id)]
            for val_unit, clause_ids in seen_values.items():
                parts = val_unit.split(maxsplit=1)
                if len(parts) == 2:
                    base = parts[1].rstrip("s").replace("calendar ", "").strip()
                    for cid in clause_ids:
                        if base not in unit_groups:
                            unit_groups[base] = []
                        unit_groups[base].append((parts[0], cid))

            for base_unit, entries in unit_groups.items():
                distinct_values = set(v for v, _ in entries)
                if len(distinct_values) > 1:
                    detail_parts = [f"§{cid} states {v}" for v, cid in entries]
                    conflicts.append(
                        f"Numeric contradiction referencing §{xref_target}: "
                        + "; ".join(detail_parts)
                        + f". These values are inconsistent."
                    )

        return conflicts
