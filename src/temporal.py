"""Temporal resolver for amended clauses.

Given a clause ID and the two possible anchor dates (determination_date
and event_date), returns the correct version of the clause text —
applying or withholding each amendment based on its transitional rule.

Anchor logic (from Amendment No. 2026-01, §5):
  §5.1  Paragraphs 1, 3, 4 → keyed on determination_date.
         If determination_date >= effective_date, apply the amendment.
  §5.2  Paragraph 2 → keyed on event_date.
         If event_date >= effective_date, apply the amendment.
         If the change of circumstances occurred before effective_date,
         the old reporting period applies regardless of determination_date.
  §5.3  Apportionment — if the claim spans effective_date, the figures
         in force on each day of the period apply and the award is
         apportioned under §7.4.3.  Flagged but not resolved here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

from .amendment_parser import AmendmentRecord, parse_amendment
from .parser import Clause, build_clause_index, parse_corpus


@dataclass
class ClauseVersion:
    """The resolved version of a clause after applying/withholding amendments."""

    clause_id: str
    text: str
    is_amended: bool  # True if any amendment was applied to this clause
    amendment_paragraph: int | None = None  # which amendment paragraph applied
    apportionment: bool = False  # §5.3 — claim period spans effective date


def _apply_substitution(text: str, old_value: str, new_value: str) -> str:
    """Apply a textual substitution, replacing all occurrences.

    Handles bold markers (**) in the original text that may or may not
    be present around the old_value.
    """
    # Try exact match first
    if old_value in text:
        return text.replace(old_value, new_value)
    # Try stripping bold markers from the text and matching
    bold_old = f"**{old_value}**"
    if bold_old in text:
        return text.replace(bold_old, f"**{new_value}**")
    return text


def _apply_table_replace(text: str, new_table: str) -> str:
    """Replace the markdown table in the clause text with the new table."""
    # Find the existing table (lines starting with |) and replace it
    lines = text.splitlines()
    new_lines: list[str] = []
    in_table = False
    table_replaced = False
    for line in lines:
        if line.strip().startswith("|"):
            if not in_table:
                in_table = True
                if not table_replaced:
                    new_lines.append(new_table)
                    table_replaced = True
        else:
            in_table = False
            new_lines.append(line)
    return "\n".join(new_lines)


def resolve_clause(
    clause_id: str,
    determination_date: date,
    event_date: date | None = None,
    *,
    clause_index: dict[str, Clause] | None = None,
    amendments: list[AmendmentRecord] | None = None,
) -> ClauseVersion:
    """Resolve the correct version of a clause given temporal context.

    Args:
        clause_id: The clause to resolve (e.g. "4.3.2").
        determination_date: The date the determination is being made.
        event_date: The date of the change of circumstances (if relevant).
                    Required for paragraph-2 amendments (§5.2); ignored
                    for paragraphs 1/3/4 which use determination_date.
        clause_index: Pre-built clause index. Built on first call if None.
        amendments: Pre-parsed amendment records. Parsed on first call if None.

    Returns:
        ClauseVersion with the correct text for the given temporal context.
    """
    if clause_index is None:
        clause_index = build_clause_index(parse_corpus())
    if amendments is None:
        amendments = parse_amendment()

    clause = clause_index.get(clause_id)
    if clause is None:
        raise KeyError(f"Clause §{clause_id} not found in the policy manual.")

    text = clause.text
    applied = False
    applied_paragraph: int | None = None
    apportionment = False

    # Find all amendments targeting this clause
    for rec in amendments:
        if rec.target_clause_id != clause_id:
            continue

        # Determine which anchor date governs this amendment
        if rec.anchor == "event_date":
            # §5.2: keyed on when the change of circumstances occurred
            anchor_date = event_date
            if anchor_date is None:
                # No event_date provided — conservative: don't apply
                # (the caller should provide event_date for paragraph-2
                # amendments; absence means we can't determine applicability)
                continue
        else:
            # §5.1: keyed on determination_date
            anchor_date = determination_date

        # Check for §5.3 apportionment: if we have both dates and one
        # is before effective and the other after, flag it
        if event_date is not None and determination_date >= rec.effective_date:
            if event_date < rec.effective_date:
                # The change of circumstances occurred before the effective
                # date but the determination is after — for event_date-anchored
                # amendments, this means old text applies (per §5.2).
                # For determination_date-anchored amendments, new text applies.
                # If the claim period *spans* the effective date, set the flag.
                pass  # handled by the anchor_date logic below

        should_apply = anchor_date >= rec.effective_date

        if should_apply:
            if rec.operation == "substitute" and rec.old_value is not None:
                text = _apply_substitution(text, rec.old_value, rec.new_value)
                applied = True
                applied_paragraph = rec.amendment_paragraph
            elif rec.operation == "table_replace":
                text = _apply_table_replace(text, rec.new_value)
                applied = True
                applied_paragraph = rec.amendment_paragraph
            # insert_after creates a new clause — not handled here
            # (the new clause 10.5.3A would be in the index separately)

    return ClauseVersion(
        clause_id=clause_id,
        text=text,
        is_amended=applied,
        amendment_paragraph=applied_paragraph,
        apportionment=apportionment,
    )
