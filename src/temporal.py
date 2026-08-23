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
from dataclasses import dataclass, field
from datetime import date

from .amendment_parser import AmendmentRecord, parse_amendment
from .parser import Clause, build_clause_index, parse_corpus


@dataclass
class ClauseVersion:
    """The resolved version of a clause after applying/withholding amendments.

    Fields
    ------
    clause_id : str
        The clause identifier.
    text : str | None
        The resolved clause text.  None in two cases:
          • exists=False — clause not yet in force (e.g. §10.5.3A pre-March).
          • ambiguous=True — event_date was not supplied for an event_date-
            anchored amendment; both versions are in old_text / new_text
            instead.  text is explicitly set to None so that any caller that
            does ``cv.text.replace(...)`` gets AttributeError on None rather
            than silently operating on whichever version happened to be stored.
    is_amended : bool
        True if at least one amendment was definitively applied (not ambiguous).
    amendment_paragraph : int | None
        Which amendment paragraph was applied (if any).
    apportionment : bool
        §5.3 flag — the claim period spans the amendment's effective date;
        apportioned under §7.4.3.  Flagged but not resolved here.
    ambiguous : bool
        True when the clause has an event_date-anchored amendment (§5.2) but
        no event_date was supplied.  When True:
          • text is None — do NOT use it.
          • old_text holds the pre-amendment text.
          • new_text holds the post-amendment text.
        The caller must obtain event_date and call resolve_clause again, or
        present both versions to the caseworker.
    old_text : str | None
        Pre-amendment clause text, populated only when ambiguous=True.
    new_text : str | None
        Post-amendment clause text, populated only when ambiguous=True.
    exists : bool
        False when the clause ID belongs to a provision inserted by an
        amendment that has not yet taken effect for the given
        determination_date (e.g. §10.5.3A before 1 March 2026).
        When False, text is None.
    """

    clause_id: str
    text: str | None
    is_amended: bool
    amendment_paragraph: int | None = None
    apportionment: bool = False
    ambiguous: bool = False
    old_text: str | None = None   # pre-amendment version, populated when ambiguous=True
    new_text: str | None = None   # post-amendment version, populated when ambiguous=True
    exists: bool = True



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
        event_date: The date of the change of circumstances (§5.2 anchor).
            Required to unambiguously resolve paragraph-2 amendments
            (§4.3.2, §9.1.4).  If None and the clause has an event_date-
            anchored amendment, the result is marked ambiguous=True and
            both the old text (ambiguous_old_text) and new text (text) are
            returned — the caller must obtain event_date and call again,
            or ask the user which version applies.  event_date is ignored
            for paragraphs 1/3/4 which are keyed on determination_date.
        clause_index: Pre-built clause index (optional; built on call if None).
        amendments: Pre-parsed amendment records (optional; parsed if None).

    Returns:
        ClauseVersion with the resolved text, flags, and both versions when
        the result is ambiguous.

    Raises:
        KeyError: If clause_id is not in the base corpus AND has no
            insert_after amendment (i.e. it genuinely does not exist).
    """
    if clause_index is None:
        clause_index = build_clause_index(parse_corpus())
    if amendments is None:
        amendments = parse_amendment()

    # ── Handle inserted clauses (e.g. §10.5.3A) ─────────────────────────
    # These don't exist in the base corpus; their text comes from the
    # insert_after amendment record.
    insert_rec = next(
        (r for r in amendments
         if r.target_clause_id == clause_id and r.operation == "insert_after"),
        None,
    )
    if insert_rec is not None:
        if determination_date >= insert_rec.effective_date:
            return ClauseVersion(
                clause_id=clause_id,
                text=insert_rec.new_value,
                is_amended=True,
                amendment_paragraph=insert_rec.amendment_paragraph,
                exists=True,
            )
        else:
            # Clause not yet in force for this determination_date
            return ClauseVersion(
                clause_id=clause_id,
                text=None,
                is_amended=False,
                exists=False,
            )

    if clause_id not in clause_index:
        raise KeyError(f"Clause §{clause_id} not found in the policy manual.")

    # ── Standard path: clause exists in base corpus ───────────────────────
    clause = clause_index[clause_id]
    text = clause.text
    applied = False
    applied_paragraph: int | None = None
    ambiguous = False
    _old_text: str | None = None
    _new_text: str | None = None
    is_apportioned = False

    # Find all amendments targeting this clause
    for rec in amendments:
        if rec.target_clause_id != clause_id:
            continue

        if rec.anchor == "event_date":
            # §5.2: keyed on when the change of circumstances occurred.
            if determination_date < rec.effective_date:
                # If determination_date is before effective_date, the amendment is
                # not yet active, so event_date must be pre-effective as well.
                # Thus, the amendment cannot apply and there is no ambiguity.
                anchor_date = determination_date
            elif event_date is None:
                # Deliberate behavior: event_date is required to resolve this
                # amendment but was not supplied.  Return both versions with
                # ambiguous=True so the caller can decide — do not silently
                # apply or withhold the amendment.
                old_text_val = text         # pre-amendment text
                new_text_val = _apply_substitution(text, rec.old_value, rec.new_value)
                ambiguous = True
                _old_text = old_text_val
                _new_text = new_text_val
                # text remains as the base for subsequent non-event_date amendments,
                # but will be set to None in the final return below.
                continue
            else:
                anchor_date = event_date
        else:
            # §5.1: keyed on determination_date
            anchor_date = determination_date

        # Check for §5.3 apportionment: if the query has both dates and one
        # is before effective and the other after, then the claim period spans
        # the effective date, meaning apportionment rules apply.
        if event_date is not None:
            d1 = min(determination_date, event_date)
            d2 = max(determination_date, event_date)
            if d1 < rec.effective_date <= d2:
                is_apportioned = True

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

    if ambiguous:
        # text=None forces callers to use old_text/new_text explicitly.
        return ClauseVersion(
            clause_id=clause_id,
            text=None,
            is_amended=False,
            amendment_paragraph=None,
            ambiguous=True,
            old_text=_old_text,
            new_text=_new_text,
            exists=True,
            apportionment=is_apportioned,
        )

    return ClauseVersion(
        clause_id=clause_id,
        text=text,
        is_amended=applied,
        amendment_paragraph=applied_paragraph,
        ambiguous=False,
        exists=True,
        apportionment=is_apportioned,
        old_text=clause.text if is_apportioned else None,
        new_text=text if is_apportioned else None,
    )
