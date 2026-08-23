"""Parser for amendment files (e.g. amendment-2026-01.md).

Extracts each numbered change as a structured record with:
- target clause ID
- operation (substitute / insert_after / table_replace)
- old value (if substitute)
- new value
- effective date
- anchor type: "determination_date" (§5.1) or "event_date" (§5.2)
- apportionment flag (§5.3)

This parser is specific to the format of Amendment No. 2026-01 but
structured so the pattern is extensible to future amendments.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path


@dataclass
class AmendmentRecord:
    """A single numbered change from an amendment file."""

    amendment_paragraph: int  # e.g. 1, 2, 3, 4
    target_clause_id: str  # e.g. "6.4.1", "4.3.2"
    operation: str  # "substitute" | "insert_after" | "table_replace"
    old_value: str | None  # text being replaced (None for insert)
    new_value: str  # replacement or inserted text
    effective_date: date  # when the amendment takes effect
    anchor: str  # "determination_date" (§5.1) or "event_date" (§5.2)
    apportionment: bool = False  # §5.3 — claim spans effective date


# ── Internal extraction helpers ──────────────────────────────────────────

# Matches "In §X.Y.Z, for "old" substitute "new""
_SUBSTITUTE_RE = re.compile(
    r'In\s+§(\d+\.\d+(?:\.\d+)?(?:\([a-z]\))?),\s+'
    r'for\s+["\u201c]([^"\u201d]+)["\u201d]'
    r'(?:\s+\(in both places where it occurs\))?\s+'
    r'substitute\s+["\u201c]\*?\*?([^"\u201d*]+)\*?\*?["\u201d]',
    re.IGNORECASE,
)

# Matches "After §X.Y.Z, insert — ..."
_INSERT_RE = re.compile(
    r'After\s+§(\d+\.\d+(?:\.\d+)?),\s+insert',
    re.IGNORECASE,
)

# Paragraph → anchor mapping from §5 transitional provisions
_PARAGRAPH_ANCHORS: dict[int, str] = {
    1: "determination_date",  # §5.1
    2: "event_date",          # §5.2
    3: "determination_date",  # §5.1
    4: "determination_date",  # §5.1
}


def _extract_table(lines: list[str], start_idx: int) -> str:
    """Extract a markdown table starting at start_idx."""
    table_lines: list[str] = []
    for line in lines[start_idx:]:
        stripped = line.strip()
        if stripped.startswith("|"):
            table_lines.append(stripped)
        elif table_lines:
            break  # end of table
    return "\n".join(table_lines)


def _extract_blockquote(lines: list[str], start_idx: int) -> str:
    """Extract a blockquote starting at start_idx."""
    bq_lines: list[str] = []
    for line in lines[start_idx:]:
        stripped = line.strip()
        if stripped.startswith(">"):
            bq_lines.append(stripped.lstrip("> ").strip())
        elif bq_lines:
            break
    return "\n".join(bq_lines)


def parse_amendment(
    path: str | Path | None = None,
) -> list[AmendmentRecord]:
    """Parse an amendment file into structured records.

    Args:
        path: Path to the amendment markdown file. Defaults to the
              standard location relative to the project root.

    Returns:
        List of AmendmentRecord objects, one per change.
    """
    if path is None:
        path = (
            Path(__file__).resolve().parent.parent
            / "1" / "Data pack" / "amendment-2026-01.md"
        )
    else:
        path = Path(path)

    if not path.exists():
        return []

    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    # Extract effective date from header
    eff_match = re.search(r"\*\*Effective:\*\*\s+(\d+\s+\w+\s+\d{4})", text)
    if eff_match:
        from datetime import datetime
        effective_date = datetime.strptime(eff_match.group(1), "%d %B %Y").date()
    else:
        effective_date = date(2026, 3, 1)  # fallback

    records: list[AmendmentRecord] = []
    current_paragraph: int = 0

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Detect paragraph headings: "## 1. Earnings disregard"
        para_match = re.match(r"^##\s+(\d+)\.\s+", stripped)
        if para_match:
            current_paragraph = int(para_match.group(1))
            continue

        # Skip transitional provisions (paragraph 5) — we use them
        # for anchor mapping but don't generate records from them
        if current_paragraph == 5:
            continue

        anchor = _PARAGRAPH_ANCHORS.get(current_paragraph, "determination_date")

        # ── Substitution: "In §X.Y.Z, for "old" substitute "new" ────────
        sub_match = _SUBSTITUTE_RE.search(stripped)
        if sub_match:
            target_raw = sub_match.group(1)
            # Strip sub-clause letter for the clause ID (e.g. "6.4.1(a)" → "6.4.1")
            target_id = re.match(r"(\d+\.\d+(?:\.\d+)?)", target_raw).group(1)
            records.append(AmendmentRecord(
                amendment_paragraph=current_paragraph,
                target_clause_id=target_id,
                operation="substitute",
                old_value=sub_match.group(2),
                new_value=sub_match.group(3),
                effective_date=effective_date,
                anchor=anchor,
            ))
            continue

        # ── Table replacement: "substitute the following —" ──────────────
        if "substitute the following" in stripped.lower():
            clause_match = re.search(r"§(\d+\.\d+(?:\.\d+)?)", stripped)
            if clause_match:
                target_id = clause_match.group(1)
                table_text = _extract_table(lines, i + 1)
                if table_text:
                    records.append(AmendmentRecord(
                        amendment_paragraph=current_paragraph,
                        target_clause_id=target_id,
                        operation="table_replace",
                        old_value=None,  # full table replacement — old value is entire existing table
                        new_value=table_text,
                        effective_date=effective_date,
                        anchor=anchor,
                    ))
            continue

        # ── Insertion: "After §X.Y.Z, insert —" ─────────────────────────
        ins_match = _INSERT_RE.search(stripped)
        if ins_match:
            target_id = ins_match.group(1)
            bq_text = _extract_blockquote(lines, i + 1)
            if bq_text:
                # Extract the new clause ID from the blockquote
                new_id_match = re.search(r"\*\*(\d+\.\d+\.\d+\w*)\*\*", bq_text)
                new_clause_id = new_id_match.group(1) if new_id_match else target_id
                records.append(AmendmentRecord(
                    amendment_paragraph=current_paragraph,
                    target_clause_id=new_clause_id,
                    operation="insert_after",
                    old_value=None,
                    new_value=bq_text,
                    effective_date=effective_date,
                    anchor=anchor,
                ))
            continue

    return records
