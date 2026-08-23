"""Corpus parser for the Calder County HSP policy manual.

Reads policy-manual.md and chunks it by §Part.Section.Paragraph numbering.
Each bold-numbered paragraph becomes one Clause.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# Matches lines like **1.2.3** or **1.2.3 Some Title**
_CLAUSE_RE = re.compile(r"^\*\*(\d+\.\d+\.\d+[A-Za-z]?)(?:\s+.*)?\*\*")

# Matches cross-references like §4.3.2 or §4.3
_XREF_RE = re.compile(r"§(\d+\.\d+(?:\.\d+[A-Za-z]?)?)")


@dataclass
class Clause:
    """A single numbered clause from the policy manual."""

    clause_id: str  # e.g. "4.3.2"
    text: str  # full paragraph text including the bold number
    cross_references: list[str] = field(default_factory=list)

    def part(self) -> int:
        """Return the Part number (first segment of the clause id)."""
        return int(self.clause_id.split(".")[0])


def parse_corpus(path: str | Path | None = None) -> list[Clause]:
    """Parse the policy manual into a list of Clause objects.

    Args:
        path: Path to policy-manual.md.  Defaults to the standard location
              relative to the project root (``1/Data pack/policy-manual.md``).

    Returns:
        Ordered list of every numbered clause found in the document.
    """
    if path is None:
        # Resolve relative to this file's grandparent (project root)
        path = Path(__file__).resolve().parent.parent / "1" / "Data pack" / "policy-manual.md"
    else:
        path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Corpus not found at {path}.  "
            "The corpus must already exist — do not generate a substitute."
        )

    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    clauses: list[Clause] = []
    current_id: str | None = None
    current_lines: list[str] = []

    def _flush() -> None:
        nonlocal current_id, current_lines
        if current_id is not None:
            body = "\n".join(current_lines).strip()
            xrefs = _XREF_RE.findall(body)
            # Remove self-references
            xrefs = [x for x in xrefs if x != current_id]
            clauses.append(Clause(clause_id=current_id, text=body, cross_references=xrefs))
        current_id = None
        current_lines = []

    for line in lines:
        m = _CLAUSE_RE.match(line.strip())
        if m:
            _flush()
            current_id = m.group(1)
            current_lines = [line]
        elif current_id is not None:
            current_lines.append(line)

    _flush()  # last clause
    return clauses


def build_clause_index(clauses: list[Clause]) -> dict[str, Clause]:
    """Return a dict mapping clause_id -> Clause for O(1) lookup."""
    clause_index = {}
    for c in clauses:
        clause_id = c.clause_id
        if clause_id in clause_index:
            import warnings
            warnings.warn(
                f"Duplicate clause ID '{clause_id}' found in manual. "
                f"First occurrence retained. Check the manual for errors.",
                UserWarning,
                stacklevel=2
            )
        else:
            clause_index[clause_id] = c
    return clause_index


# Matches lines like "# Part 3 — Residence" or "## Part 3 — Residence"
_PART_HEADING_RE = re.compile(r"^#{1,2}\s+(Part\s+(\d+)[^\n]*)", re.MULTILINE)


def parse_part_headings(path: str | Path | None = None) -> dict[int, str]:
    """Return a mapping of part number -> heading string.

    Example: {1: "Part 1 — Scope and Definitions", 2: "Part 2 — General Conditions of Eligibility", ...}
    """
    if path is None:
        path = Path(__file__).resolve().parent.parent / "1" / "Data pack" / "policy-manual.md"
    else:
        path = Path(path)

    if not path.exists():
        return {}

    text = path.read_text(encoding="utf-8")
    headings: dict[int, str] = {}
    for m in _PART_HEADING_RE.finditer(text):
        heading_text = m.group(1).strip()
        part_num = int(m.group(2))
        if part_num not in headings:
            headings[part_num] = heading_text
    return headings
