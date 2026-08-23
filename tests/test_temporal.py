"""Unit tests for the temporal resolver and amendment parser.

Key test cases from the user requirements:
  1. Determination in April 2026 for a February 2026 change of circumstances:
     → §4.3.2 and §9.1.4 should return OLD text (10/30-day), not the new
       14-day text, because event_date < 1 March 2026 (§5.2).
  2. Straightforward post-March determination for everything else:
     → §6.4.1 should return $175 (new), §10.5.2 should return 15% (new),
       because determination_date >= 1 March 2026 (§5.1).
  3. Pre-March determination → old values for all clauses.
  4. Amendment parser extracts correct number and type of records.
"""

import sys
import unittest
from datetime import date
from pathlib import Path

# Ensure the project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.amendment_parser import parse_amendment
from src.parser import build_clause_index, parse_corpus
from src.temporal import ClauseVersion, resolve_clause


class TestAmendmentParser(unittest.TestCase):
    """Test that the amendment file is parsed into correct records."""

    def setUp(self) -> None:
        self.records = parse_amendment()

    def test_record_count(self) -> None:
        """Amendment 2026-01 should produce 6 records."""
        self.assertEqual(len(self.records), 6)

    def test_paragraph_1_earnings_disregard(self) -> None:
        """Paragraph 1: §6.4.1 substitute $120→$175, anchor=determination_date."""
        rec = [r for r in self.records if r.target_clause_id == "6.4.1"][0]
        self.assertEqual(rec.amendment_paragraph, 1)
        self.assertEqual(rec.operation, "substitute")
        self.assertEqual(rec.old_value, "$120 per month")
        self.assertEqual(rec.new_value, "$175 per month")
        self.assertEqual(rec.anchor, "determination_date")

    def test_paragraph_2_reporting_period(self) -> None:
        """Paragraph 2: §4.3.2 and §9.1.4 both use event_date anchor."""
        p2 = [r for r in self.records if r.amendment_paragraph == 2]
        self.assertEqual(len(p2), 2)
        for rec in p2:
            self.assertEqual(rec.anchor, "event_date")
        cids = {r.target_clause_id for r in p2}
        self.assertEqual(cids, {"4.3.2", "9.1.4"})

    def test_paragraph_3_table_replace(self) -> None:
        """Paragraph 3: §6.6.1 table replacement, anchor=determination_date."""
        rec = [r for r in self.records if r.target_clause_id == "6.6.1"][0]
        self.assertEqual(rec.operation, "table_replace")
        self.assertIn("$1,225", rec.new_value)
        self.assertEqual(rec.anchor, "determination_date")

    def test_paragraph_4_sanctions(self) -> None:
        """Paragraph 4: §10.5.2 substitute + §10.5.3A insert, anchor=determination_date."""
        p4 = [r for r in self.records if r.amendment_paragraph == 4]
        self.assertEqual(len(p4), 2)
        sub = [r for r in p4 if r.operation == "substitute"][0]
        self.assertEqual(sub.target_clause_id, "10.5.2")
        self.assertEqual(sub.old_value, "20 per cent")
        self.assertEqual(sub.new_value, "15 per cent")
        ins = [r for r in p4 if r.operation == "insert_after"][0]
        self.assertEqual(ins.target_clause_id, "10.5.3A")

    def test_effective_date(self) -> None:
        """All records should have effective_date = 1 March 2026."""
        for rec in self.records:
            self.assertEqual(rec.effective_date, date(2026, 3, 1))


class TestTemporalResolver(unittest.TestCase):
    """Test the temporal resolver returns correct clause versions."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.clause_index = build_clause_index(parse_corpus())
        cls.amendments = parse_amendment()

    def _resolve(self, clause_id: str, det: date, evt: date | None = None) -> ClauseVersion:
        return resolve_clause(
            clause_id,
            determination_date=det,
            event_date=evt,
            clause_index=self.clause_index,
            amendments=self.amendments,
        )

    # ── KEY TEST 1: April 2026 determination, February 2026 event ────────
    # §5.2: event_date < 1 March 2026 → OLD reporting periods apply

    def test_432_feb_event_returns_old_10_days(self) -> None:
        """§4.3.2 with Feb 2026 change of circumstances: should still say '10 calendar days'."""
        cv = self._resolve("4.3.2", det=date(2026, 4, 15), evt=date(2026, 2, 10))
        self.assertIn("10 calendar days", cv.text)
        self.assertNotIn("14 calendar days", cv.text)
        self.assertFalse(cv.is_amended)

    def test_914_feb_event_returns_old_30_days(self) -> None:
        """§9.1.4 with Feb 2026 change of circumstances: should still say '30 calendar days'."""
        cv = self._resolve("9.1.4", det=date(2026, 4, 15), evt=date(2026, 2, 10))
        self.assertIn("30 calendar days", cv.text)
        self.assertNotIn("14 calendar days", cv.text)
        self.assertFalse(cv.is_amended)

    # ── KEY TEST 2: Post-March determination for determination_date clauses ─
    # §5.1: determination_date >= 1 March 2026 → new values apply

    def test_641_post_march_returns_new_175(self) -> None:
        """§6.4.1 with April determination: should say '$175 per month'."""
        cv = self._resolve("6.4.1", det=date(2026, 4, 15))
        self.assertIn("$175 per month", cv.text)
        self.assertNotIn("$120 per month", cv.text)
        self.assertTrue(cv.is_amended)

    def test_1052_post_march_returns_new_15_pct(self) -> None:
        """§10.5.2 with April determination: should say '15 per cent'."""
        cv = self._resolve("10.5.2", det=date(2026, 4, 15))
        self.assertIn("15 per cent", cv.text)
        self.assertNotIn("20 per cent", cv.text)
        self.assertTrue(cv.is_amended)

    def test_661_post_march_returns_new_thresholds(self) -> None:
        """§6.6.1 with April determination: table should have $1,225."""
        cv = self._resolve("6.6.1", det=date(2026, 4, 15))
        self.assertIn("$1,225", cv.text)
        self.assertNotIn("$1,180", cv.text)
        self.assertTrue(cv.is_amended)

    # ── Pre-March determination → all old values ─────────────────────────

    def test_641_pre_march_returns_old_120(self) -> None:
        """§6.4.1 with February determination: should still say '$120 per month'."""
        cv = self._resolve("6.4.1", det=date(2026, 2, 15))
        self.assertIn("$120 per month", cv.text)
        self.assertNotIn("$175 per month", cv.text)
        self.assertFalse(cv.is_amended)

    def test_1052_pre_march_returns_old_20_pct(self) -> None:
        """§10.5.2 with February determination: should still say '20 per cent'."""
        cv = self._resolve("10.5.2", det=date(2026, 2, 15))
        self.assertIn("20 per cent", cv.text)
        self.assertNotIn("15 per cent", cv.text)
        self.assertFalse(cv.is_amended)

    # ── Post-March event for paragraph-2 clauses → new values ────────────

    def test_432_march_event_returns_new_14_days(self) -> None:
        """§4.3.2 with March 2026 change of circumstances: should say '14 calendar days'."""
        cv = self._resolve("4.3.2", det=date(2026, 4, 15), evt=date(2026, 3, 15))
        self.assertIn("14 calendar days", cv.text)
        self.assertNotIn("10 calendar days", cv.text)
        self.assertTrue(cv.is_amended)

    def test_914_march_event_returns_new_14_days(self) -> None:
        """§9.1.4 with March 2026 change of circumstances: should say '14 calendar days'."""
        cv = self._resolve("9.1.4", det=date(2026, 4, 15), evt=date(2026, 3, 15))
        self.assertIn("14 calendar days", cv.text)
        self.assertNotIn("30 calendar days", cv.text)
        self.assertTrue(cv.is_amended)

    # ── Unamended clause passes through unchanged ────────────────────────

    def test_unamended_clause_unchanged(self) -> None:
        """§2.1.2 has no amendment — text should pass through as-is."""
        cv = self._resolve("2.1.2", det=date(2026, 4, 15))
        original = self.clause_index["2.1.2"]
        self.assertEqual(cv.text, original.text)
        self.assertFalse(cv.is_amended)

    # ── Edge: event_date exactly on effective_date ────────────────────────

    def test_432_event_exactly_on_effective_date(self) -> None:
        """§4.3.2 with event on exactly 1 March 2026: should apply new text."""
        cv = self._resolve("4.3.2", det=date(2026, 4, 15), evt=date(2026, 3, 1))
        self.assertIn("14 calendar days", cv.text)
        self.assertTrue(cv.is_amended)

    def test_641_determination_exactly_on_effective_date(self) -> None:
        """§6.4.1 with determination on exactly 1 March 2026: should apply new text."""
        cv = self._resolve("6.4.1", det=date(2026, 3, 1))
        self.assertIn("$175 per month", cv.text)
        self.assertTrue(cv.is_amended)


class TestAmbiguousResolution(unittest.TestCase):
    """Test deliberate ambiguous=True behavior when event_date is not supplied
    for event_date-anchored clauses (§4.3.2, §9.1.4).

    Design decision: rather than silently applying or withholding the amendment,
    the resolver returns ambiguous=True with BOTH versions — text (new) and
    ambiguous_old_text (old) — so the caller can ask for event_date and call
    again, or present the caseworker with both options.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.clause_index = build_clause_index(parse_corpus())
        cls.amendments = parse_amendment()

    def _resolve(self, clause_id: str, det: date, evt: date | None = None) -> ClauseVersion:
        return resolve_clause(
            clause_id,
            determination_date=det,
            event_date=evt,
            clause_index=self.clause_index,
            amendments=self.amendments,
        )

    def test_432_no_event_date_is_ambiguous(self) -> None:
        """§4.3.2 with no event_date: result must be ambiguous=True."""
        cv = self._resolve("4.3.2", det=date(2026, 4, 15), evt=None)
        self.assertTrue(cv.ambiguous, "Expected ambiguous=True when event_date is missing")

    def test_432_no_event_date_text_is_new_version(self) -> None:
        """§4.3.2 ambiguous: text field holds the NEW (post-amendment) version."""
        cv = self._resolve("4.3.2", det=date(2026, 4, 15), evt=None)
        self.assertIn("14 calendar days", cv.text)

    def test_432_no_event_date_old_text_is_old_version(self) -> None:
        """§4.3.2 ambiguous: ambiguous_old_text holds the OLD (pre-amendment) version."""
        cv = self._resolve("4.3.2", det=date(2026, 4, 15), evt=None)
        self.assertIsNotNone(cv.ambiguous_old_text)
        self.assertIn("10 calendar days", cv.ambiguous_old_text)

    def test_432_no_event_date_is_not_definitively_amended(self) -> None:
        """§4.3.2 ambiguous: is_amended must be False — no amendment was definitively applied."""
        cv = self._resolve("4.3.2", det=date(2026, 4, 15), evt=None)
        self.assertFalse(cv.is_amended)

    def test_914_no_event_date_is_ambiguous(self) -> None:
        """§9.1.4 with no event_date: result must be ambiguous=True."""
        cv = self._resolve("9.1.4", det=date(2026, 4, 15), evt=None)
        self.assertTrue(cv.ambiguous)
        self.assertIn("30 calendar days", cv.ambiguous_old_text)
        self.assertIn("14 calendar days", cv.text)
        self.assertFalse(cv.is_amended)

    def test_non_event_date_clause_not_ambiguous(self) -> None:
        """§6.4.1 (determination_date anchor) is never ambiguous — event_date irrelevant."""
        cv = self._resolve("6.4.1", det=date(2026, 4, 15), evt=None)
        self.assertFalse(cv.ambiguous)
        self.assertIsNone(cv.ambiguous_old_text)


class TestInsertedClause(unittest.TestCase):
    """Test resolve_clause for §10.5.3A — a clause inserted by the amendment
    that does not exist in the base corpus.

    Pre-effective determination_date → exists=False, text=None.
    Post-effective determination_date → exists=True, full inserted text.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.clause_index = build_clause_index(parse_corpus())
        cls.amendments = parse_amendment()

    def _resolve(self, clause_id: str, det: date) -> ClauseVersion:
        return resolve_clause(
            clause_id,
            determination_date=det,
            clause_index=self.clause_index,
            amendments=self.amendments,
        )

    def test_10503a_pre_march_does_not_exist(self) -> None:
        """§10.5.3A with February determination: clause not yet in force, exists=False."""
        cv = self._resolve("10.5.3A", det=date(2026, 2, 15))
        self.assertFalse(cv.exists)
        self.assertIsNone(cv.text)
        self.assertFalse(cv.is_amended)

    def test_10503a_post_march_exists_with_text(self) -> None:
        """§10.5.3A with April determination: clause in force, text returned."""
        cv = self._resolve("10.5.3A", det=date(2026, 4, 15))
        self.assertTrue(cv.exists)
        self.assertIsNotNone(cv.text)
        self.assertIn("10.5.3A", cv.text)
        self.assertIn("failure to report", cv.text)
        self.assertIn("increased the award", cv.text)
        self.assertTrue(cv.is_amended)

    def test_10503a_exactly_on_effective_date_exists(self) -> None:
        """§10.5.3A with determination exactly on 1 March 2026: should exist."""
        cv = self._resolve("10.5.3A", det=date(2026, 3, 1))
        self.assertTrue(cv.exists)
        self.assertIsNotNone(cv.text)

    def test_genuinely_missing_clause_raises(self) -> None:
        """A clause ID that doesn't exist at all should raise KeyError."""
        with self.assertRaises(KeyError):
            resolve_clause(
                "99.9.9",
                determination_date=date(2026, 4, 15),
                clause_index=self.clause_index,
                amendments=self.amendments,
            )


if __name__ == "__main__":
    unittest.main()
