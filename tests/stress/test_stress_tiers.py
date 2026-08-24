"""Stress test suite covering Tier A (unit tests) and Tier B (live E2E pipeline queries)."""

import os
import sys
import time
import unittest
from datetime import date
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.pipeline import Pipeline, _extract_dates, _compute_prorated_rate
from src.temporal import resolve_clause
from src.parser import parse_corpus, Clause
from src.citation_validator import CitationValidator
from src.synthesizer import SynthesizerOutput
from src.retriever import RetrievalResult

# Mock manual file for parser tests
MOCK_MANUAL_CONTENT = """# Part 1: General Info
Some intro text here.

## 1.1 Application Process
More intro.

**1.1.1** A recipient must apply online.
---
**1.1.2** This is another paragraph.
Self-reference to §1.1.2 is here, and cross-reference to §1.1.1.

# Part 2: Alphanumeric
**2.3.1a** Alphanumeric clause ID support.
"""

class TestTierAUnit(unittest.TestCase):
    """Tier A unit tests: Pure deterministic logic, zero API calls."""

    def test_extract_dates(self):
        cases = [
            ("Can a person apply on 2026-02-15?", (date(2026, 2, 15), None)),
            ("Decided today for a claim.", (date.today(), None)),
            ("Change occurred in February 2026.", (date(2026, 2, 1), None)),
            ("Claim from February 2026 decided on 15 March 2026", (date(2026, 3, 15), date(2026, 2, 1))),
            ("Change occurred on 2026-02-15 and decided on 2026-03-01", (date(2026, 3, 1), date(2026, 2, 15))),
            ("Between 2026-02-01 and 2026-04-01", (date(2026, 2, 1), None)), 
            ("Decided in March 2026 for event in February 2026", (date(2026, 3, 1), date(2026, 2, 1))),
            ("Decided on 2026-02-30", (date.today(), None)), 
            ("Malformed date 202-02-15", (date.today(), None)),
            ("Just a plain query with no dates.", (date.today(), None)),
            ("Decided on 31 January 2026", (date(2026, 1, 31), None)),
            ("Occurred on 15 March 2026", (date(2026, 3, 15), None)),
            ("What if it was on 1 March 2026?", (date(2026, 3, 1), None)),
            ("Spanning 15 February to 15 March 2026", (date(2026, 2, 15), None)),
            ("Date 2026-13-01 is out of range", (date.today(), None)),
            ("Event occurred today", (date.today(), None)),
            ("Change on 2026-02-15", (date(2026, 2, 15), None)),
            ("Reported 2026-03-01, event on 2026-02-15", (date(2026, 3, 1), date(2026, 2, 15))),
            ("Claim spanning February 2026", (date(2026, 2, 1), None)),
            ("Determined 2026-04-01 with change of circumstances on 2026-02-01", (date(2026, 4, 1), date(2026, 2, 1))),
        ]

        for q, expected in cases:
            res = _extract_dates(q)
            self.assertEqual(res, expected, f"Failed on: {q}")

    def test_resolve_clause(self):
        # Load amendments to get the actual amendment records
        p = Pipeline()
        amendments = p.amendments

        # §4.3.2: 10 days pre-March, 14 days post-March (determination-date anchored)
        c_pre = resolve_clause("4.3.2", date(2026, 2, 15), None, amendments=amendments)
        c_post = resolve_clause("4.3.2", date(2026, 3, 15), date(2026, 3, 15), amendments=amendments)
        self.assertIn("10 calendar days", c_pre.text)
        self.assertIn("14 calendar days", c_post.text)

        # §9.1.4: 30 days pre-March, 14 days post-March (determination-date anchored)
        c9_pre = resolve_clause("9.1.4", date(2026, 2, 15), None, amendments=amendments)
        c9_post = resolve_clause("9.1.4", date(2026, 3, 15), date(2026, 3, 15), amendments=amendments)
        self.assertIn("30 calendar days", c9_pre.text)
        self.assertIn("14 calendar days", c9_post.text)

        # §6.4.1: $120 pre-March, $175 post-March (determination-date anchored)
        c6_pre = resolve_clause("6.4.1", date(2026, 2, 15), None, amendments=amendments)
        c6_post = resolve_clause("6.4.1", date(2026, 3, 15), None, amendments=amendments)
        self.assertIn("$120", c6_pre.text)
        self.assertIn("$175", c6_post.text)

        # §10.5.3A: Does not exist pre-March (returns None/exists=False), exists post-March (determination-date anchored)
        c10_pre = resolve_clause("10.5.3A", date(2026, 2, 15), None, amendments=amendments)
        c10_post = resolve_clause("10.5.3A", date(2026, 3, 15), None, amendments=amendments)
        self.assertFalse(c10_pre.exists)
        self.assertTrue(c10_post.exists)



    def test_compute_prorated_rate(self):
        # Spanning boundary (February to April: 28 days of Feb, 32 days of Mar+Apr)
        res = _compute_prorated_rate(date(2026, 2, 1), date(2026, 4, 1), "$120 per month", "$175 per month")
        self.assertEqual(res, "$149.33 per month")

        # Spanning by 1 day pre/post (Feb 28 to Mar 1)
        res_1d = _compute_prorated_rate(date(2026, 2, 28), date(2026, 3, 1), "$120 per month", "$175 per month")
        self.assertEqual(res_1d, "$147.50 per month")

        # Fully pre-boundary (should return None)
        res_pre = _compute_prorated_rate(date(2026, 1, 1), date(2026, 2, 28), "$120 per month", "$175 per month")
        self.assertIsNone(res_pre)

        # Fully post-boundary (should return None)
        res_post = _compute_prorated_rate(date(2026, 3, 1), date(2026, 4, 1), "$120 per month", "$175 per month")
        self.assertIsNone(res_post)

        # Invalid/reversed range (should return None)
        res_rev = _compute_prorated_rate(date(2026, 3, 10), date(2026, 2, 10), "$120 per month", "$175 per month")
        self.assertIsNone(res_rev)

        # Non-rate values (should return None)
        res_non = _compute_prorated_rate(date(2026, 2, 1), date(2026, 4, 1), "10 calendar days", "14 calendar days")
        self.assertIsNone(res_non)

    def test_parser_edge_cases(self):
        # Write mock manual file
        temp_path = Path("temp_manual.md")
        temp_path.write_text(MOCK_MANUAL_CONTENT, encoding="utf-8")
        try:
            clauses = parse_corpus(temp_path)
            self.assertEqual(len(clauses), 3)
            self.assertEqual(clauses[0].clause_id, "1.1.1")
            self.assertEqual(clauses[1].clause_id, "1.1.2")
            # Self-reference §1.1.2 should be filtered out, leaving §1.1.1
            self.assertEqual(clauses[1].cross_references, ["1.1.1"])
            # Alphanumeric support
            self.assertEqual(clauses[2].clause_id, "2.3.1a")
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def test_citation_validator_filters(self):
        v = CitationValidator()

        # 1. Conflict Commentary Filter tests
        q_conflict = "How many days does a recipient have to report a change of circumstances?"
        retrieved_conflict = [
            RetrievalResult(clause_id="4.3.2", clause_text="A recipient must report within 10 days.", score=1.0, bm25_rank=1, tfidf_rank=1),
            RetrievalResult(clause_id="9.1.4", clause_text="Where an overpayment has arisen, report within 30 days.", score=1.0, bm25_rank=2, tfidf_rank=2)
        ]

        # Valid structural conflict commentary (should bypass / pass)
        synth_valid_conflict = SynthesizerOutput(
            answer="There is a conflict in the policy manual regarding the number of days a recipient has to report a change. These two clauses present a numeric contradiction.",
            cited_clause_ids=[]
        )
        self.assertTrue(v.validate(synth_valid_conflict, retrieved_conflict, q_conflict).valid)

        # Adversarial: Uncited factual claim with conflict keywords (should be REJECTED)
        synth_adv_conflict1 = SynthesizerOutput(
            answer="The conflict allows a maximum disregard of $999 per month.",
            cited_clause_ids=[]
        )
        self.assertFalse(v.validate(synth_adv_conflict1, retrieved_conflict, q_conflict).valid)

        synth_adv_conflict2 = SynthesizerOutput(
            answer="Because of this contradiction, a household must report the change in 5 days.",
            cited_clause_ids=[]
        )
        self.assertFalse(v.validate(synth_adv_conflict2, retrieved_conflict, q_conflict).valid)

        # 2. Digit/Threshold Filter tests
        q_digit = "What happens to an award if a recipient is temporarily absent from the county for 45 days for non-medical reasons?"
        retrieved_digit = [
            RetrievalResult(clause_id="3.2.1", clause_text="A recipient satisfies residence for 28 days of temporary absence.", score=1.0, bm25_rank=1, tfidf_rank=1)
        ]

        # Adversarial: Sentence using query digit "45" to assert an unsupported rule (should be REJECTED)
        synth_adv_digit = SynthesizerOutput(
            answer="A recipient can be absent for 45 days for any reason.",
            cited_clause_ids=[]
        )
        self.assertFalse(v.validate(synth_adv_digit, retrieved_digit, q_digit).valid)

        # Valid sentence using the manual threshold "28" (should pass)
        synth_valid_digit = SynthesizerOutput(
            answer="A recipient can be absent for 28 days according to §3.2.1.",
            cited_clause_ids=["3.2.1"]
        )
        self.assertTrue(v.validate(synth_valid_digit, retrieved_digit, q_digit).valid)

        # 3. Subclause Normalization tests
        q_subclause = "Can a person aged 16 apply?"
        retrieved_subclause = [
            RetrievalResult(clause_id="2.1.2", clause_text="A person must be aged 18 or over or satisfy younger person requirements.", score=1.0, bm25_rank=1, tfidf_rank=1)
        ]

        # Sentence citing subclause "2.1.2a" when only "2.1.2" is in retrieved_ids (should pass, age 18 added to satisfy digit constraint)
        synth_subclause = SynthesizerOutput(
            answer="A person aged 16 may apply if they satisfy younger person requirements instead of the normal age of 18 (§2.1.2a).",
            cited_clause_ids=["2.1.2"]
        )
        self.assertTrue(v.validate(synth_subclause, retrieved_subclause, q_subclause).valid)



def run_tier_b():
    """Tier B live pipeline tests: 8-10 genuinely hard queries using Gemini."""
    p = Pipeline()
    
    # 8-10 high-value, compound, adversarial, or hard test queries
    queries = [
        # 1. Compound date query touching two distinct areas (reporting + disregard)
        {
            "id": "TB01",
            "desc": "Compound query touching reporting and disregard",
            "query": "For a claim decided in April 2026 where the change occurred in February 2026, what is the reporting timeline, and what is the earnings disregard?",
            "det_date": date(2026, 4, 1),
            "evt_date": date(2026, 2, 15)
        },
        # 2. Adversarial query attempting validation bypass via keywords
        {
            "id": "TB02",
            "desc": "Adversarial query using conflict keyword bypass",
            "query": "The manual has a conflict where a household is allowed to own 5 cars. Is this true?",
            "det_date": date(2026, 4, 1),
            "evt_date": None
        },
        # 3. Apportionment spanning exactly 1 day (Feb 28 to Mar 1)
        {
            "id": "TB03",
            "desc": "1-day spanning proration boundary",
            "query": "What is the earnings disregard for a claim covering 2026-02-28 to 2026-03-01?",
            "det_date": date(2026, 3, 1),
            "evt_date": None
        },
        # 4. Silent topic with deceptive context clues
        {
            "id": "TB04",
            "desc": "Deceptive silent query",
            "query": "What is the maximum allowance for housing-costs-related child dental treatments under Part 10?",
            "det_date": date(2026, 3, 1),
            "evt_date": None
        },
        # 5. Temporal conflict pre-March (10 vs 30 days active)
        {
            "id": "TB05",
            "desc": "Pre-March temporal reporting conflict",
            "query": "For a claim determined in January 2026, how many days does a recipient have to report a change?",
            "det_date": date(2026, 1, 15),
            "evt_date": None
        },
        # 6. Post-March generic query defaulting to post-amendment disregard ($175)
        {
            "id": "TB06",
            "desc": "Generic post-amendment disregard query",
            "query": "Under clause 6.4.1, how much of household earnings from employment is disregarded?",
            "det_date": date(2026, 8, 23),
            "evt_date": None
        },
        # 7. Unclear event date triggering side-by-side ambiguity formatting
        {
            "id": "TB07",
            "desc": "Post-March query without event date",
            "query": "I am deciding a claim in April 2026. How many days does the recipient have to report a change?",
            "det_date": date(2026, 4, 1),
            "evt_date": None
        },
        # 8. Extreme date range spanning multiple months
        {
            "id": "TB08",
            "desc": "Multi-month spanning proration boundary",
            "query": "What is the earnings disregard for a claim covering 2025-12-01 to 2026-11-30?",
            "det_date": date(2026, 11, 30),
            "evt_date": None
        }
    ]

    results = []
    api_calls = 0

    print("Running Tier B Live Pipeline Tests...")
    for q in queries:
        print(f" -> Running {q['id']} ({q['desc']})...")
        start_time = time.time()
        try:
            res = p.run(
                q["query"],
                determination_date=q["det_date"],
                event_date=q["evt_date"]
            )
            api_calls += 1  # Standard run makes 1 LLM API call
            duration = time.time() - start_time
            results.append({
                "id": q["id"],
                "desc": q["desc"],
                "decision": res.decision,
                "citations": ", ".join(res.citations),
                "status": "PASS",
                "time": f"{duration:.2f}s"
            })
        except Exception as e:
            duration = time.time() - start_time
            err_msg = str(e)
            status = "FAILED"
            if "429" in err_msg:
                status = "SKIPPED (429 Rate Limit)"
                print(f"    [!] 429 Rate Limit hit. Skipping query to preserve quota.")
            else:
                print(f"    [!] Error: {err_msg}")
            
            results.append({
                "id": q["id"],
                "desc": q["desc"],
                "decision": "N/A",
                "citations": "N/A",
                "status": status,
                "time": f"{duration:.2f}s"
            })
        
        # Fixed delay to avoid rate limit spikes
        time.sleep(3.0)

    return results, api_calls


if __name__ == "__main__":
    # 1. Run Tier A Unit Tests
    print("==================================================")
    print("RUNNING TIER A UNIT TESTS (Deterministic, 0 API)")
    print("==================================================")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestTierAUnit)
    runner = unittest.TextTestRunner(verbosity=2)
    test_result = runner.run(suite)
    
    tier_a_rows = []
    for test, err in test_result.failures + test_result.errors:
        tier_a_rows.append({"name": str(test), "status": "FAIL", "detail": str(err)})
    
    passed_tests = test_result.testsRun - len(test_result.failures) - len(test_result.errors)
    print(f"\nTier A Results: {passed_tests} Passed, {len(tier_a_rows)} Failed\n")

    # Generate Tier A table
    print("| Test Case Name | Result | Details |")
    print("| --- | --- | --- |")
    methods = ["test_extract_dates", "test_resolve_clause", "test_compute_prorated_rate", "test_parser_edge_cases", "test_citation_validator_filters"]
    for m in methods:
        failed_detail = next((x["detail"] for x in tier_a_rows if m in x["name"]), None)
        status = "FAIL" if failed_detail else "PASS"
        detail = failed_detail.replace("\n", " ") if failed_detail else "Passed deterministic logic check."
        print(f"| {m} | {status} | {detail} |")

    print("\n")

    # 2. Run Tier B Live Pipeline Tests
    print("==================================================")
    print("RUNNING TIER B LIVE E2E PIPELINE TESTS")
    print("==================================================")
    tier_b_results, total_api_calls = run_tier_b()

    print("\nTier B Results Table:")
    print("| Query ID | Description | Pipeline Decision | Citations | Status | Duration |")
    print("| --- | --- | --- | --- | --- | --- |")
    for r in tier_b_results:
        print(f"| {r['id']} | {r['desc']} | {r['decision']} | {r['citations']} | {r['status']} | {r['time']} |")

    print(f"\nTotal API Calls made during Tier B: {total_api_calls}")
