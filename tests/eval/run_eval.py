"""Evaluation harness for the Brite Spark 2026 RAG pipeline."""
import json
import re
import time
from datetime import date
from pathlib import Path
from src.pipeline import Pipeline

QUESTIONS = [
    {
        "id": "Q01",
        "query": "Can a person aged 16 apply for assistance?",
        "expected_decision": "ANSWER",
        "notes": "Normal eligibility question. Must cite §2.3.1. Tests basic retrieval."
    },
    {
        "id": "Q02",
        "query": "How many days does a recipient have to report a change of circumstances?",
        "expected_decision": "ANSWER",
        "notes": "Tests the current post-March rule under the system date (14 days, no conflict)."
    },
    {
        "id": "Q03",
        "query": "What is the needs figure calculation for a full-time student?",
        "expected_decision": "FLAG_CONFLICT",
        "notes": "Dead reference §7.1.3 → §5.4. Manual is silent on the actual rule."
    },
    {
        "id": "Q04",
        "query": "Can a household own a car and still be eligible for assistance?",
        "expected_decision": "ANSWER",
        "notes": "Tests resource disregards. Must cite §2.4.2(b). One vehicle per household."
    },
    {
        "id": "Q05",
        "query": "What happens to an award if a recipient is temporarily absent from the county for 45 days for non-medical reasons?",
        "expected_decision": "ANSWER",
        "notes": "Tests §3.2.1 (28-day limit) and §3.2.4 (eligibility ends day after expiry). 45 days exceeds the standard limit."
    },
    {
        "id": "Q06",
        "query": "Can a sanction be imposed on a household that includes a one-year-old child?",
        "expected_decision": "ANSWER",
        "notes": "Tests §10.5.3. Sanction must not be imposed where household includes a dependent child under age 2."
    },
    {
        "id": "Q07",
        "query": "How long does the Department have to complete a review?",
        "expected_decision": "ANSWER",
        "notes": "Tests §11.2.3. Review must be completed within 30 days of the request."
    },
    {
        "id": "Q08",
        "query": "What is the maximum deduction from an ongoing award to recover an overpayment?",
        "expected_decision": "ANSWER",
        "notes": "Tests §9.3.2. Standard 10%, up to 20% under §9.6. Must cite both."
    },
    {
        "id": "Q09",
        "query": "What support is available for recipients who want to appeal a decision about their housing costs?",
        "expected_decision": "ANSWER",
        "notes": "Apparent gap question. The manual covers appeal rights (Part 12) but says nothing specific about housing costs as a distinct category. Retriever will return appeal clauses. System should answer on appeal process only and must not invent a housing-specific rule. Acceptable if it answers on general appeal rights with accurate citations."
    },
    {
        "id": "Q10",
        "query": "Does the program cover dental treatment costs?",
        "expected_decision": "ANSWER",
        "notes": "Out-of-scope refusal test. The manual is entirely silent on dental or medical treatment costs. System must state the manual does not cover this and direct to supervisor or district office. A confident wrong answer here is a hard failure."
    },
    {
        "id": "Q11",
        "query": "Under §6.4.1, how much of household earnings from employment is disregarded?",
        "expected_decision": "ANSWER",
        "determination_date": date(2026, 4, 15),
        "notes": "Post-March query about §6.4.1 (earnings disregard) — expect $175."
    },
    {
        "id": "Q12",
        "query": "Under §6.4.1, how much of household earnings from employment is disregarded?",
        "expected_decision": "ANSWER",
        "determination_date": date(2026, 2, 15),
        "notes": "Pre-March query about §6.4.1 (earnings disregard) — expect $120."
    },
    {
        "id": "Q13",
        "query": "How many days does a recipient have to report a change of circumstances?",
        "expected_decision": "FLAG_CONFLICT",
        "determination_date": date(2026, 4, 15),
        "notes": "Reporting-deadline query with no event date mentioned — expect both 10 and 14 days with explanation of ambiguity depending on whether change occurred before or on/after 1 March 2026."
    },
    {
        "id": "Q14",
        "query": "Can a sanction be imposed for a failure to report where the change of circumstances would have increased the award?",
        "expected_decision": "ANSWER",
        "determination_date": date(2026, 2, 15),
        "notes": "Query about §10.5.3A pre-March — expect clause is not yet in force, so it states the manual does not address this."
    },
    {
        "id": "Q15",
        "query": "Can a sanction be imposed for a failure to report where the change of circumstances would have increased the award?",
        "expected_decision": "ANSWER",
        "determination_date": date(2026, 4, 15),
        "notes": "Query about §10.5.3A post-March — expect clause is in force (no sanction allowed)."
    },
    {
        "id": "Q16",
        "query": "How many days does a recipient have to report a change of circumstances?",
        "expected_decision": "FLAG_CONFLICT",
        "determination_date": date(2026, 2, 15),
        "notes": "Historical regression case for Q02: pre-March 2026 contradiction between §4.3.2 (10 days) and §9.1.4 (30 days)."
    },
    {
        "id": "Q17",
        "query": "For a claim I'm deciding in April 2026, how many days does a recipient have to report a change of circumstances?",
        "expected_decision": "FLAG_CONFLICT",
        "notes": "Tests temporal ambiguity via natural-language date extraction (April 2026 determination, unknown event date)."
    },
    {
        "id": "Q18",
        "query": "For a claim in February 2026, how many days does a recipient have to report a change of circumstances?",
        "expected_decision": "FLAG_CONFLICT",
        "notes": "Tests pre-March contradiction via natural-language date extraction (February 2026 determination)."
    },
    {
        "id": "Q19",
        "query": "what is the earnings disregard amount for a claim covering 15 February to 15 March 2026",
        "expected_decision": "ANSWER",
        "notes": "Tests claim period spanning 1 March 2026 boundary (apportionment under §5.3 and §7.4.3)."
    },
    {
        "id": "Q20",
        "query": "What is the earnings disregard for a claim covering the period 2026-02-01 to 2026-04-01?",
        "expected_decision": "ANSWER",
        "notes": (
            "Tests pre-computed proration injection. Claim: 1 Feb to 1 Apr 2026. "
            "28 days before 1 March (at $120) + 32 days on/after (at $175) = 60 days total. "
            "Prorated = (28*120 + 32*175)/60 = 8960/60 = $149.33 per month. "
            "Pipeline must state $149.33 in the answer, still citing §5.3, §7.4.3, §6.4.1."
        )
    },
]


def run_evaluation() -> None:
    print("Starting pipeline evaluation...")
    import requests
    
    pipeline = Pipeline()
    results = []
    
    passed_count = 0
    failed_count = 0
    start_time = time.time()

    for i, q in enumerate(QUESTIONS):
        # check total timeout (15 minutes — 20 questions × ~30s average = ~600s, plus buffer)
        if time.time() - start_time > 900:
            print(f"\nHarness timeout exceeded. Skipping remaining {len(QUESTIONS) - i} questions.")
            for remaining_q in QUESTIONS[i:]:
                failed_count += 1
                results.append({
                    "id": remaining_q["id"],
                    "query": remaining_q["query"],
                    "decision": "TIMEOUT",
                    "expected_decision": remaining_q["expected_decision"],
                    "pass": False,
                    "fail_reason": "Harness timeout",
                    "answer": "",
                    "citations": [],
                })
            break

        print(f"\nRunning question {i+1}/{len(QUESTIONS)} ({q['id']}): {q['query']}")

        api_error_sleep = 5  # default sleep; extended on rate-limit hits
        try:
            det_date = q.get("determination_date")
            evt_date = q.get("event_date")
            result = pipeline.run(q["query"], determination_date=det_date, event_date=evt_date)
            
            # Detect swallowed API rate limit/error in pipeline result and set 30s backoff sleep
            if result.decision == "REFUSE" and ("HTTP error 429" in result.answer or "HTTP error 503" in result.answer or "temporarily unavailable" in result.answer):
                api_error_sleep = 30
                print(f"  -> Detected API error/429 in answer, sleeping {api_error_sleep}s before next question...")


            # PASS/FAIL Criteria
            passed = True
            fail_reason = None

            
            # Check decision match
            # Note special condition for Q10: REFUSE is acceptable if accompanied by escalation info
            if q["id"] == "Q10" and result.decision == "REFUSE":
                # Check for escalation reference
                ans_lower = result.answer.lower()
                if "supervisor" not in ans_lower and "district office" not in ans_lower:
                    passed = False
                    fail_reason = "REFUSE decision for Q10 did not include escalation instructions"
            elif result.decision != q["expected_decision"]:
                passed = False
                fail_reason = f"Decision mismatch: expected {q['expected_decision']}, got {result.decision}"
                
            # Check validation result
            if passed and result.validation and not result.validation.valid:
                passed = False
                fail_reason = f"Citation validation failed: {result.validation.errors}"
                
            # Check escalation instruction for FLAG_CONFLICT
            if passed and result.decision == "FLAG_CONFLICT":
                escalation_phrase = "This matter should be referred to a supervisor before any determination is made."
                if escalation_phrase not in result.answer:
                    passed = False
                    fail_reason = "Escalation instruction not present in FLAG_CONFLICT answer"
                elif q["id"] in ("Q16", "Q18"):
                    qid = q["id"]
                    if "4.3.2" not in result.answer or "9.1.4" not in result.answer:
                        passed = False
                        fail_reason = f"Clause numbers 4.3.2 and 9.1.4 not present in {qid} answer"
                    elif ("10 calendar days" not in result.answer.lower() and "10 days" not in result.answer.lower()) or ("30 calendar days" not in result.answer.lower() and "30 calendar day" not in result.answer.lower()):
                        passed = False
                        fail_reason = f"Verbatim text of conflicting clauses not present in {qid} answer"
                    elif "operative" not in result.answer.lower() or ("consequence" not in result.answer.lower() and "provision" not in result.answer.lower() and "downstream" not in result.answer.lower()):
                        passed = False
                        fail_reason = f"Explanation of operative rule vs downstream consequence not present in {qid} answer"
                elif q["id"] in ("Q13", "Q17"):
                    qid = q["id"]
                    if "4.3.2" not in result.answer:
                        passed = False
                        fail_reason = f"Clause number 4.3.2 not present in {qid} answer"
                    elif "10" not in result.answer or "14" not in result.answer:
                        passed = False
                        fail_reason = f"Both 10 and 14 calendar days must be mentioned to address ambiguity in {qid}"
                    elif "before 1 march" not in result.answer.lower() or "after 1 march" not in result.answer.lower():
                        passed = False
                        fail_reason = f"Answer must distinguish between changes occurring before vs on/after 1 March 2026 in {qid}"

            # Check custom assertions for Q11-Q15
            if passed:
                if q["id"] == "Q11":
                    if "$175" not in result.answer:
                        passed = False
                        fail_reason = "Post-March earnings disregard amount of $175 not present in answer"
                elif q["id"] == "Q12":
                    if "$120" not in result.answer:
                        passed = False
                        fail_reason = "Pre-March earnings disregard amount of $120 not present in answer"
                elif q["id"] == "Q14":
                    if "10.5.3A" in result.citations or "10.5.3A" in result.answer:
                        passed = False
                        fail_reason = "Clause 10.5.3A must not be cited or present for pre-March Q14"
                elif q["id"] == "Q15":
                    if "10.5.3A" not in result.citations:
                        passed = False
                        fail_reason = "Clause 10.5.3A must be cited for post-March Q15"
                elif q["id"] == "Q19":
                    if "5.3" not in result.citations or "7.4.3" not in result.citations or "6.4.1" not in result.citations:
                        passed = False
                        fail_reason = "Required citations (5.3, 7.4.3, 6.4.1) not present in Q19"
                    elif "apportion" not in result.answer.lower():
                        passed = False
                        fail_reason = "Answer must state that the award must be apportioned in Q19"
                elif q["id"] == "Q20":
                    # Hand-verified: (28*120 + 32*175)/60 = 8960/60 = $149.33
                    if "5.3" not in result.citations or "7.4.3" not in result.citations or "6.4.1" not in result.citations:
                        passed = False
                        fail_reason = "Required citations (5.3, 7.4.3, 6.4.1) not present in Q20"
                    elif "149.33" not in result.answer:
                        passed = False
                        fail_reason = "Pre-computed prorated figure $149.33 not present in Q20 answer"
                    elif "apportion" not in result.answer.lower():
                        passed = False
                        fail_reason = "Answer must state that the award must be apportioned in Q20"

            
            if passed:
                passed_count += 1
                status_str = "PASS"
                print(f"[{status_str}]")
            else:
                failed_count += 1
                status_str = "FAIL"
                print(f"[{status_str}] - Reason: {fail_reason}")
                
            # Record result
            results.append({
                "id": q["id"],
                "query": q["query"],
                "decision": result.decision,
                "expected_decision": q["expected_decision"],
                "pass": passed,
                "fail_reason": fail_reason,
                "answer": result.answer,
                "citations": result.citations,
            })
            
        except (requests.exceptions.Timeout, requests.exceptions.HTTPError) as e:
            failed_count += 1
            print(f"[FAIL] - API error: {str(e)}")
            results.append({
                "id": q["id"],
                "query": q["query"],
                "decision": "ERROR",
                "expected_decision": q["expected_decision"],
                "pass": False,
                "fail_reason": f"API error: {str(e)}",
                "answer": "",
                "citations": [],
            })
            # Back off longer after HTTP errors to let rate-limit window reset
            if "429" in str(e) or "503" in str(e):
                api_error_sleep = 30
                print(f"  -> Rate-limit/503 error, sleeping {api_error_sleep}s before next question...")

        except Exception as e:
            failed_count += 1
            print(f"[FAIL] - Pipeline errored: {str(e)}")
            results.append({
                "id": q["id"],
                "query": q["query"],
                "decision": "ERROR",
                "expected_decision": q["expected_decision"],
                "pass": False,
                "fail_reason": f"Pipeline exception: {str(e)}",
                "answer": "",
                "citations": [],
            })
        
        # Sleep to prevent hitting Gemini rate limits (15 RPM).
        # Use a longer sleep after 429/503 errors to reset the rate-limit window.
        time.sleep(api_error_sleep)

    # Ensure tests/eval directory exists
    output_path = Path("tests/eval/results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    print("\n" + "="*40)
    print(f"Evaluation completed. Passed: {passed_count}, Failed: {failed_count}")
    print(f"Results written to: {output_path.resolve()}")
    print("="*40)


if __name__ == "__main__":
    run_evaluation()
