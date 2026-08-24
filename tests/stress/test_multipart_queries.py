import unittest
import os
import sys
import time

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.pipeline import Pipeline

class TestMultipartQueries(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Override GEMINI_MODEL to gemini-3.1-flash-lite to avoid rate limits
        os.environ["GEMINI_MODEL"] = "gemini-3.1-flash-lite"
        cls.pipeline = Pipeline()

    def test_run_multipart_queries(self):
        queries = {
            "M01": "Am I eligible and how do I apply?",
            "M02": "What is the resource limit and what counts as a resource?",
            "M03": "Can a 16-year-old apply and what happens if they are also a full-time student?",
            "M04": "How long does a review take and what if I disagree with the outcome?",
            "M05": "What is the needs figure for a couple with two children one of whom is under 2 and one of whom is a full-time student and the household receives housing assistance from another program?"
        }

        print("\n=== STARTING MULTIPART LIVE PIPELINE RUNS ===")
        # Sleeping initially to clear any active rate limit windows
        time.sleep(10)

        for q_id, q_text in queries.items():
            print(f"\n==================================================")
            print(f"Running {q_id}: {repr(q_text)}")
            print(f"==================================================")
            
            start_time = time.time()
            try:
                result = self.pipeline.run(q_text)
                elapsed = time.time() - start_time
                print(f"Decision: {result.decision}")
                print(f"Answer:\n{result.answer}")
                print(f"Citations: {result.citations}")
                print(f"Validation: {'Valid' if result.validation and result.validation.valid else 'Invalid/None'}")
                if result.validation and not result.validation.valid:
                    print(f"Validation errors: {result.validation.errors}")
                print(f"Gate Decision: {result.gate_decision.decision}")
                print(f"Gate Reason: {result.gate_decision.reason}")
                print(f"Time Taken: {elapsed:.2f}s")
            except Exception as e:
                print(f"Execution Failed: {type(e).__name__}: {e}")
            
            # Rate-limiting cushion between calls
            print("Sleeping 12 seconds...")
            time.sleep(12)

        print("\n=== END OF MULTIPART LIVE PIPELINE RUNS ===")

if __name__ == "__main__":
    unittest.main()
