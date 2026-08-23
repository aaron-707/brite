import unittest
from unittest.mock import patch, MagicMock
import requests
import json
import time
import os

from src.synthesizer import Synthesizer
from src.gate import GateDecision

class TestAPIFailures(unittest.TestCase):
    def setUp(self):
        # We need a dummy Synthesizer instance.
        # Ensure GEMINI_API_KEY is set in env for testing.
        if "GEMINI_API_KEY" not in os.environ:
            os.environ["GEMINI_API_KEY"] = "mock_api_key"
        self.synthesizer = Synthesizer()
        self.gate_decision = GateDecision(decision="ANSWER", reason="Gate passed")

    @patch("requests.post")
    def test_http_500_internal_server_error(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "500 Internal Server Error", response=mock_resp
        )
        mock_post.return_value = mock_resp

        result = self.synthesizer.generate("test question", [], self.gate_decision)
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result.get("decision"), "REFUSE")
        self.assertIn("temporarily unavailable", result.get("answer"))

    @patch("requests.post")
    def test_malformed_json_response(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.side_effect = json.JSONDecodeError("Expecting value", "not valid json {{{{", 0)
        mock_post.return_value = mock_resp

        result = self.synthesizer.generate("test question", [], self.gate_decision)
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result.get("decision"), "REFUSE")
        self.assertIn("unexpected response", result.get("answer"))

    @patch("requests.post")
    def test_empty_response_body(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.side_effect = json.JSONDecodeError("Expecting value", "", 0)
        mock_post.return_value = mock_resp

        result = self.synthesizer.generate("test question", [], self.gate_decision)
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result.get("decision"), "REFUSE")
        self.assertIn("unexpected response", result.get("answer"))

    @patch("requests.post")
    def test_request_timeout(self, mock_post):
        mock_post.side_effect = requests.exceptions.Timeout("Request timed out")

        result = self.synthesizer.generate("test question", [], self.gate_decision)
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result.get("decision"), "REFUSE")
        self.assertIn("temporarily unavailable", result.get("answer"))

    @patch("requests.post")
    def test_repeated_429_rate_limit(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "429 Too Many Requests", response=mock_resp
        )
        mock_post.return_value = mock_resp

        start_time = time.time()
        result = self.synthesizer.generate("test question", [], self.gate_decision)
        elapsed = time.time() - start_time
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result.get("decision"), "REFUSE")
        self.assertIn("temporarily unavailable", result.get("answer"))
        # Verify elapsed time is under 10 seconds
        self.assertLess(elapsed, 10.0, f"Expected test to run in < 10s, but took {elapsed:.2f}s")

if __name__ == "__main__":
    unittest.main()
