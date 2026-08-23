import unittest
import tempfile
import os
import warnings
from pathlib import Path

from src.parser import parse_corpus, Clause, build_clause_index

class TestCorpusIntegrity(unittest.TestCase):
    def setUp(self):
        self.temp_files = []

    def tearDown(self):
        for f in self.temp_files:
            try:
                os.remove(f)
            except OSError:
                pass

    def _create_temp_file(self, content: str) -> str:
        fd, path = tempfile.mkstemp(suffix=".md", text=True)
        self.temp_files.append(path)
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(content)
        return path

    def test_missing_corpus_file(self):
        # Test 1: Missing corpus file
        non_existent_path = "does_not_exist_file_path.md"
        with self.assertRaises(FileNotFoundError) as context:
            parse_corpus(non_existent_path)
        self.assertIn("Corpus not found at", str(context.exception))

    def test_empty_file(self):
        # Test 2: Empty file
        path = self._create_temp_file("")
        result = parse_corpus(path)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 0)

    def test_truncated_file(self):
        # Test 3: Truncated file — valid clauses followed by incomplete clause
        content = (
            "**1.1.1** Full valid clause text here.\n"
            "**1.1.2** Another valid clause.\n"
            "**1.1.3** This clause is cut off mid-sent"
        )
        path = self._create_temp_file(content)
        result = parse_corpus(path)
        self.assertIsNotNone(result)
        self.assertIn(len(result), [2, 3])

    def test_duplicate_clause_ids(self):
        # Test 4: Duplicate clause IDs
        content = (
            "**1.1.1** First definition of this clause.\n"
            "**1.1.1** Second definition of this clause.\n"
            "**1.1.2** Normal clause.\n"
        )
        path = self._create_temp_file(content)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                clauses = parse_corpus(path)
                index = build_clause_index(clauses)
                
                # Check if we logged a warning or raised ValueError
                has_warning = len(w) > 0
                if not has_warning:
                    # If no warning, we must not silently overwrite with the second definition
                    self.assertEqual(
                        index["1.1.1"].text, 
                        "**1.1.1** First definition of this clause.", 
                        "Silently overwrote duplicate clause ID without warning/error"
                    )
                    self.assertTrue(has_warning, "Kept first occurrence but did not log a warning.")
            except ValueError as ve:
                self.assertIn("1.1.1", str(ve))

    def test_malformed_paragraph_headers(self):
        # Test 5: Malformed paragraph headers — wrong format
        content = (
            "## Part 1\n"
            "1.1.1 Missing bold markers entirely.\n"
            "**1.1** Only two-part reference, not three.\n"
            "**X.Y.Z** Non-numeric part identifiers.\n"
            "**1.1.1** This one is valid.\n"
        )
        path = self._create_temp_file(content)
        result = parse_corpus(path)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].clause_id, "1.1.1")

    def test_unicode_content(self):
        # Test 6: Unicode content in clause text
        content = (
            "**1.1.1** A household may include a person whose name "
            "contains non-ASCII characters, e.g. José García or "
            "Müller. Awards are paid in US dollars ($)."
        )
        path = self._create_temp_file(content)
        result = parse_corpus(path)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].clause_id, "1.1.1")
        self.assertIn("José García", result[0].text)
        self.assertIn("Müller", result[0].text)

    def test_windows_line_endings(self):
        # Test 7: Windows line endings (CRLF)
        content = (
            "**1.1.1** Clause one text.\r\n"
            "**1.1.2** Clause two text.\r\n"
        )
        path = self._create_temp_file(content)
        result = parse_corpus(path)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].clause_id, "1.1.1")
        self.assertEqual(result[1].clause_id, "1.1.2")

    def test_very_long_clause_text(self):
        # Test 8: Very long clause text
        long_text = "a" * 10000
        content = f"**1.1.1** {long_text}\n"
        path = self._create_temp_file(content)
        result = parse_corpus(path)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].clause_id, "1.1.1")
        self.assertTrue(len(result[0].text) >= 10000)

    def test_only_part_headers_no_clauses(self):
        # Test 9: Corpus with only Part headers and no clauses
        content = (
            "# Part 1 — Definitions\n"
            "## 1.1 General\n"
            "# Part 2 — Eligibility\n"
            "## 2.1 Conditions\n"
        )
        path = self._create_temp_file(content)
        result = parse_corpus(path)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 0)

if __name__ == "__main__":
    unittest.main()
