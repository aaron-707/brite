"""Synthesizer: calls Gemini REST API directly via requests.

No SDK — raw HTTPS to generativelanguage.googleapis.com.
API key from GEMINI_API_KEY in .env via python-dotenv.
Model name from GEMINI_MODEL in .env.
Uses structured JSON output via responseSchema.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import requests
import yaml
from dotenv import load_dotenv

from .gate import GateDecision
from .retriever import RetrievalResult

# Load .env from project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

_GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)

# System instruction for the LLM
_SYSTEM_INSTRUCTION = """\
You are a policy-manual assistant for the Calder County Household Support Program.

RULES — follow every one without exception:
1. Use ONLY the clause texts provided below to answer the question.
2. Do NOT use outside knowledge, even if it seems correct.
3. Every substantive claim in your answer MUST cite the clause id (e.g. "4.3.2") \
from the provided clauses. Do not invent clause ids.
4. If the provided clauses flag a conflict or inconsistency, state the conflict \
explicitly and cite both sides.
5. If the clauses do not contain enough information to answer, say so rather than \
guessing.
6. Return your response as JSON with two fields:
   - "answer": your full answer text with inline clause id citations
   - "citations": an array of every clause id you cited (strings like "4.3.2")
"""

# JSON schema for structured output
_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "answer": {
            "type": "STRING",
            "description": "The full answer text with inline clause id citations.",
        },
        "citations": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
            "description": "Array of clause ids cited in the answer (e.g. '4.3.2').",
        },
    },
    "required": ["answer", "citations"],
}


@dataclass
class SynthesizerOutput:
    """Result from the synthesizer."""

    answer: str
    cited_clause_ids: list[str]
    raw_response: dict = field(default_factory=dict, repr=False)


class Synthesizer:
    """Calls Gemini REST API to generate a grounded answer."""

    def __init__(self, config_path: str | Path = "config/gate_thresholds.yaml") -> None:
        self.api_key = os.environ.get("GEMINI_API_KEY", "")
        if not self.api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY not set. Add it to your .env file."
            )

        self.model = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")

        # Load retry config
        config_path = Path(config_path)
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        else:
            cfg = {}
        self.max_retries: int = cfg.get("synthesizer", {}).get("max_retries", 1)

    def generate(
        self,
        question: str,
        clauses: list[RetrievalResult],
        gate_decision: GateDecision,
        correction: str | None = None,
    ) -> SynthesizerOutput:
        """Generate an answer using the Gemini REST API.

        Args:
            question: The user's question.
            clauses: Retrieved clause results to ground the answer.
            gate_decision: The gate's decision (may include conflicts).
            correction: Optional correction instruction for retry attempts.

        Returns:
            SynthesizerOutput with the answer, cited clause ids, and raw response.
        """
        # Build the clause context block
        clause_block = "\n\n".join(
            f"§{c.clause_id}:\n{c.clause_text}" for c in clauses
        )

        # Build the user prompt
        prompt_parts = [f"QUESTION: {question}\n\nPROVIDED CLAUSES:\n{clause_block}"]

        if gate_decision.conflicts:
            conflict_text = "\n".join(f"- {c}" for c in gate_decision.conflicts)
            prompt_parts.append(
                f"\nFLAGGED CONFLICTS (address these explicitly in your answer):\n{conflict_text}"
            )

        if correction:
            prompt_parts.append(f"\nCORRECTION INSTRUCTION: {correction}")

        user_prompt = "\n".join(prompt_parts)

        # Build request body
        body = {
            "system_instruction": {
                "parts": [{"text": _SYSTEM_INSTRUCTION}]
            },
            "contents": [
                {
                    "parts": [{"text": user_prompt}]
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "responseMimeType": "application/json",
                "responseSchema": _RESPONSE_SCHEMA,
            },
        }

        url = _GEMINI_ENDPOINT.format(model=self.model)
        headers = {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json",
        }

        resp = requests.post(url, json=body, headers=headers, timeout=60)
        resp.raise_for_status()
        raw = resp.json()

        return self._parse_response(raw)

    def _parse_response(self, raw: dict) -> SynthesizerOutput:
        """Extract answer and citations from the Gemini response JSON."""
        import json

        try:
            candidates = raw.get("candidates", [])
            if not candidates:
                return SynthesizerOutput(
                    answer="",
                    cited_clause_ids=[],
                    raw_response=raw,
                )

            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            if not parts:
                return SynthesizerOutput(
                    answer="",
                    cited_clause_ids=[],
                    raw_response=raw,
                )

            text = parts[0].get("text", "")

            # The response should be JSON due to responseMimeType
            parsed = json.loads(text)
            answer = parsed.get("answer", "")
            citations = parsed.get("citations", [])

            return SynthesizerOutput(
                answer=answer,
                cited_clause_ids=citations,
                raw_response=raw,
            )
        except (json.JSONDecodeError, KeyError, IndexError):
            # Fallback: extract what we can from the raw text
            text = ""
            try:
                text = raw["candidates"][0]["content"]["parts"][0]["text"]
            except (KeyError, IndexError):
                pass

            # Try to extract citations from raw text
            cite_re = re.compile(r"§?(\d+\.\d+\.\d+)")
            found = cite_re.findall(text)

            return SynthesizerOutput(
                answer=text,
                cited_clause_ids=list(set(found)),
                raw_response=raw,
            )
