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
You are a policy assistant for the Calder County Household Support Program. Your job is to answer questions from front-line caseworkers using ONLY the provided policy manual clauses.

You must follow these rules without exception:

1. CITATION DISCIPLINE
- Each clause reference (e.g., "4.3.2") must appear at most once in your entire answer. Place the citation at the point of first use.
- Do not repeat a citation at the end of the answer if it was already cited inline.
- Only cite clauses that are explicitly provided in the retrieved context. Never invent or assume other clause numbers.
- Spell out what a clause says in plain terms first, and then cite it immediately afterwards (e.g., "A recipient must report changes in income (4.3.2)").

2. GROUNDING DISCIPLINE
- Every substantive factual claim in your answer must trace directly to a specific retrieved clause.
- Do not use general knowledge of benefit programs. If a detail is not written in the provided clauses, treat it as completely non-existent.
- If the retrieved clauses only partially answer the question, state exactly what is covered and what is missing.

3. CONFLICT HANDLING (FLAG_CONFLICT Decisions)
When the question is flagged with a conflict, you must:
- Describe the conflict clearly.
- If it is a numeric contradiction between two clauses (e.g. different timeframes like 10 days vs 30 days):
  (a) Identify both clause numbers that disagree (e.g. 4.3.2 and 9.1.4) and detail the exact discrepancy.
  (b) Explicitly state which is the operative rule/obligation and which is the downstream consequence (for example, §4.3.2 is the operative rule/obligation requiring reporting within 10 days, while §9.1.4 is the downstream consequence related to recovering overpayments within 30 days), without silently resolving which one "wins."
- If it is a structural reference issue or dead reference (e.g. a clause referencing a section that is topically unrelated or missing):
  (a) State clearly what is broken and name the target clause/section that is referenced.
- For all conflicts:
  (a) Under a "Conflicting provisions" heading, print the full text of the conflicting or referencing clauses verbatim.
  (b) State what is known from the clauses that are not in conflict.
  (c) Do not resolve the conflict. Do not pick one side or speculate.
  (d) End your answer with this exact sentence: "This matter should be referred to a supervisor before any determination is made."

4. REFUSAL (Uncovered Matters)
If the retrieved clauses do not address the caseworker's question at all:
- State clearly that the manual does not cover this matter.
- Do not speculate or extrapolate.
- End your answer with this exact sentence: "Please contact your district office or supervisor for guidance."

5. PLAIN LANGUAGE
- Write for a front-line caseworker, not a lawyer. Use clear, simple language.
- Use short sentences. Avoid complex subordinate clauses.
- Do not use formal legalistic terms like "aforementioned", "pursuant to", or "herein".

6. OUTPUT FORMAT
Return your response as a JSON object with the following fields:
- "answer": The plain-language answer text containing inline clause citations in parentheses.
- "citations": An array of every clause id cited in the answer (e.g. ["4.3.2", "9.1.4"]).
"""

_NO_COVERAGE_INSTRUCTION = """\
The retriever found no clauses in the manual relevant to this query. Do not speculate or draw on general knowledge. Tell the caseworker clearly that the manual does not cover this matter and direct them to contact their district office or supervisor for guidance.

Return your response as a JSON object with the following fields:
- "answer": The answer text directing the caseworker to contact their district office or supervisor.
- "citations": An empty array [].
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
            conflict_details = []
            for conflict in gate_decision.conflicts:
                conflict_details.append(f"- {conflict}")
                cids = re.findall(r"(\d+\.\d+\.\d+)", conflict)
                for cid in cids:
                    for res in clauses:
                        if res.clause_id == cid:
                            conflict_details.append(f"  Verbatim text of §{cid}: \"{res.clause_text}\"")
            conflict_text = "\n".join(conflict_details)
            prompt_parts.append(
                f"\nFLAGGED CONFLICTS (address these explicitly in your answer):\n{conflict_text}"
            )

        if correction:
            prompt_parts.append(f"\nCORRECTION INSTRUCTION: {correction}")

        user_prompt = "\n".join(prompt_parts)

        system_instruction = _NO_COVERAGE_INSTRUCTION if getattr(gate_decision, "no_coverage", False) else _SYSTEM_INSTRUCTION

        # Build request body
        body = {
            "system_instruction": {
                "parts": [{"text": system_instruction}]
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

        try:
            # Call the API with exponential backoff on 429 rate limit errors
            import time
            max_api_attempts = 3
            backoff = 2.0
            backoff_cap = 16.0
            for attempt in range(max_api_attempts):
                resp = requests.post(url, json=body, headers=headers, timeout=60)
                if resp.status_code == 429:
                    if attempt < max_api_attempts - 1:
                        sleep_time = min(backoff, backoff_cap)
                        time.sleep(sleep_time)
                        backoff *= 2.0
                        continue
                resp.raise_for_status()
                break
            else:
                resp.raise_for_status()

            raw = resp.json()
            return self._parse_response(raw)

        except (requests.exceptions.Timeout,
                requests.exceptions.ConnectionError) as e:
            return {
                "decision": "REFUSE",
                "answer": "The system is temporarily unavailable. "
                          "Please try again or contact your supervisor.",
                "citations": [],
                "conflicts": []
            }
        except requests.exceptions.HTTPError as e:
            return {
                "decision": "REFUSE",
                "answer": f"The system is temporarily unavailable (HTTP error {e.response.status_code if e.response is not None else 'unknown'}). "
                          "Please try again or contact your supervisor.",
                "citations": [],
                "conflicts": []
            }
        except ValueError as e:
            return {
                "decision": "REFUSE",
                "answer": "The system received an unexpected response. "
                          "Please try again or contact your supervisor.",
                "citations": [],
                "conflicts": []
            }
        except Exception as e:
            return {
                "decision": "REFUSE",
                "answer": f"An unexpected error occurred: {str(e)}. "
                          "Please try again or contact your supervisor.",
                "citations": [],
                "conflicts": []
            }

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
