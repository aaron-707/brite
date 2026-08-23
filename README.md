# Brite Spark 2026 — The Grounded Answer

A clause-grounded RAG system for the Calder County Household Support Program
policy manual. Given a caseworker's question, it retrieves the relevant policy
clauses, checks for internal contradictions and dead cross-references, and
produces a plain-language answer with exact clause citations. When the manual is
broken or silent, it says so and directs the caseworker to escalate.

## What it does and does not do

Does: - Answer policy questions grounded strictly in the HSP manual. - Cite the
exact clause (§X.Y.Z) for every substantive claim. - Detect and surface numeric
contradictions between clauses. - Detect and flag dead cross-references in the
manual. - Refuse cleanly when the manual is silent on a topic. - Direct
caseworkers to a supervisor when the manual conflicts or is broken. - Handle
colloquial and plain-English queries via a dynamic synonym map generated from
the manual's own definitions. - Update automatically when the policy manual
changes — no code changes required.

Does not: - Support multi-turn conversation or session memory. - Answer
questions from any document other than data/policy-manual.md. - Fine-tune or
train any model. - Provide a web interface. The CLI is the complete interface. -
Resolve "supervisor" to a named contact or role.

## Prerequisites

Python 3.10 or higher. No deep learning frameworks required.

Dependencies (all lightweight): - scikit-learn — TF-IDF retrieval - numpy —
matrix operations - pyyaml — configuration parsing - requests — Gemini REST API
calls - python-dotenv — API key loading

## Setup

### 1. Clone and navigate

git clone https://github.com/aaron-707/brite.git cd brite

### 2. Install dependencies

pip install -r requirements.txt

### 3. Configure environment

    # Linux / macOS:
    cp .env.example .env

    # Windows:
    copy .env.example .env

Open .env and set your Gemini API key:

GEMINI_API_KEY=your_actual_api_key_here GEMINI_MODEL=gemini-3.5-flash

## Usage

### Single query

python -m src.pipeline "Can a person aged 16 apply?"

Output:

    Question: Can a person aged 16 apply?
    Decision: ANSWER
    Answer: A person aged 16 or 17 may be eligible if they are not a member of any other household and have no person with parental responsibility able and willing to support them, or are a parent of a dependent child living with them (2.3.1).
    Citations: §2.1.2, §2.3.1

### Evaluation harness

python -m tests.eval.run_eval

Runs the pipeline against the 10-question evaluation set. Results written to
tests/eval/results.json. The harness does not exit non-zero on question failures
— honest pass/fail reporting is the point. Final result: 10/10 PASS.

### Stress tests

python -m unittest tests/stress/test_corpus_integrity.py -v python -m unittest
tests/stress/test_api_failures.py -v

## Architecture overview

    Query
      ↓
    Query Expansion (dynamic synonym map from Part 1 definitions,
                     cached by corpus MD5 fingerprint)
      ↓
    Hybrid Retrieval (BM25 + TF-IDF, RRF fusion)
      ↓
    Gate (dead reference detection + numeric contradiction
          detection + soft fallback for zero-coverage queries)
      ↓
    Synthesizer (Gemini REST, temperature 0.1, strict grounding
                 prompt, escalation instruction on conflict)
      ↓
    Citation Validator
      ↓
    Output

## Known limitations

- Multi-part queries are answered on the highest-scoring topic only. Sub-questions are not split.
- The escalation instruction names "a supervisor" generically. A production deployment needs a staff directory integration.
- Very short cross-referencing sentences with sparse context may produce false dead-reference flags.
