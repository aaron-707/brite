# Brite Spark 2026 — The Grounded Answer

A clause-grounded retrieval-augmented generation (RAG) system for the Calder County Household Support Program policy manual.

## Out of Scope
This system is designed strictly for single-turn clause grounding against the Calder County Household Support Program policy manual; it does *not* support multi-turn chat sessions, indexing or parsing other document manuals, or model fine-tuning.

---

## Prerequisites

- **Python Version**: Python 3.10 or higher.
- **Dependencies**: The system runs entirely on standard library modules and lightweight packages. It does not require any heavy deep-learning frameworks (like `torch` or `sentence-transformers`) or proprietary SDKs (like `google-generativeai`).
  - `scikit-learn` (for TF-IDF lexical retrieval)
  - `numpy` (for matrix operations)
  - `pyyaml` (for configuration parsing)
  - `requests` (for raw HTTP REST queries to the Gemini API)
  - `python-dotenv` (for loading API keys from `.env`)

---

## Getting Started

### 1. Clone the Repository and Navigate to the Directory

```bash
git clone https://github.com/aaron-707/brite.git
cd brite
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure the Environment

Create your environment configuration file from the template:
```bash
# On Linux / macOS:
cp .env.example .env

# On Windows:
copy .env.example .env
```
Open `.env` in a text editor and set your personal Gemini API key:

```text
GEMINI_API_KEY=your_actual_api_key_here
GEMINI_MODEL=gemini-3.5-flash
```

---

## Usage

### Run a single query

    python -m src.pipeline "Can a person aged 16 apply for assistance?"

Output format:
    Question: <query>
    Decision: ANSWER | FLAG_CONFLICT
    Conflicts: <list, if any>
    Answer: <plain language answer with inline citations>
    Citations: <list of §-references>

### Run the evaluation harness

    python -m tests.eval.run_eval

Runs the pipeline against the 10-question evaluation set. Results are
written to tests/eval/results.json. The harness does not exit non-zero
on question failures — honest pass/fail reporting is the point.

## Known issues

- Static query expansion table requires manual update when the policy
  manual changes. Automating from Part 1 definitions is the next step.
- §11.2.3 (review completion timeframe) may be absent from the corpus.
  If confirmed missing, flag to the manual maintainer.
- The escalation instruction directs to "a supervisor" generically.
  A production deployment should resolve this to a named role or contact.
