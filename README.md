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

### Run a Single Query

To query the pipeline via the CLI on a single question, run:

```bash
python -m src.pipeline "Can a person aged 16 apply for assistance?"
```

### Run the Evaluation Harness

The evaluation suite runs the pipeline against a 10-question set covering normal answers, expected refusals, the known contradiction (§4.3.2 vs §9.1.4), and the dead reference (§7.1.3 → §5.4). Pass/fail results are printed to stdout and written to `tests/eval/results.json`.

```bash
python -m tests.eval.run_eval
```

The harness exits with code 0 if all questions produce a decision (ANSWER or FLAG_CONFLICT). It exits with code 1 if the pipeline errors. It does not exit non-zero for questions that fail — honest failure reporting is the point.

## What the system does and does not do

**Does:**
- Answer policy questions grounded strictly in the Calder County HSP policy manual.
- Cite the specific clause (§X.Y.Z) for every substantive claim.
- Detect and surface numeric contradictions between clauses.
- Detect and flag dead cross-references in the manual.
- Refuse to answer (FLAG_CONFLICT) when the manual is broken or internally inconsistent, and tell the user to escalate.

**Does not:**
- Support multi-turn conversation or session memory.
- Answer questions from any document other than the corpus in `data/policy-manual.md`.
- Fine-tune or train any model.
- Provide a web interface. The CLI is the intended and complete interface.
- Resolve legal ambiguities — where the manual conflicts, the system surfaces the conflict and stops.

## Known issues

- The gate incorrectly flags §7.1.3 → §7.3 as a dead reference. §7.3 is a live, legitimate cross-reference. A fix (structural connective phrase whitelist) has been identified and is tracked in DECISIONS.md Section 5.
- The conflict list for the §4.3.2 / §9.1.4 numeric contradiction contains a duplicate entry. This does not affect the answer but will be cleaned up.
