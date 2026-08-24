# Brite Spark 2026 — The Grounded Answer

A clause-grounded RAG system for the Calder County Household Support Program
policy manual. Given a caseworker's question, it retrieves the relevant policy
clauses, checks for internal contradictions and dead cross-references, and
produces a plain-language answer with exact clause citations. When the manual is
broken or silent, it says so and directs the caseworker to escalate.

## What it does and does not do

Does:
- Answer policy questions grounded strictly in the HSP manual.
- Parse and apply amendments from `1/Data pack/amendment-2026-01.md` alongside the base manual `1/Data pack/policy-manual.md`.
- Support date-based temporal routing using determination and event dates to resolve amended clauses.
- Cite the exact clause (§X.Y.Z) for every substantive claim.
- Compute and state the exact prorated dollar figure for claim periods spanning the 1 March 2026 amendment boundary.
- Detect and surface numeric contradictions between clauses.
- Detect and flag dead cross-references in the manual.
- Refuse cleanly when the manual is silent on a topic.
- Direct caseworkers to a supervisor when the manual conflicts or is broken.
- Handle colloquial and plain-English queries via a dynamic synonym map generated from the manual's own definitions.
- Update automatically when the policy manual changes — no code changes required.

Does not:
- Support multi-turn conversation or session memory.
- Fine-tune or train any model.
- Provide a web interface. The CLI is the complete interface.
- Resolve "supervisor" to a named contact or role.

## Prerequisites

Python 3.10 or higher. No deep learning frameworks required.

Dependencies (all lightweight):
- scikit-learn — TF-IDF retrieval
- numpy — matrix operations
- pyyaml — configuration parsing
- requests — Gemini REST API calls
- python-dotenv — API key loading

## Setup

### 1. Clone and navigate

```bash
git clone https://github.com/aaron-707/brite.git
cd brite
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

On Linux / macOS:
```bash
cp .env.example .env
```
On Windows:
```cmd
copy .env.example .env
```

Open `.env` and set your Gemini API key:
```env
GEMINI_API_KEY=your_actual_api_key_here
GEMINI_MODEL=gemini-flash-lite-latest
```

## Usage

### CLI Arguments & Options

You can specify the temporal context for a query using CLI options:
* `--as-of YYYY-MM-DD`: Sets the determination date (defaults to current system date).
* `--event-date YYYY-MM-DD`: Sets when the event under claim occurred.
* `--source §X.Y.Z`: A fast-path lookup tool that bypasses RAG and outputs the exact parsed text of the specified clause.

#### Temporal Resolution Example

Demonstrating the temporal resolver resolving pre-amendment vs. post-amendment rules using `--as-of`:

**Pre-amendment ($120 disregard):**
```bash
python -m src.pipeline "Under §6.4.1, how much of household earnings from employment is disregarded?" --as-of 2026-02-15
```
Output:
```
Question: Under §6.4.1, how much of household earnings from employment is disregarded?
Decision: ANSWER
Answer: The first $120 per month of household earnings from employment is disregarded (§6.4.1). This disregard applies once per household and not once per earner (§6.4.2).
Citations: §6.4.1, §6.4.2
```

**Post-amendment ($175 disregard):**
```bash
python -m src.pipeline "Under §6.4.1, how much of household earnings from employment is disregarded?" --as-of 2026-04-15
```
Output:
```
Question: Under §6.4.1, how much of household earnings from employment is disregarded?
Decision: ANSWER
Answer: The first $175 per month of household earnings from employment is disregarded (§6.4.1). This disregard applies once per household and not once per earner (§6.4.2).
Citations: §6.4.1, §6.4.2
```


### Single query

```bash
python -m src.pipeline "Can a person aged 16 apply?"
```

Output:
```
Question: Can a person aged 16 apply?
Decision: ANSWER
Answer: A person aged 16 or 17 may be eligible if they are not a member of any other household and have no person with parental responsibility able and willing to support them, or are a parent of a dependent child living with them (2.3.1).
Citations: §2.1.2, §2.3.1
```

### Numeric Contradiction (FLAG_CONFLICT)

Demonstrating the gate catching a policy conflict (the pre-March 10 vs. 30 day reporting deadline contradiction) and refusing to guess:

```bash
python -m src.pipeline "How many days does a recipient have to report a change of circumstances?" --as-of 2026-02-15
```
Output:
```
Question: How many days does a recipient have to report a change of circumstances?
Decision: FLAG_CONFLICT
Conflicts: ['Numeric contradiction referencing §4.3: §4.3.2 states 10; §9.1.4 states 30. These values are inconsistent.']
Answer: There is a conflict in the policy manual regarding the number of days a recipient has to report a change of circumstances. Clause 4.3.2 states that a recipient must report a change within 10 calendar days of the change occurring or becoming aware of it, whichever is later (4.3.2). However, clause 9.1.4 refers to a 30 calendar days requirement under §4.3 regarding overpayments (9.1.4). These values are inconsistent. Clause 4.3.2 is the operative rule, while clause 9.1.4 is the downstream consequence... This matter should be referred to a supervisor before any determination is made.
Citations: §1.4.8, §4.3.2, §4.3.3, §9.1.4
```

### Soft Refusal (ANSWER with escalation)

Demonstrating the system cleanly declining to answer when the manual is silent on a topic:

```bash
python -m src.pipeline "Does the program cover dental treatment costs?"
```
Output:
```
Question: Does the program cover dental treatment costs?
Decision: ANSWER
Answer: The manual does not cover this matter. Please contact your district office or supervisor for guidance.
```


### Evaluation harness

```bash
python -m tests.eval.run_eval
```

Runs the pipeline against the 20-question evaluation set. Results written to
`tests/eval/results.json`. The harness does not exit non-zero on question failures
— honest pass/fail reporting is the point. Final result: 20/20 PASS.


### Stress tests

```bash
python -m unittest tests/stress/test_corpus_integrity.py -v
python -m unittest tests/stress/test_api_failures.py -v
```

## Architecture overview

```
    Query & Date Flags (--as-of, --event-date)
      ↓
    Query Expansion (dynamic synonym map from Part 1 definitions,
                     cached by corpus MD5 fingerprint)
      ↓
    Hybrid Retrieval (BM25 + TF-IDF, RRF fusion)
      ↓
    Temporal Resolver (resolves correct clause versions using dates)
      ↓
    Gate (dead reference detection + numeric contradiction
          detection + soft fallback for zero-coverage queries)
      ↓
    Synthesizer (Gemini REST, temperature 0.1, strict grounding
                 prompt, escalation instruction on conflict)
      ↓
    Citation Validator (checks citations, digit limits, and context)
      ↓
    Output
```

## Scope boundaries

- Multi-part queries are answered on the highest-scoring topic only; sub-questions are not split. This maintains citation precision — each answer is grounded in a single retrieval pass with verifiable clause references. See DECISIONS.md Section 8 for the proposed v2 fix.
- Escalation instructions name "a supervisor" generically, because the manual itself does not specify roles or contacts. A production deployment would integrate a staff directory.
- The dead-reference detector is subject to false positives (very short sentences with sparse preceding context may be incorrectly flagged as dead) and false negatives (broken references using structural connectives like "under" or "subject to" will be missed). See DECISIONS.md Section 3 for the detailed tuning reasoning.

## Further Documentation

* See [DECISIONS.md](DECISIONS.md) for the reasoning behind every design choice, including what was rejected and what the system deliberately does not do.
* See [AI-USAGE.md](AI-USAGE.md) for the AI disclosure.

