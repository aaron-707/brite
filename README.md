# Brite Spark 2026 — The Grounded Answer

Clause-grounded RAG system for the Calder County Household Support Program
policy manual.

## Architecture

```
question → Retriever → Gate → Synthesizer → Citation Validator → answer
```

Four decoupled components:

1. **Retriever** — Hybrid BM25 Okapi + TF-IDF (scikit-learn) with weighted RRF
   fusion. Fully offline, zero model downloads.
2. **Gate** — Pre-LLM decision (ANSWER / REFUSE / FLAG_CONFLICT). Checks
   retrieval confidence, term overlap, cross-reference integrity, and numeric
   contradictions. All thresholds in `config/gate_thresholds.yaml`.
3. **Synthesizer** — Calls Gemini REST API directly via `requests` (no SDK).
   Structured JSON output with `responseSchema`. API key from `.env`.
4. **Citation Validator** — Post-checks every clause citation against the
   retrieved set. Rejects hallucinated citations and unsupported claims.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your GEMINI_API_KEY
```

## Usage

```bash
python -m src.pipeline "What is the reporting window for changes?"
```

## Configuration

All thresholds and weights are in `config/gate_thresholds.yaml` — never
hardcoded in source code.

## Corpus

The policy manual lives at `1/Data pack/policy-manual.md`. It is read-only and
must never be generated or replaced.
