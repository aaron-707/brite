# Architectural Decisions

This document captures the design decisions, component evolutions, and current state of the Brite Spark 2026 Grounded Answer pipeline.

## 1. Corpus Parsing (Commit `ea4305f`)
- **Implementation**: Chunking is strictly driven by the manual's structural paragraph hierarchy: `§Part.Section.Paragraph` (e.g., `**X.Y.Z**`).
- **Statistics**: Yields exactly 148 clauses across 12 Parts.
- **Rationale**: Chunking by paragraph length or semantic similarity would destroy the reference boundaries of the policy manual and make validation/citation tracing unreliable.

## 2. Retriever (Commit `922bba8`)
- **Implementation**: Hybrid lexical retrieval combining a hand-rolled BM25 Okapi retriever ($k_1=1.5, b=0.75$) and scikit-learn's `TfidfVectorizer` (cosine similarity), combined via rank-reciprocal fusion (RRF).
- **Rationale**: 
  - Zero heavy deep-learning dependencies (`torch` / `sentence-transformers` are completely banned).
  - Guarantees reliability and speed in clean-clone grading environments with zero model downloads.
  - Lexical matching is ideal for a legal-style manual with highly specific, defined terminology (e.g., "countable resources", "disregards", "full-time student") where exact term overlap indicates topical relevance.

## 3. Gate (Commits `acebfc0`, `82d9ea0`, and `2e869d4`)
> [!IMPORTANT]
> **NEEDS HUMAN REVIEW**
> The generalization of the term-overlap approach to cross-references beyond the 12 tested cases is not fully verified.

### Evolution
- **First Attempt**: TF-IDF cosine similarity between the referencing sentence and target clause. The threshold was tuned iteratively ($0.3 \to 0.2 \to 0.15$) against the known dead reference (§7.1.3 $\to$ §5.4).
- **The Finding**: Testing against 12 real cross-references showed that the dead reference's similarity ($0.123$) fell inside the range of legitimate structural references ($0.099 - 0.405$). No static cosine similarity threshold could separate correct cross-references from incorrect ones.
- **Second Attempt (Term Overlap)**: Replaced TF-IDF with a targeted term-overlap check. For each citation (e.g., "§5.4"), the gate extracts up to 6 non-stopword tokens (minimum length 2 characters) immediately preceding the citation in the source sentence. If overlap between these preceding tokens (or their 5-character prefixes) and the resolved target clause is zero, it flags a `FLAG_CONFLICT` ("dead reference").
- **Third Attempt (Document Frequency Filtering)**: The initial term-overlap check allowed any non-stopword to trigger a match. This meant generic administrative terms (like "within", "period", "person", "member", "days", "household") could trigger false-positive matches. To fix this, we computed the Document Frequency (DF) of all terms across the 148 clauses. Any term appearing in $>15\%$ of the manual's clauses (41 terms total) is now excluded from being used as a valid match.

### Current Status of Specificity-Weighting
- By introducing the $>15\%$ Document Frequency filter, the system successfully filters out generic administrative words. This resolves the specificity issue, forcing the check to rely on higher-value topical words (e.g., matching on "sanction" instead of "person", or "circumstances" instead of "within"). An adversarial test between unrelated clauses (e.g., §3.1.1 vs §12.2.1) correctly yields `NO` overlap.
- **Known Limitation**: While this cleanly separates the tested cases and unrelated clauses, very short cross-referencing sentences with sparse preceding context may still fail to yield sufficient specific terms, leading to potential false dead-reference flags. More complex syntax parsing remains unverified.

## 4. Synthesizer (Commit `cf6c919`)
- **Implementation**: Google Gemini REST API accessed via raw HTTP requests (using python `requests`). 
- **Model Choice**: `gemini-3.5-flash` (newer generation Flash).
- **Rationale**: Direct REST calls eliminate bulkier SDK dependencies (`google-generativeai`). `gemini-3.5-flash` was selected over newer models (like `gemini-3.6` or `gemini-3.7` previews) because standard control over generation configuration (`temperature=0.1`, `top_p`, `top_k`) is preserved, which is essential for deterministic, repeatable, and low-temperature grounded outputs. The API key is loaded strictly from a `.env` file via `python-dotenv`.

## 5. Current Outstanding Tasks (What the system does NOT do yet)
- **Harness Verification**: The evaluation harness has not yet been executed against the required 10-question evaluation set.
- **Live Verification**: End-to-end integration and pipeline testing against the live Gemini API is not yet verified.
- **Documentation**: The root `README.md` contains basic setup instructions but needs update with actual, verified CLI usage instructions once the full pipeline is tested.
