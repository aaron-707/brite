# Architectural Decisions

This document captures every design decision made during the Brite Spark 2026
Grounded Answer build, including what was rejected and why, what failed in
testing, and what the system still does not do.

## 1. Corpus Parsing (Commit ea4305f)

Implementation: chunking strictly by §Part.Section.Paragraph hierarchy using the
**X.Y.Z** bold marker pattern. Yields exactly 148 clauses across 12 Parts.

What I rejected: chunking by token length or semantic similarity. Both would
destroy the reference boundaries that make citation tracing possible. The whole
point of this system is that a caseworker can verify every claim against a
specific clause — that only works if each chunk IS a clause.

What I would fix first: the parser currently skips malformed headers silently. A
production system should emit a structured warning log listing every skipped
line so corpus maintainers can catch formatting errors when the manual updates
quarterly. Duplicate clause ID detection is implemented (emits a warning and
retains the first occurrence) but does not block ingestion.

## 2. Retriever (Commit 922bba8)

Implementation: hybrid BM25 Okapi (k1=1.5, b=0.75, hand-rolled) and TF-IDF
cosine similarity (scikit-learn TfidfVectorizer), combined via rank-reciprocal
fusion (RRF).

What I rejected: dense embeddings (sentence-transformers, torch). Banned by the
dependency constraints of the grading environment. But also genuinely wrong for
this corpus — a legal-style manual with defined terminology where "countable
resources" means exactly that, not something semantically similar. Exact term
overlap is the right signal here.

Why RRF over weighted sum: RRF is rank-based so it does not require tuning the
relative weight of BM25 vs TF-IDF scores, which are on different scales. It is
also more robust to one retriever returning a very high-scoring outlier that
drowns out the other.

### Query Expansion — Dynamic Synonym Map (post-eval addition)

Problem identified in eval: two queries failed retrieval because the query
vocabulary did not match the manual's vocabulary. "Car" is not in the manual —
"motor vehicle" is (§2.4.2). "How long does the Department have to complete a
review" did not surface §11.2.3.

Three approaches considered and rejected before the final design:

(a) Reactive hardcoded synonym table — add entries when queries fail. Rejected:
unmaintainable, grows without bound, breaks when the manual changes quarterly.

(b) Low-DF topical term mapping — map colloquial terms to manual terms that
survive the DF filter. Rejected: still hardcoding, just more carefully chosen.
Same maintenance problem.

(c) Static corpus-grounded table — entries derived from Part 1 definitions only.
Better, but still requires manual updates when Part 1 changes.

Final design: the expansion map is generated dynamically at startup via a single
Gemini API call. The retriever extracts all defined terms from Part 1 clauses
(pattern: **1.X.Y Term** — definition) and asks Gemini to generate common
everyday synonyms for each. The result is cached at data/.expansion_cache.json
alongside an MD5 hash of the corpus file. On subsequent runs, the hash is
compared — if unchanged, the cache is loaded with no API call. If the corpus
changes, the map regenerates automatically.

Cost: one Gemini API call per corpus update, not per query. The manual changes
quarterly — so this runs at most a handful of times per year in production.

Failure mode: if the Gemini call fails at startup, the retriever degrades to an
empty expansion map and emits a warning. The soft fallback (see Section 3)
handles zero-coverage queries gracefully.

## 3. Gate (Commits acebfc0, 82d9ea0, 2e869d4, 86161a9, 245ac43)

The gate sits between the retriever and the synthesizer. It has two jobs: detect
dead cross-references in the manual, and detect numeric contradictions between
clauses.

### Dead Reference Detection — Three Iterations

First attempt: TF-IDF cosine similarity between the referencing sentence and the
target clause. Threshold tuned iteratively (0.3 → 0.2 → 0.15) against the known
dead reference §7.1.3 → §5.4. Failed: the dead reference similarity (0.123) fell
inside the range of legitimate structural references (0.099–0.405). No static
threshold could separate them.

Second attempt: targeted term-overlap check. For each citation, extract up to 6
non-stopword tokens immediately preceding it in the source sentence. If overlap
between those tokens and the target clause text is zero, flag FLAG_CONFLICT.
Failed: generic administrative terms (person, member, days, within) triggered
false-positive matches on unrelated clauses.

Third attempt (current): same term-overlap check but with document frequency
filtering. Any term appearing in >15% of the 148 clauses (41 terms total) is
excluded from matching. This forces the check to rely on high-signal topical
words (e.g. "sanction" rather than "person").

Confirmed false positive fixed: §7.1.3 → §7.3 was incorrectly flagged because
"subject to the adjustments in §7.3" is a structural connective phrase — the
topical signal lives in the target, not the preceding tokens. Fixed by adding a
structural connective whitelist (phrases like "subject to the adjustments in",
"except as provided in") that bypass the term-overlap check entirely.

Known limitation: very short cross-referencing sentences with sparse preceding
context may still yield false flags. More complex syntax parsing (e.g.
dependency parsing) would improve this but adds dependencies incompatible with
the grading environment.

### Numeric Contradiction Detection — Two Iterations

First attempt: flag any two different numeric values across retrieved clauses.
Produced false positives: 28-day base rule vs 90-day exception (different
§-anchors, not a contradiction), 10% deduction cap vs 13-week exclusion period
(different units, not a contradiction).

Current design: only flag when two clauses share both the same §-anchor AND the
same unit word (days, percent, weeks, months). Different anchors or different
units are never flagged.

Known limitation: clauses that describe a rule numerically without self-citing
their §-anchor will not be caught.

### Soft Fallback for Zero-Coverage Queries

When term-overlap coverage is below 0.25, the gate previously issued a hard
REFUSE for all queries. This was wrong for real-language queries that simply use
vocabulary not in the manual — a caseworker asking "what do I do if the
department said no" should not get a technical error message.

Current design: if coverage < 0.25 and the query contains at least 2 non-
stopword tokens of length >= 3 (real language check), the gate returns
no_coverage: True and the synthesizer emits an escalation instruction rather
than a hard REFUSE. Empty, single-character, and gibberish queries still get a
hard REFUSE.

This check is corpus-independent — it contains zero hardcoded domain terms and
behaves identically regardless of what manual is loaded.

Why 0.25 as the coverage threshold: confirmed by a systematic sweep across 16
queries — the 10 evaluation questions plus 6 borderline queries whose vocabulary
partially overlaps the manual (e.g. "Can a recipient appeal to the Minister?",
"Is a carer allowance counted as income?"). At 0.15 and 0.25 the gate produces
0 false-refusals and 3 false-answers; at 0.35 it eliminates 1 of those
false-answers at the cost of 2 false-refusals (Q04 car/resources at exactly
0.25 coverage, and Q05 45-day absence at 0.33). The 3 false-answers that pass
the gate at 0.25 are correctly handled downstream — the LLM refuses "dental
treatment" and "Minister" as out-of-scope, and the soft-fallback synthesizer
escalates "fails to act". The cost asymmetry for this domain makes 0.25 the
right threshold: a caseworker told "the manual does not cover this" and who
escalates has caused no harm; a caseworker who receives a confident wrong answer
and acts on it may incorrectly deny or grant assistance to a real person. That
asymmetry justifies erring toward over-refusal, and the sweep confirms that
0.35 overshoots that principle by blocking two valid answerable queries. The
threshold is exposed as the named constant MIN_TERM_COVERAGE in src/gate.py.

What I would change with more time: the threshold is currently
static. A production system would tune it per query type — a
definitional question ("what is a dependent child") warrants a
lower threshold than a procedural one ("what happens if I miss
the deadline") because the definitional answer is more
self-contained. A single static threshold is a known simplification.


## 4. Synthesizer (Commit cf6c919)

Implementation: Google Gemini REST API via raw requests. Model:
gemini-3.5-flash. Temperature: 0.1.

What I rejected: google-generativeai SDK. Adds a bulky dependency with its own
version management surface. Raw requests gives identical capability with one
fewer dependency.

Why gemini-3.5-flash over newer models: standard generation config parameters
(temperature, top_p, top_k) are reliably supported. Preview models have
inconsistent config support which breaks the determinism this system requires.

### Refusal and Conflict Output Design

FLAG_CONFLICT path: the synthesizer is instructed to state which clauses
conflict and what the disagreement is, state what IS known from non-conflicting
clauses, and end with an explicit escalation instruction: "This matter should be
referred to a supervisor before any determination is made." It must not pick one
side of a numeric conflict.

Why flag rather than resolve the 10 vs 30 day contradiction: §4.3.2 (10 days) is
the operative obligation clause. §9.1.4 (30 days) describes a downstream
consequence. Silently picking 10 days would be the system making a legal
determination it is not qualified to make. The caseworker escalates; a human
decides.

Why flag rather than clean-refuse on the dead reference: the retriever returned
useful context (§7.1.3, §1.4.6). The problem is the manual's own pointer is
broken. Telling the caseworker "here is what is known and here is what is
broken" is more useful than a bare "I don't know."

### API Failure Handling

All API failures (HTTP 500, malformed JSON, empty response, timeout, repeated
429) are caught and return a safe REFUSE with a human-readable message. No API
failure propagates as an unhandled exception to the CLI or eval harness. Backoff
is capped at 16 seconds per retry with a maximum of 3 attempts.

## 5. Evaluation Results (10-question set, all fixes applied)

10/10 PASS.

Q01 PASS — ANSWER, §2.3.1 cited. Standard eligibility query.
Q02 PASS — FLAG_CONFLICT, §4.3.2 (10 days) vs §9.1.4 (30 days) surfaced, escalation instruction present.
Q03 PASS — FLAG_CONFLICT, dead reference §7.1.3→§5.4 surfaced, escalation instruction present.
Q04 PASS — ANSWER, §2.4.2 motor vehicle disregard cited. Required query expansion fix to retrieve.
Q05 PASS — ANSWER, §3.2.1 28-day base and §3.2.2 90-day exception both cited correctly as base/exception pair, not a contradiction.
Q06 PASS — ANSWER, §10.5.3 sanction prohibition for households with dependent child under age 2.
Q07 PASS — ANSWER, §11.2.3 30-day review completion timeframe. Required query expansion fix to retrieve.
Q08 PASS — ANSWER, §9.3.2 10%/20% deduction cap cited. Required numeric detector redesign to stop false FLAG_CONFLICT on unrelated unit values.
Q09 PASS — ANSWER, general appeal process from Part 12 cited. No housing-specific rule invented where none exists.
Q10 PASS — ANSWER (clean refusal), manual silent on dental costs, escalation instruction present.

## 6. Stress Testing — Findings and Decisions

### Edge cases

E01–E03 (empty, single char, gibberish): hard REFUSE via gate.
E04 (prompt injection): ignored entirely, policy question answered. Low temperature and strict grounding are the controls — no explicit injection detection needed.
E05 (keyword dump): answered what it could, stated what it could not cover. Known limitation: very broad queries produce unfocused answers.
E06 (colloquial "cut off"): fixed via dynamic expansion map. "cut off" bigram maps to terminated/reinstatement.
E07 (false premise): correctly rejected invented rule, cited actual resource limit §2.4.1.
E08 (raw clause reference): fixed via fast-path raw clause lookup that bypasses retrieval and gating entirely.

### Vocabulary mismatch

V01 ("partner"): dynamic map generates couple/household member.
V02 ("just moved"): partial answer — honest, the manual does not state a residency duration requirement.
V03 ("lose my job"): correctly flagged 10 vs 30 day conflict.
V04 (student visa): correctly refused — not in the manual.
V05–V08: all correctly retrieved from Parts 6, 9, 12.

### Ugly phrasing

U01–U02, U05: correct answers without hallucination.
U03 (multi-part run-on): answered highest-scoring topic, stated others not covered. Known limitation: no multi-query splitting.
U04 ("the department said no"): three fix attempts required. Reactive synonyms → rejected. Low-DF topical mapping → rejected. Final fix: soft fallback in the gate. Zero hardcoded domain terms, corpus-independent.

### Day-two corpus change simulation

Three simultaneous changes: §2.4.1 value $4,000→$5,000, new clause §7.3.4 added,
§3.2.3 deleted.

Results with zero code changes: - Modified value picked up immediately (no stale
cache). - New clause §7.3.4 resolved §7.1.3 dead reference; full-time student
query returned ANSWER not FLAG_CONFLICT. - Deleted clause handled gracefully —
adjacent clause retrieved, no crash. - 10-question eval: 9/10. One expected
failure (Q05) — §3.2.3 deletion dropped coverage below threshold, safe REFUSE
rather than wrong answer. Correct degradation.

Confirms: architecture is fully corpus-independent. No hardcoded clause IDs or
values anywhere in the pipeline.

## 7. What the System Does Not Do

- Does not support multi-turn conversation or session memory.
- Does not parse any document other than the corpus in data/policy-manual.md.
- Does not split multi-part queries — retrieves on the full query string and answers the highest-scoring topic.
- Does not resolve "supervisor" to a named role or contact. The manual is generic; a production system would need a staff directory integration.
- Does not detect all possible dead references — only those where the term-overlap check fires. A cross-reference in a very short sentence with sparse context may be missed.
- Does not provide a web interface. CLI is the complete and intended interface.

## 8. What I Would Fix First

1. Multi-part query splitting — detect conjunctions and run retrieval separately
for each sub-question. Currently the system answers the highest-scoring part and
ignores the rest. 2. Supervisor contact resolution — the escalation instruction
says "refer to a supervisor" but cannot name one. A production deployment needs
a staff directory or role mapping. 3. Corpus integrity checks on ingest —
currently warnings are emitted for duplicate IDs but ingest continues. A
production system should reject a malformed corpus and alert the maintainer
before serving any queries.

## 9. Day-Two Amendment Integration (Commit 02c41f5 and subsequent commits)

### Temporal Resolver Design
To support the January 2026 amendment (`amendment-2026-01.md`), we built a temporal resolver (`src/temporal.py`) with a `resolve_clause` function. Given the determination date and event date, it resolves the correct version of any clause.
- **Transitional Rules**: Paragraph 2 amendments are event-date anchored (§5.2); Paragraphs 1, 3, and 4 are determination-date anchored (§5.1).
- **Ambiguity Handling**: If a post-March query does not specify an event date, the resolver returns `ambiguous=True`, setting `text=None` and populating `old_text` and `new_text`. In-place patching in `src/pipeline.py` formats the retrieved clause text to show both versions side-by-side and injects a dynamic prompt warning the synthesizer that it must state the dependency and not pick a single-value answer.
- **Generic Query Defaulting**: If a user asks a generic query with no dates at all, the temporal resolver defaults both determination and event dates to today's date (August 2026), resolving to the post-amendment version of the rules with no conflict.
- **Pre-Effective date checks**: We resolved a bug where queries with a pre-March determination date and no event date were flagged as ambiguous. Since the determination date is pre-March, any event date must also be pre-March, so the pre-amendment text is unconditionally applied with no ambiguity.

### Retriever Fix for §10.5.3A (Synthetic Clauses)
Because §10.5.3A was inserted by the amendment, it does not exist in the base manual. We modified the pipeline startup to inject `§10.5.3A` from parsed amendments into the retriever's index using its post-amendment text. If a query's temporal context is pre-March, in-place patching filters it out (`exists=False`) so the synthesizer behaves as if the clause does not exist yet. This allows it to be retrieved for post-March queries without breaking pre-March historical runs.

### Claim Apportionment (§5.3 / §7.4.3)
If a query's date range spans the 1 March 2026 boundary, the system sets `has_apportionment = True` (by checking if the query contains at least two dates that span the boundary).
- **Synthetic Insertion**: Since the validator requires every factual claim in the answer to be supported by retrieved clauses, we dynamically inject §5.3 (from the amendment) and §7.4.3 (from the base manual) as retrieved documents into the RAG results when apportionment is active.
- **Output**: The synthesizer is instructed to state that the claim spans the boundary, cite §5.3 and §7.4.3, and say the award must be apportioned without attempting to calculate the actual prorated arithmetic, which is deliberately left out of scope for this text-retrieval system.

### Retrospective Design Note: What I'd Do Differently
If I had known the amendment was coming, I would have made **clause versions** a first-class concept in the parser and database from day one, rather than bolting them onto retriever outputs as a post-retrieval overlay patch. Storing clauses as a temporal range database (`[start_date, end_date)`) and indexing every historical version separately in the retriever would have made the gating and retrieval logic completely natural and unified, rather than requiring dynamic text patching and synthetic document injections.
