"""Hybrid retriever: hand-rolled BM25 Okapi + scikit-learn TF-IDF.

Both scoring methods run locally with zero model downloads.
Results are fused via weighted Reciprocal Rank Fusion (RRF).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .parser import Clause

# ── Stopwords ────────────────────────────────────────────────────────────────
# Frozen set — no download required.
STOPWORDS: frozenset[str] = frozenset(
    "a an and are as at be but by for from has have he her his i if in into is"
    " it its me my no nor not of on or our own she so than that the their them"
    " then there these they this to up us was we were what when where which who"
    " whom why will with you your".split()
)


def _tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumeric, drop stopwords."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in STOPWORDS]


# ── Result dataclass ─────────────────────────────────────────────────────────

@dataclass
class RetrievalResult:
    """A single retrieval hit."""

    clause_id: str
    clause_text: str
    score: float  # fused RRF score
    bm25_rank: int
    tfidf_rank: int


# ── BM25 Okapi (hand-rolled) ────────────────────────────────────────────────

class BM25:
    """BM25 Okapi scorer.  k1=1.5, b=0.75 per spec."""

    def __init__(self, corpus_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.corpus_tokens = corpus_tokens
        self.n_docs = len(corpus_tokens)
        self.doc_lens = [len(d) for d in corpus_tokens]
        self.avgdl = sum(self.doc_lens) / self.n_docs if self.n_docs else 1.0

        # Document frequency for each term
        self.df: dict[str, int] = {}
        for doc in corpus_tokens:
            for term in set(doc):
                self.df[term] = self.df.get(term, 0) + 1

        # Pre-compute IDF
        self.idf: dict[str, float] = {}
        for term, df in self.df.items():
            # Standard BM25 IDF: log((N - df + 0.5) / (df + 0.5) + 1)
            self.idf[term] = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0)

        # Term frequencies per document
        self.tf: list[dict[str, int]] = []
        for doc in corpus_tokens:
            tf: dict[str, int] = {}
            for t in doc:
                tf[t] = tf.get(t, 0) + 1
            self.tf.append(tf)

    def score(self, query_tokens: list[str]) -> list[float]:
        """Return BM25 scores for every document given query tokens."""
        scores = [0.0] * self.n_docs
        for q in query_tokens:
            if q not in self.idf:
                continue
            idf_q = self.idf[q]
            for i in range(self.n_docs):
                tf_qi = self.tf[i].get(q, 0)
                if tf_qi == 0:
                    continue
                dl = self.doc_lens[i]
                numerator = tf_qi * (self.k1 + 1)
                denominator = tf_qi + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[i] += idf_q * numerator / denominator
        return scores


# ── Hybrid Retriever ─────────────────────────────────────────────────────────

_DEFAULT_CONFIG = {
    "retriever": {
        "w_bm25": 1.0,
        "w_tfidf": 1.0,
        "rrf_k": 60,
        "default_top_k": 10,
    }
}


QUERY_EXPANSIONS = {
    "car": ["motor vehicle", "vehicle"],
    "vehicle": ["motor vehicle", "car"],
    "automobile": ["motor vehicle", "car"],
    "how long": ["timeframe", "deadline", "period", "days", "weeks"],
    "complete": ["conclude", "finish", "determine"],
    "review": ["review", "determination", "assessment"],
    
    # colloquial for award termination / reapplication
    "cut off": ["terminated", "termination", "ceased", "reinstatement"],
    "kicked off": ["terminated", "termination", "ceased"],
    "stopped": ["terminated", "ceased", "suspended"],
    "benefits": ["assistance", "award", "program"],
    "help": ["assistance", "eligible", "award"],
    "mum": ["person", "applicant", "recipient"],
    "mom": ["person", "applicant", "recipient"],
    "she": ["person", "applicant", "recipient"],
    "he": ["person", "applicant", "recipient"],
    "they": ["household", "applicant", "recipient"],
    "my": [],   # drop possessives — no expansion needed
    "get": ["apply", "receive", "eligible"],

    "partner":  ["spouse", "couple", "household member", "member", "joint"],
    "husband":  ["spouse", "couple", "household member"],
    "wife":     ["spouse", "couple", "household member"],
    "boyfriend": ["couple", "household member", "member"],
    "girlfriend": ["couple", "household member", "member"],
    "family":   ["household", "household member", "dependent child"],
    "kids":     ["dependent child", "children", "child"],
    "child":    ["dependent child"],
    "children": ["dependent child"],
    "job":      ["employment", "income", "earnings"],
    "work":     ["employment", "income", "earnings"],
    "salary":   ["income", "earnings", "countable income"],
    "wage":     ["income", "earnings", "countable income"],
    "moved":    ["residence", "resident", "residency", "reside"],
    "move":     ["residence", "resident", "residency"],
    "live":     ["reside", "residence", "resident"],
    "living":   ["reside", "residence", "resident"],
    "savings":  ["resources", "countable resources", "resource"],
}


def _expand_query(query: str) -> str:
    # Strip basic punctuation to ensure clean matching
    q_norm = query.lower().replace("?", "").replace(".", "").replace(",", "").replace("!", "")
    tokens = q_norm.split()
    expansions = []
    # Check bigrams first
    for i in range(len(tokens) - 1):
        bigram = tokens[i] + " " + tokens[i+1]
        if bigram in QUERY_EXPANSIONS:
            expansions.extend(QUERY_EXPANSIONS[bigram])
    # Then single tokens
    for token in tokens:
        if token in QUERY_EXPANSIONS:
            expansions.extend(QUERY_EXPANSIONS[token])
    if expansions:
        return query + " " + " ".join(set(expansions))
    return query


class HybridRetriever:
    """Hybrid BM25 + TF-IDF retriever with RRF fusion."""

    def __init__(self, clauses: list[Clause], config_path: str | Path = "config/gate_thresholds.yaml") -> None:
        self.clauses = clauses
        self.clause_ids = [c.clause_id for c in clauses]
        self.clause_texts = [c.text for c in clauses]

        # Load config
        config_path = Path(config_path)
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        else:
            cfg = {}

        ret_cfg = cfg.get("retriever", _DEFAULT_CONFIG["retriever"])
        self.w_bm25: float = ret_cfg.get("w_bm25", 1.0)
        self.w_tfidf: float = ret_cfg.get("w_tfidf", 1.0)
        self.rrf_k: int = ret_cfg.get("rrf_k", 60)
        self.default_top_k: int = ret_cfg.get("default_top_k", 10)

        # Build BM25 index
        self._corpus_tokens = [_tokenize(t) for t in self.clause_texts]
        self._bm25 = BM25(self._corpus_tokens)

        # Build TF-IDF index
        self._tfidf_vec = TfidfVectorizer(
            lowercase=True,
            token_pattern=r"[a-zA-Z0-9]+",
            stop_words="english",
        )
        self._tfidf_matrix = self._tfidf_vec.fit_transform(self.clause_texts)

    def query(self, question: str, top_k: int | None = None) -> list[RetrievalResult]:
        """Retrieve the top-k clauses for a question.

        Args:
            question: The user's query string.
            top_k: Number of results to return.  Defaults to config value.

        Returns:
            Ranked list of RetrievalResult, highest score first.
        """
        if top_k is None:
            top_k = self.default_top_k

        n = len(self.clauses)

        expanded_question = _expand_query(question)

        # ── BM25 scores & ranks ──
        bm25_scores = self._bm25.score(_tokenize(expanded_question))
        bm25_ranking = sorted(range(n), key=lambda i: bm25_scores[i], reverse=True)
        bm25_rank_map = {idx: rank for rank, idx in enumerate(bm25_ranking)}

        # ── TF-IDF scores & ranks ──
        q_vec = self._tfidf_vec.transform([expanded_question])
        tfidf_scores = cosine_similarity(q_vec, self._tfidf_matrix).flatten()
        tfidf_ranking = sorted(range(n), key=lambda i: tfidf_scores[i], reverse=True)
        tfidf_rank_map = {idx: rank for rank, idx in enumerate(tfidf_ranking)}

        # ── Reciprocal Rank Fusion ──
        rrf_scores: list[float] = []
        for i in range(n):
            s = (self.w_bm25 / (self.rrf_k + bm25_rank_map[i])
                 + self.w_tfidf / (self.rrf_k + tfidf_rank_map[i]))
            rrf_scores.append(s)

        # Sort by fused score
        fused_ranking = sorted(range(n), key=lambda i: rrf_scores[i], reverse=True)

        results: list[RetrievalResult] = []
        for i in fused_ranking[:top_k]:
            results.append(RetrievalResult(
                clause_id=self.clause_ids[i],
                clause_text=self.clause_texts[i],
                score=rrf_scores[i],
                bm25_rank=bm25_rank_map[i],
                tfidf_rank=tfidf_rank_map[i],
            ))
        return results
