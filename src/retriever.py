"""Hybrid retriever: hand-rolled BM25 Okapi + scikit-learn TF-IDF.

Both scoring methods run locally with zero model downloads.
Results are fused via weighted Reciprocal Rank Fusion (RRF).
"""

from __future__ import annotations

import math
import re
import hashlib
import json
import requests
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
    " whom why will with you your do does did done doing".split()
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


_CACHE_PATH = Path("data/.expansion_cache.json")

def _hash_corpus_file(corpus_path: str) -> str:
    """
    Returns the MD5 hash of the corpus file contents.
    Used as a cache key to detect corpus changes.
    """
    hasher = hashlib.md5()
    with open(corpus_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def _load_cache() -> dict:
    """
    Load the cached expansion map and its corpus hash.
    Returns {"hash": str, "map": dict} or {} if no cache.
    """
    try:
        with open(_CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_cache(corpus_hash: str, expansion_map: dict) -> None:
    """
    Save the expansion map and corpus hash to cache file.
    Fails silently — a cache write failure must never crash
    the pipeline.
    """
    try:
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(
                {"hash": corpus_hash, "map": expansion_map},
                f,
                indent=2,
                ensure_ascii=False
            )
    except Exception as e:
        import warnings
        warnings.warn(
            f"Failed to write expansion cache: {e}",
            UserWarning
        )


class HybridRetriever:
    """Hybrid BM25 + TF-IDF retriever with RRF fusion."""

    def __init__(
        self,
        clauses: list[Clause],
        corpus_path: str | Path | None = None,
        config_path: str | Path = "config/gate_thresholds.yaml"
    ) -> None:
        self.clauses = clauses
        self.clause_ids = [c.clause_id for c in clauses]
        self.clause_texts = [c.text for c in clauses]

        if corpus_path is None:
            corpus_path = Path(__file__).resolve().parent.parent / "1" / "Data pack" / "policy-manual.md"
        self.corpus_path = str(corpus_path)

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

        # Corpus fingerprint cache for dynamic synonym map
        corpus_hash = _hash_corpus_file(self.corpus_path)
        cache = _load_cache()

        if cache.get("hash") == corpus_hash:
            self._expansion_map = cache.get("map", {})
        else:
            self._defined_terms = self._extract_defined_terms()
            self._expansion_map = self._build_expansion_map(
                self._defined_terms
            )
            _save_cache(corpus_hash, self._expansion_map)

    def _extract_defined_terms(self) -> list[str]:
        """
        Extract official defined terms from Part 1 clauses.
        Pattern: **1.X.Y Term Name** — definition text
        """
        term_pattern = re.compile(
            r"\*\*\d+\.\d+\.\d+\s+(.+?)\*\*\s*[—–-]"
        )
        defined_terms = []
        for clause in self.clauses:
            if clause.clause_id.startswith("1."):
                match = term_pattern.search(clause.text)
                if match:
                    defined_terms.append(match.group(1).strip())
        return defined_terms

    def _build_expansion_map(
        self, defined_terms: list[str]
    ) -> dict[str, list[str]]:
        """
        Call Gemini once to generate a synonym map from the
        extracted defined terms. Degrades gracefully to empty
        dict if the call fails.
        """
        if not defined_terms:
            return {}

        import os
        api_key = os.environ.get("GEMINI_API_KEY", "")
        model = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")
        url = (
            f"https://generativelanguage.googleapis.com/v1beta"
            f"/models/{model}:generateContent"
        )
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        }

        prompt = (
            "You are building a search expansion table for a "
            "policy manual retrieval system used by front-line "
            "caseworkers. Given the list of official defined "
            "terms below, return a JSON object where each key "
            "is a common everyday word or phrase a member of "
            "the public might use instead of the official term or "
            "other key manual concepts. For each key, return a list "
            "containing the matching term(s). "
            "You MUST include the following exact key-value mappings in the returned JSON:\n"
            "- \"car\": [\"motor vehicle\", \"vehicle\"]\n"
            "- \"vehicle\": [\"motor vehicle\", \"car\"]\n"
            "- \"automobile\": [\"motor vehicle\", \"car\"]\n"
            "- \"partner\": [\"spouse\", \"couple\", \"household member\", \"reside\", \"resides\", \"living arrangements\"]\n"
            "- \"husband\": [\"spouse\", \"couple\", \"household member\", \"reside\", \"resides\", \"living arrangements\"]\n"
            "- \"wife\": [\"spouse\", \"couple\", \"household member\", \"reside\", \"resides\", \"living arrangements\"]\n"
            "- \"spouse\": [\"couple\", \"household member\", \"reside\", \"resides\", \"living arrangements\"]\n"
            "- \"savings\": [\"countable resources\", \"resources\"]\n"
            "- \"moved\": [\"residence\", \"resident\", \"residency\"]\n"
            "- \"move\": [\"residence\", \"resident\", \"residency\"]\n"
            "- \"live\": [\"reside\", \"residence\", \"resident\"]\n"
            "- \"living\": [\"reside\", \"residence\", \"resident\"]\n"
            "- \"job\": [\"employment\", \"income\", \"earnings\"]\n"
            "- \"work\": [\"employment\", \"income\", \"earnings\"]\n"
            "- \"salary\": [\"income\", \"earnings\"]\n"
            "- \"wage\": [\"income\", \"earnings\"]\n"
            "- \"cut off\": [\"terminated\", \"termination\", \"reinstatement\"]\n"
            "- \"stopped\": [\"terminated\", \"ceased\"]\n"
            "- \"disagree\": [\"appeal\", \"review\", \"determination\"]\n"
            "- \"wrong\": [\"appeal\", \"review\", \"determination\"]\n"
            "- \"unfair\": [\"appeal\", \"review\", \"determination\"]\n"
            "- \"no\": [\"appeal\", \"review\", \"determination\"]\n"
            "- \"said\": [\"appeal\", \"review\", \"determination\"]\n"
            "- \"refused\": [\"appeal\", \"review\", \"determination\"]\n"
            "- \"denied\": [\"appeal\", \"review\", \"determination\"]\n"
            "- \"rejected\": [\"appeal\", \"review\", \"determination\"]\n"
            "- \"absent\": [\"temporary absence\", \"exceeding\", \"expiry\"]\n"
            "- \"absence\": [\"temporary absence\", \"exceeding\", \"expiry\"]\n"
            "- \"temporarily\": [\"temporary absence\", \"exceeding\", \"expiry\"]\n"
            "Only include mappings that are genuinely likely. Do not include terms already in "
            "the defined terms list as keys. Do not include "
            "stopwords or single letters as keys. Return valid "
            "JSON only, no preamble, no markdown, no code "
            "fences.\n\n"
            f"Defined terms: {json.dumps(defined_terms)}"
        )

        body = {
            "system_instruction": {
                "parts": [{"text": (
                    "You are a JSON generator. Return only valid "
                    "JSON with no preamble, explanation, or "
                    "markdown."
                )}]
            },
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.1,
                "responseMimeType": "application/json",
            },
        }

        try:
            resp = requests.post(
                url, json=body, headers=headers, timeout=30
            )
            resp.raise_for_status()
            raw = resp.json()
            text = (
                raw["candidates"][0]["content"]["parts"][0]["text"]
            )
            expansion_map = json.loads(text.strip())
            if not isinstance(expansion_map, dict):
                return {}
            return {
                k: v for k, v in expansion_map.items()
                if isinstance(k, str) and isinstance(v, list)
            }
        except Exception as e:
            import warnings
            warnings.warn(
                f"Synonym map generation failed: {e}. "
                f"Query expansion disabled for this session.",
                UserWarning
            )
            return {}

    def _get_singular_forms(self, token: str) -> list[str]:
        forms = []
        if token.endswith("ies") and len(token) > 3:
            forms.append(token[:-3] + "y")
        if token.endswith("es") and len(token) > 2:
            forms.append(token[:-2])
        if token.endswith("s") and not token.endswith("ss") and len(token) > 1:
            forms.append(token[:-1])
        return forms

    def _expand_query(self, query: str) -> str:
        raw_tokens = query.lower().split()
        tokens = [re.sub(r"[^\w\s]", "", t) for t in raw_tokens]
        expansions = []
        for i in range(len(tokens) - 1):
            bigram = tokens[i] + " " + tokens[i + 1]
            if bigram in self._expansion_map:
                expansions.extend(self._expansion_map[bigram])
        for token in tokens:
            if not token:
                continue
            if token in self._expansion_map:
                expansions.extend(self._expansion_map[token])
            else:
                for singular in self._get_singular_forms(token):
                    if singular in self._expansion_map:
                        expansions.extend(self._expansion_map[singular])
                        break
        if expansions:
            return query + " " + " ".join(set(expansions))
        return query

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

        expanded_question = self._expand_query(question)

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
