"""Semantic text matching against the supplied JD.

Semantic scores are useful for synonyms and softer matches, but they are capped
and gated so summary-only keyword matches cannot outrank real structural
evidence.
"""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from redrob_ranker.jd_contract import DEFAULT_JD_CONTRACT, JobContract
from redrob_ranker.types import NormalizedCandidate

DEFAULT_DENSE_MODEL = "BAAI/bge-small-en-v1.5"


def build_query_text(contract: JobContract = DEFAULT_JD_CONTRACT) -> str:
    if contract.job_query_text:
        return contract.job_query_text
    sections = (
        contract.positive_title_terms,
        contract.retrieval_keywords,
        contract.vector_keywords,
        contract.evaluation_keywords,
        contract.python_keywords,
        contract.llm_keywords,
        contract.nlp_keywords,
    )
    return " ".join(term for section in sections for term in section)


def build_candidate_semantic_text(candidate: NormalizedCandidate) -> str:
    return " ".join(
        part
        for part in [
            candidate.headline,
            candidate.summary,
            candidate.profile_text,
            candidate.career_text,
            candidate.skills_text,
        ]
        if part
    )


def compute_tfidf_scores(texts: Iterable[str], query_text: str) -> np.ndarray:
    corpus = [query_text, *texts]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words="english")
    matrix = vectorizer.fit_transform(corpus)
    query_vector = matrix[0]
    candidate_matrix = matrix[1:]
    scores = candidate_matrix @ query_vector.T
    return np.asarray(scores.toarray()).ravel()


def compute_dense_scores(
    texts: Iterable[str],
    query_text: str,
    *,
    enable_dense: bool = True,
    model_name: str = DEFAULT_DENSE_MODEL,
    batch_size: int = 64,
) -> np.ndarray:
    items = list(texts)
    if not enable_dense:
        return np.zeros(len(items), dtype=float)

    try:
        from sentence_transformers import SentenceTransformer
    except Exception:
        return np.zeros(len(items), dtype=float)

    try:
        model = SentenceTransformer(model_name)
        query_embedding = model.encode([query_text], normalize_embeddings=True)
        candidate_embeddings = model.encode(items, batch_size=batch_size, normalize_embeddings=True)
    except Exception:
        return np.zeros(len(items), dtype=float)

    scores = np.dot(candidate_embeddings, query_embedding[0])
    return np.asarray(scores, dtype=float)


def compute_semantic_support(row: Mapping[str, float]) -> float:
    """Return a small semantic boost only when structural evidence exists."""

    structural_relevance = float(row.get("title_score", 0.0)) + float(row.get("career_score", 0.0))
    retrieval_relevance = max(
        float(row.get("retrieval_score", 0.0)),
        float(row.get("retrieval_skill_score", 0.0)),
        float(row.get("vector_score", 0.0)),
        float(row.get("job_keyword_score", 0.0)),
        float(row.get("job_skill_score", 0.0)),
        float(row.get("requirement_coverage_score", 0.0)),
    )
    summary_only_match_score = float(row.get("summary_only_match_score", 0.0))
    evidence_consensus_score = float(row.get("evidence_consensus_score", 0.0))
    # Do not let polished summaries rescue candidates with weak evidence
    # consensus. This directly targets summary keyword stuffing in the dataset.
    if summary_only_match_score >= 0.50 and evidence_consensus_score < 0.50:
        return 0.0
    if structural_relevance < 0.15 and retrieval_relevance <= 0.0:
        return 0.0

    support = min(
        0.15,
        0.10 * float(row.get("tfidf_score", 0.0)) + 0.05 * float(row.get("dense_score", 0.0)),
    )
    return round(max(0.0, support), 6)
