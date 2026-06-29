from __future__ import annotations

from copy import deepcopy

from redrob_ranker.features.career_history import extract_career_features
from redrob_ranker.features.normalize import build_normalized_candidate


def test_extract_career_features_finds_retrieval_and_evaluation_evidence(sample_candidate):
    candidate = deepcopy(sample_candidate)
    candidate["profile"]["current_industry"] = "SaaS"
    candidate["career_history"][0]["industry"] = "SaaS"
    candidate["career_history"][0]["description"] = (
        "Shipped a hybrid retrieval and ranking system in production using Elasticsearch, "
        "FAISS, dense embeddings, and an LLM reranker. Owned NDCG and MRR offline evaluation."
    )
    candidate["career_history"][1]["description"] = (
        "Built recommendation and search pipelines with Python and A/B testing loops."
    )

    normalized = build_normalized_candidate(candidate)
    features = extract_career_features(normalized)

    assert features["career_score"] > 0.5
    assert features["retrieval_score"] > 0.7
    assert features["evaluation_score"] > 0.4
    assert features["product_background_score"] > 0.4
    assert "elasticsearch" in features["career_evidence"]


def test_extract_career_features_downweights_service_only_history(sample_candidate):
    normalized = build_normalized_candidate(sample_candidate)

    features = extract_career_features(normalized)

    assert features["service_background_score"] >= 0.5
