from __future__ import annotations

from copy import deepcopy

from redrob_ranker.features.normalize import build_normalized_candidate
from redrob_ranker.features.skills import extract_skill_features


def test_extract_skill_features_captures_retrieval_vector_and_python_support(sample_candidate):
    candidate = deepcopy(sample_candidate)
    candidate["skills"] = [
        {"name": "Python", "proficiency": "expert", "endorsements": 21, "duration_months": 48},
        {"name": "Pinecone", "proficiency": "advanced", "endorsements": 9, "duration_months": 24},
        {"name": "FAISS", "proficiency": "advanced", "endorsements": 6, "duration_months": 24},
        {"name": "Embeddings", "proficiency": "advanced", "endorsements": 8, "duration_months": 20},
        {"name": "NDCG", "proficiency": "advanced", "endorsements": 3, "duration_months": 14},
    ]

    normalized = build_normalized_candidate(candidate)
    features = extract_skill_features(normalized)

    assert features["python_score"] > 0.5
    assert features["vector_score"] > 0.6
    assert features["evaluation_skill_score"] > 0.2
    assert "pinecone" in features["skill_evidence"]


def test_extract_skill_features_flags_short_duration_expert_stuffing(sample_candidate):
    candidate = deepcopy(sample_candidate)
    candidate["skills"] = [
        {"name": f"AI Skill {index}", "proficiency": "expert", "endorsements": 1, "duration_months": 3}
        for index in range(12)
    ]

    normalized = build_normalized_candidate(candidate)
    features = extract_skill_features(normalized)

    assert features["expert_skill_count"] == 12
    assert features["short_duration_expert_count"] == 12
