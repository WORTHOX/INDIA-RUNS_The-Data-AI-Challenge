from __future__ import annotations

from copy import deepcopy

from redrob_ranker.features.normalize import build_normalized_candidate
from redrob_ranker.features.title_alignment import extract_title_features
from redrob_ranker.job_description import build_job_contract


def test_extract_title_features_rewards_ai_and_search_titles(sample_candidate):
    candidate = deepcopy(sample_candidate)
    candidate["profile"]["current_title"] = "ML Engineer"
    candidate["career_history"][0]["title"] = "Search Engineer"
    candidate["career_history"][1]["title"] = "Data Scientist"

    normalized = build_normalized_candidate(candidate)
    features = extract_title_features(normalized)

    assert features["title_score"] > 0.6
    assert features["negative_title_score"] == 0.0
    assert "ml engineer" in features["title_evidence"]


def test_extract_title_features_penalizes_non_engineering_title_families(sample_candidate):
    candidate = deepcopy(sample_candidate)
    candidate["profile"]["current_title"] = "Marketing Manager"
    candidate["career_history"][0]["title"] = "Content Writer"
    candidate["career_history"][1]["title"] = "Sales Executive"

    normalized = build_normalized_candidate(candidate)
    features = extract_title_features(normalized)

    assert features["title_score"] == 0.0
    assert features["negative_title_score"] > 0.6
    assert "marketing manager" in features["negative_title_evidence"]


def test_extract_title_features_supports_unseen_dynamic_jd_titles(sample_candidate):
    candidate = deepcopy(sample_candidate)
    candidate["profile"]["current_title"] = "Compliance Analyst"
    for role in candidate["career_history"]:
        role["title"] = "Compliance Analyst"

    contract = build_job_contract("Title: Senior Compliance Analyst\nOwn regulatory audits and risk controls.")
    normalized = build_normalized_candidate(candidate)
    features = extract_title_features(normalized, contract)

    assert features["title_score"] == 1.0
    assert features["negative_title_score"] == 0.0


def test_extract_title_features_treats_jd_specific_engineering_hint_as_strong_adjacent_title(sample_candidate):
    candidate = deepcopy(sample_candidate)
    candidate["profile"]["current_title"] = "NLP Engineer"
    for role in candidate["career_history"]:
        role["title"] = "NLP Engineer"

    contract = build_job_contract(
        """
        Job Description: Senior AI Engineer
        Need retrieval, ranking, embeddings, Python, and significant NLP/IR exposure.
        """
    )
    normalized = build_normalized_candidate(candidate)
    features = extract_title_features(normalized, contract)

    assert features["title_score"] >= 0.8
    assert features["negative_title_score"] == 0.0


def test_extract_title_features_does_not_reuse_ai_hints_for_unrelated_dynamic_jd(sample_candidate):
    candidate = deepcopy(sample_candidate)
    candidate["profile"]["current_title"] = "NLP Engineer"
    for role in candidate["career_history"]:
        role["title"] = "NLP Engineer"

    contract = build_job_contract("Job Description: Marketing Manager\nOwn SEO, campaigns, and brand strategy.")
    normalized = build_normalized_candidate(candidate)
    features = extract_title_features(normalized, contract)

    assert features["title_score"] == 0.0
