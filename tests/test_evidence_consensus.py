from __future__ import annotations

from copy import deepcopy

from redrob_ranker.features.evidence_consensus import extract_evidence_consensus_features
from redrob_ranker.features.normalize import build_normalized_candidate
from redrob_ranker.job_description import build_job_contract


def test_evidence_consensus_rewards_requirements_across_title_career_and_skills(sample_candidate):
    candidate = deepcopy(sample_candidate)
    candidate["profile"]["current_title"] = "Senior Data Engineer"
    candidate["career_history"][0]["title"] = "Senior Data Engineer"
    candidate["career_history"][0]["description"] = "Built production data pipelines using Python, Spark, and Airflow."
    candidate["skills"] = [
        {"name": "Python", "proficiency": "advanced", "endorsements": 12, "duration_months": 48},
        {"name": "Spark", "proficiency": "advanced", "endorsements": 10, "duration_months": 42},
        {"name": "Airflow", "proficiency": "intermediate", "endorsements": 8, "duration_months": 36},
    ]
    contract = build_job_contract(
        "Role: Senior Data Engineer\nMust have: Python, Spark, Airflow, data pipelines.\nNice to have: dbt."
    )
    normalized = build_normalized_candidate(candidate)

    features = extract_evidence_consensus_features(
        normalized,
        contract,
        title_features={"title_score": 1.0, "negative_title_score": 0.0},
        career_features={"job_keyword_score": 0.9, "retrieval_score": 0.0, "evaluation_score": 0.0},
        skill_features={"job_skill_score": 0.75, "retrieval_skill_score": 0.0, "python_score": 1.0},
    )

    assert features["must_have_coverage_score"] >= 0.75
    assert features["evidence_channel_count"] >= 3
    assert features["evidence_consensus_score"] >= 0.75
    assert features["summary_only_match_score"] == 0.0


def test_evidence_consensus_flags_summary_only_keyword_match(sample_candidate):
    candidate = deepcopy(sample_candidate)
    candidate["profile"]["current_title"] = "Marketing Manager"
    candidate["profile"]["summary"] = "Interested in Python, Spark, Airflow, and data pipelines."
    candidate["career_history"][0]["title"] = "Marketing Manager"
    candidate["career_history"][0]["description"] = "Owned campaign planning and content calendars."
    candidate["skills"] = [
        {"name": "Marketing", "proficiency": "advanced", "endorsements": 10, "duration_months": 48},
    ]
    contract = build_job_contract("Role: Senior Data Engineer\nMust have: Python, Spark, Airflow, data pipelines.")
    normalized = build_normalized_candidate(candidate)

    features = extract_evidence_consensus_features(
        normalized,
        contract,
        title_features={"title_score": 0.0, "negative_title_score": 1.0},
        career_features={"job_keyword_score": 0.0, "retrieval_score": 0.0, "evaluation_score": 0.0},
        skill_features={"job_skill_score": 0.0, "retrieval_skill_score": 0.0, "python_score": 0.0},
    )

    assert features["summary_keyword_score"] >= 0.75
    assert features["summary_only_match_score"] >= 0.5
    assert features["title_description_mismatch_score"] == 0.0


def test_evidence_consensus_flags_wrong_title_with_jd_keyword_description(sample_candidate):
    candidate = deepcopy(sample_candidate)
    candidate["profile"]["current_title"] = "Marketing Manager"
    candidate["career_history"][0]["title"] = "Marketing Manager"
    candidate["career_history"][0]["description"] = "Built production data pipelines using Python and Spark."
    contract = build_job_contract("Role: Senior Data Engineer\nMust have: Python, Spark, data pipelines.")
    normalized = build_normalized_candidate(candidate)

    features = extract_evidence_consensus_features(
        normalized,
        contract,
        title_features={"title_score": 0.0, "negative_title_score": 1.0},
        career_features={"job_keyword_score": 0.8, "retrieval_score": 0.0, "evaluation_score": 0.0},
        skill_features={"job_skill_score": 0.0, "retrieval_skill_score": 0.0, "python_score": 0.0},
    )

    assert features["title_description_mismatch_score"] >= 0.5
    assert features["evidence_consensus_score"] < 0.75
