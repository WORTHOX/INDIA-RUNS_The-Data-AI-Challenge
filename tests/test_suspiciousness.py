from __future__ import annotations

from copy import deepcopy

from redrob_ranker.features.honeypot import extract_honeypot_features
from redrob_ranker.features.normalize import build_normalized_candidate
from redrob_ranker.features.skills import extract_skill_features
from redrob_ranker.features.title_alignment import extract_title_features
from redrob_ranker.scoring.suspiciousness import compute_suspiciousness_penalty


def test_compute_suspiciousness_penalty_catches_keyword_stuffer(sample_candidate):
    candidate = deepcopy(sample_candidate)
    candidate["profile"]["current_title"] = "Marketing Manager"
    candidate["career_history"][0]["title"] = "Marketing Manager"
    candidate["career_history"][1]["title"] = "Content Writer"
    candidate["skills"] = [
        {"name": f"AI Skill {index}", "proficiency": "expert", "endorsements": 1, "duration_months": 4}
        for index in range(12)
    ]

    normalized = build_normalized_candidate(candidate)
    title_features = extract_title_features(normalized)
    skill_features = extract_skill_features(normalized)
    honeypot_features = extract_honeypot_features(normalized)

    penalty = compute_suspiciousness_penalty(
        normalized,
        title_features=title_features,
        skill_features=skill_features,
        honeypot_features=honeypot_features,
    )

    assert penalty >= 0.35


def test_compute_suspiciousness_penalty_penalizes_clear_wrong_title_without_ai_stuffing(sample_candidate):
    candidate = deepcopy(sample_candidate)
    candidate["profile"]["current_title"] = "Marketing Manager"
    candidate["career_history"][0]["title"] = "Marketing Manager"
    candidate["career_history"][1]["title"] = "Content Writer"
    candidate["skills"] = [
        {"name": "SEO", "proficiency": "intermediate", "endorsements": 4, "duration_months": 18},
        {"name": "Content Writing", "proficiency": "intermediate", "endorsements": 5, "duration_months": 24},
    ]

    normalized = build_normalized_candidate(candidate)
    title_features = extract_title_features(normalized)
    skill_features = extract_skill_features(normalized)
    honeypot_features = extract_honeypot_features(normalized)

    penalty = compute_suspiciousness_penalty(
        normalized,
        title_features=title_features,
        skill_features=skill_features,
        honeypot_features=honeypot_features,
    )

    assert penalty >= 0.20


def test_compute_suspiciousness_penalty_uses_summary_only_and_title_description_noise():
    row = {
        "negative_title_score": 1.0,
        "title_score": 0.0,
        "ai_like_skill_count": 0,
        "expert_skill_count": 0,
        "short_duration_expert_count": 0,
        "claimed_experience_gap_years": 0.0,
        "research_only_score": 0.0,
        "service_background_score": 0.0,
        "total_roles": 2,
        "summary_only_match_score": 0.8,
        "title_description_mismatch_score": 0.65,
    }

    penalty = compute_suspiciousness_penalty(
        row,
        title_features=row,
        skill_features=row,
        honeypot_features=row,
    )

    assert penalty >= 0.45
