from __future__ import annotations

from copy import deepcopy

from redrob_ranker.features.behavioral import compute_reference_date, extract_behavioral_features
from redrob_ranker.features.normalize import build_normalized_candidate
from redrob_ranker.job_description import build_job_contract


def test_compute_reference_date_uses_max_last_active_date(sample_candidates):
    reference_date = compute_reference_date(
        candidate["redrob_signals"]["last_active_date"] for candidate in sample_candidates[:5]
    )

    assert reference_date.isoformat() == "2026-05-20"


def test_extract_behavioral_features_applies_recency_and_location_support(sample_candidate):
    candidate = deepcopy(sample_candidate)
    candidate["profile"]["location"] = "Pune, Maharashtra"
    candidate["redrob_signals"]["open_to_work_flag"] = True
    candidate["redrob_signals"]["last_active_date"] = "2026-05-15"
    candidate["redrob_signals"]["recruiter_response_rate"] = 0.74
    candidate["redrob_signals"]["notice_period_days"] = 30
    candidate["redrob_signals"]["interview_completion_rate"] = 0.9
    candidate["redrob_signals"]["github_activity_score"] = 60

    normalized = build_normalized_candidate(candidate)
    reference_date = compute_reference_date(["2026-05-20", "2026-04-01"])
    features = extract_behavioral_features(normalized, reference_date)

    assert features["days_since_active"] == 5
    assert features["preferred_location_score"] == 1.0
    assert features["availability_multiplier"] > 1.0


def test_extract_behavioral_features_penalizes_outside_preferred_country_without_relocation(sample_candidate):
    contract = build_job_contract(
        """
        Role: Senior AI Engineer
        Location: Pune/Noida, India. Outside India case-by-case; no visa sponsorship.
        """
    )
    reference_date = compute_reference_date(["2026-05-20"])

    india_candidate = deepcopy(sample_candidate)
    india_candidate["profile"]["location"] = "Vizag, Andhra Pradesh"
    india_candidate["profile"]["country"] = "India"
    india_candidate["redrob_signals"]["last_active_date"] = "2026-05-10"
    india_candidate["redrob_signals"]["open_to_work_flag"] = True
    india_candidate["redrob_signals"]["recruiter_response_rate"] = 0.80
    india_candidate["redrob_signals"]["notice_period_days"] = 30

    outside_candidate = deepcopy(india_candidate)
    outside_candidate["profile"]["location"] = "Toronto"
    outside_candidate["profile"]["country"] = "Canada"
    outside_candidate["redrob_signals"]["willing_to_relocate"] = False
    outside_candidate["redrob_signals"]["preferred_work_mode"] = "onsite"

    india_features = extract_behavioral_features(build_normalized_candidate(india_candidate), reference_date, contract)
    outside_features = extract_behavioral_features(build_normalized_candidate(outside_candidate), reference_date, contract)

    assert india_features["preferred_country_score"] == 1.0
    assert outside_features["preferred_country_score"] == 0.0
    assert outside_features["availability_multiplier"] < india_features["availability_multiplier"]
