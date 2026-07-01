from __future__ import annotations

from redrob_ranker.scoring.fit_score import compute_fit_score


def test_compute_fit_score_rewards_evidence_consensus_and_requirement_coverage():
    base_row = {
        "title_score": 0.7,
        "career_score": 0.6,
        "retrieval_score": 0.2,
        "retrieval_skill_score": 0.2,
        "vector_score": 0.0,
        "job_keyword_score": 0.2,
        "job_skill_score": 0.2,
        "evaluation_score": 0.0,
        "evaluation_skill_score": 0.0,
        "python_score": 0.0,
        "product_background_score": 0.5,
        "years_of_experience": 6.0,
        "requirement_coverage_score": 0.0,
        "evidence_consensus_score": 0.0,
    }
    consensus_row = {
        **base_row,
        "requirement_coverage_score": 1.0,
        "evidence_consensus_score": 1.0,
    }

    assert compute_fit_score(consensus_row) > compute_fit_score(base_row)


def test_compute_fit_score_rewards_broad_role_specific_coverage_over_single_saturated_signal():
    single_signal_row = {
        "title_score": 0.8,
        "career_score": 0.5,
        "retrieval_score": 0.0,
        "retrieval_skill_score": 0.0,
        "vector_score": 1.0,
        "job_keyword_score": 0.0,
        "job_skill_score": 0.0,
        "evaluation_score": 0.0,
        "evaluation_skill_score": 0.0,
        "python_score": 0.0,
        "product_background_score": 0.5,
        "years_of_experience": 6.0,
        "requirement_coverage_score": 0.0,
        "evidence_consensus_score": 0.5,
    }
    broad_coverage_row = {
        **single_signal_row,
        "retrieval_score": 0.6,
        "retrieval_skill_score": 0.6,
        "vector_score": 0.6,
        "job_keyword_score": 0.6,
        "job_skill_score": 0.6,
        "requirement_coverage_score": 0.6,
        "python_score": 0.6,
    }

    assert compute_fit_score(broad_coverage_row) > compute_fit_score(single_signal_row)


def test_compute_fit_score_rewards_work_maturity_when_years_are_light():
    base_row = {
        "title_score": 0.7,
        "career_score": 0.7,
        "retrieval_score": 0.6,
        "retrieval_skill_score": 0.6,
        "vector_score": 0.4,
        "job_keyword_score": 0.6,
        "job_skill_score": 0.6,
        "evaluation_score": 0.5,
        "evaluation_skill_score": 0.5,
        "python_score": 1.0,
        "product_background_score": 0.7,
        "years_of_experience": 3.8,
        "jd_min_years": 5.0,
        "jd_max_years": 8.0,
        "requirement_coverage_score": 0.8,
        "evidence_consensus_score": 1.0,
        "evidence_channel_count": 4,
    }
    early_mature_row = {
        **base_row,
        "work_maturity_score": 0.9,
        "seniority_alignment_score": 0.85,
        "underqualified_seniority_risk": 0.1,
    }
    years_only_row = {
        **base_row,
        "work_maturity_score": 0.2,
        "seniority_alignment_score": 0.45,
        "underqualified_seniority_risk": 0.6,
    }

    assert compute_fit_score(early_mature_row) > compute_fit_score(years_only_row)
