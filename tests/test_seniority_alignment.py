from __future__ import annotations

from redrob_ranker.features.career_history import extract_career_features
from redrob_ranker.features.evidence_consensus import extract_evidence_consensus_features
from redrob_ranker.features.normalize import build_normalized_candidate
from redrob_ranker.features.seniority_alignment import extract_seniority_alignment_features
from redrob_ranker.features.skills import extract_skill_features
from redrob_ranker.features.title_alignment import extract_title_features
from redrob_ranker.job_description import build_job_contract


def _candidate(
    *,
    candidate_id: str,
    title: str,
    years: float,
    description: str,
    skills: list[str] | None = None,
) -> dict:
    return {
        "candidate_id": candidate_id,
        "profile": {
            "headline": title,
            "summary": "",
            "location": "Bengaluru",
            "country": "India",
            "years_of_experience": years,
            "current_title": title,
            "current_company": "ProductCo",
            "current_company_size": "51-200",
            "current_industry": "Software",
        },
        "career_history": [
            {
                "company": "ProductCo",
                "title": title,
                "start_date": "2022-01-01",
                "end_date": None,
                "duration_months": int(years * 12),
                "is_current": True,
                "industry": "Software",
                "company_size": "51-200",
                "description": description,
            }
        ],
        "skills": [
            {"name": skill, "proficiency": "expert", "endorsements": 12, "duration_months": 30}
            for skill in (skills or ["Python", "Retrieval", "Ranking", "Evaluation"])
        ],
        "redrob_signals": {
            "open_to_work_flag": True,
            "recruiter_response_rate": 0.8,
            "notice_period_days": 30,
        },
    }


def _seniority_features(candidate: dict, jd_text: str) -> dict:
    contract = build_job_contract(jd_text)
    normalized = build_normalized_candidate(candidate)
    title_features = extract_title_features(normalized, contract)
    career_features = extract_career_features(normalized, contract)
    skill_features = extract_skill_features(normalized, contract)
    consensus_features = extract_evidence_consensus_features(
        normalized,
        contract,
        title_features=title_features,
        career_features=career_features,
        skill_features=skill_features,
    )
    return extract_seniority_alignment_features(
        normalized,
        contract,
        career_features=career_features,
        skill_features=skill_features,
        consensus_features=consensus_features,
    )


def test_early_high_ownership_work_offsets_lighter_tenure_for_senior_role():
    jd_text = (
        "Job Description: Senior AI Engineer\n"
        "Requirements: Python, retrieval, ranking, evaluation.\n"
        "Need 5-8 years building production search systems."
    )
    candidate = _candidate(
        candidate_id="CAND_EARLY_MATURE",
        title="Machine Learning Engineer",
        years=3.8,
        description=(
            "Owned end-to-end retrieval and ranking systems, architected production search, "
            "launched evaluation dashboards, optimized latency, and mentored junior engineers."
        ),
    )

    features = _seniority_features(candidate, jd_text)

    assert features["early_maturity_signal"] >= 0.75
    assert features["underqualified_seniority_risk"] <= 0.25
    assert features["seniority_alignment_score"] >= 0.75


def test_director_for_hands_on_senior_ic_role_gets_downgrade_and_management_risk():
    jd_text = (
        "Job Description: Senior AI Engineer\n"
        "Requirements: Python, retrieval, ranking, evaluation.\n"
        "Hands-on role building production AI systems."
    )
    candidate = _candidate(
        candidate_id="CAND_DIRECTOR",
        title="Director of Engineering",
        years=14.0,
        description="Managed managers, owned quarterly planning, hiring, performance reviews, and roadmap governance.",
        skills=["Python", "Machine Learning"],
    )

    features = _seniority_features(candidate, jd_text)

    assert features["overqualified_downgrade_risk"] >= 0.45
    assert features["management_mismatch_score"] >= 0.45
    assert features["seniority_alignment_score"] <= 0.60
