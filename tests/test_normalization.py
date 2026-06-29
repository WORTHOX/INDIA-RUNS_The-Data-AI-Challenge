from __future__ import annotations

from redrob_ranker.features.normalize import build_normalized_candidate


def test_build_normalized_candidate_extracts_core_fields(sample_candidate):
    normalized = build_normalized_candidate(sample_candidate)

    assert normalized.candidate_id == "CAND_0000001"
    assert normalized.current_title == "Backend Engineer"
    assert normalized.current_company == "Mindtree"
    assert normalized.current_industry == "IT Services"
    assert normalized.years_of_experience == 6.9
    assert normalized.last_active_date == "2026-05-20"
    assert normalized.total_roles == 2
    assert normalized.total_skills == 17
    assert "backend engineer" in normalized.title_text
    assert "spark" in normalized.profile_text
    assert "mindtree" in normalized.company_text
    assert normalized.skill_names[0] == "tailwind"


def test_build_normalized_candidate_orders_roles_by_recency(sample_candidate):
    normalized = build_normalized_candidate(sample_candidate)

    assert normalized.roles[0].company == "Mindtree"
    assert normalized.roles[0].is_current is True
    assert normalized.roles[1].company == "Dunder Mifflin"
