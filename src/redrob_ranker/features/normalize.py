"""Normalize raw candidate JSON into stable ranking input.

Raw candidates contain nested profile, career, skills, and Redrob signal data.
This module flattens that shape once so the rest of the pipeline can work with
consistent text fields and numeric values.
"""

from __future__ import annotations

from datetime import date

from redrob_ranker.types import NormalizedCandidate, NormalizedRole, NormalizedSkill


def _as_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _as_lower_text(value: object) -> str:
    return _as_text(value).lower()


def _sort_roles(raw_roles: list[dict]) -> list[dict]:
    def sort_key(role: dict) -> tuple[int, date, date]:
        end_date = _parse_date(role.get("end_date")) or date.min
        start_date = _parse_date(role.get("start_date")) or date.min
        return (1 if role.get("is_current") else 0, end_date, start_date)

    return sorted(raw_roles, key=sort_key, reverse=True)


def _parse_date(value: object) -> date | None:
    text = _as_text(value)
    if not text:
        return None
    return date.fromisoformat(text)


def build_normalized_candidate(candidate: dict) -> NormalizedCandidate:
    profile = candidate.get("profile", {})
    raw_roles = candidate.get("career_history", [])
    raw_skills = candidate.get("skills", [])
    signals = candidate.get("redrob_signals", {})

    sorted_roles = _sort_roles(raw_roles)
    roles = [
        NormalizedRole(
            company=_as_text(role.get("company")),
            title=_as_text(role.get("title")),
            start_date=_as_text(role.get("start_date")),
            end_date=role.get("end_date"),
            duration_months=int(role.get("duration_months", 0) or 0),
            is_current=bool(role.get("is_current")),
            industry=_as_text(role.get("industry")),
            company_size=_as_text(role.get("company_size")),
            description=_as_text(role.get("description")),
            title_text=_as_lower_text(role.get("title")),
            description_text=_as_lower_text(role.get("description")),
        )
        for role in sorted_roles
    ]
    skills = [
        NormalizedSkill(
            name=_as_lower_text(skill.get("name")),
            proficiency=_as_lower_text(skill.get("proficiency")),
            endorsements=int(skill.get("endorsements", 0) or 0),
            duration_months=int(skill.get("duration_months", 0) or 0),
        )
        for skill in raw_skills
    ]

    title_parts = [profile.get("current_title"), *(role.title for role in roles)]
    company_parts = [profile.get("current_company"), *(role.company for role in roles)]
    profile_parts = [
        profile.get("headline"),
        profile.get("summary"),
        profile.get("current_title"),
        profile.get("current_company"),
        profile.get("current_industry"),
        profile.get("location"),
        profile.get("country"),
    ]
    career_text = " ".join(part for part in [*(role.description for role in roles)] if part).lower()
    skills_text = " ".join(skill.name for skill in skills)
    salary_range = signals.get("expected_salary_range_inr_lpa", {})

    return NormalizedCandidate(
        candidate_id=_as_text(candidate.get("candidate_id")),
        headline=_as_text(profile.get("headline")),
        summary=_as_text(profile.get("summary")),
        location=_as_text(profile.get("location")),
        country=_as_text(profile.get("country")),
        years_of_experience=float(profile.get("years_of_experience", 0.0) or 0.0),
        current_title=_as_text(profile.get("current_title")),
        current_company=_as_text(profile.get("current_company")),
        current_company_size=_as_text(profile.get("current_company_size")),
        current_industry=_as_text(profile.get("current_industry")),
        title_text=" ".join(_as_lower_text(part) for part in title_parts if _as_text(part)),
        profile_text=" ".join(_as_lower_text(part) for part in profile_parts if _as_text(part)),
        company_text=" ".join(_as_lower_text(part) for part in company_parts if _as_text(part)),
        career_text=career_text,
        skills_text=skills_text,
        roles=roles,
        skills=skills,
        skill_names=[skill.name for skill in skills],
        total_roles=len(roles),
        total_skills=len(skills),
        profile_completeness_score=float(signals.get("profile_completeness_score", 0.0) or 0.0),
        signup_date=_as_text(signals.get("signup_date")),
        last_active_date=_as_text(signals.get("last_active_date")),
        open_to_work_flag=bool(signals.get("open_to_work_flag")),
        profile_views_received_30d=int(signals.get("profile_views_received_30d", 0) or 0),
        applications_submitted_30d=int(signals.get("applications_submitted_30d", 0) or 0),
        recruiter_response_rate=float(signals.get("recruiter_response_rate", 0.0) or 0.0),
        avg_response_time_hours=float(signals.get("avg_response_time_hours", 0.0) or 0.0),
        skill_assessment_scores=dict(signals.get("skill_assessment_scores", {}) or {}),
        connection_count=int(signals.get("connection_count", 0) or 0),
        endorsements_received=int(signals.get("endorsements_received", 0) or 0),
        notice_period_days=int(signals.get("notice_period_days", 0) or 0),
        expected_salary_min_lpa=float(salary_range.get("min", 0.0) or 0.0),
        expected_salary_max_lpa=float(salary_range.get("max", 0.0) or 0.0),
        preferred_work_mode=_as_text(signals.get("preferred_work_mode")),
        willing_to_relocate=bool(signals.get("willing_to_relocate")),
        github_activity_score=float(signals.get("github_activity_score", -1.0) or 0.0),
        search_appearance_30d=int(signals.get("search_appearance_30d", 0) or 0),
        saved_by_recruiters_30d=int(signals.get("saved_by_recruiters_30d", 0) or 0),
        interview_completion_rate=float(signals.get("interview_completion_rate", 0.0) or 0.0),
        offer_acceptance_rate=float(signals.get("offer_acceptance_rate", -1.0) or -1.0),
        verified_email=bool(signals.get("verified_email")),
        verified_phone=bool(signals.get("verified_phone")),
        linkedin_connected=bool(signals.get("linkedin_connected")),
    )
