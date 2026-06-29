"""Normalized data containers used after raw JSON parsing.

The raw candidate file is nested and inconsistent for ranking use. Normalization
converts it into these small dataclasses so every feature module works with the
same field names and text formats.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class NormalizedRole:
    company: str
    title: str
    start_date: str
    end_date: str | None
    duration_months: int
    is_current: bool
    industry: str
    company_size: str
    description: str
    title_text: str
    description_text: str


@dataclass(slots=True)
class NormalizedSkill:
    name: str
    proficiency: str
    endorsements: int
    duration_months: int


@dataclass(slots=True)
class NormalizedCandidate:
    candidate_id: str
    headline: str
    summary: str
    location: str
    country: str
    years_of_experience: float
    current_title: str
    current_company: str
    current_company_size: str
    current_industry: str
    title_text: str
    profile_text: str
    company_text: str
    career_text: str
    skills_text: str
    roles: list[NormalizedRole] = field(default_factory=list)
    skills: list[NormalizedSkill] = field(default_factory=list)
    skill_names: list[str] = field(default_factory=list)
    total_roles: int = 0
    total_skills: int = 0
    profile_completeness_score: float = 0.0
    signup_date: str = ""
    last_active_date: str = ""
    open_to_work_flag: bool = False
    profile_views_received_30d: int = 0
    applications_submitted_30d: int = 0
    recruiter_response_rate: float = 0.0
    avg_response_time_hours: float = 0.0
    skill_assessment_scores: dict[str, float] = field(default_factory=dict)
    connection_count: int = 0
    endorsements_received: int = 0
    notice_period_days: int = 0
    expected_salary_min_lpa: float = 0.0
    expected_salary_max_lpa: float = 0.0
    preferred_work_mode: str = ""
    willing_to_relocate: bool = False
    github_activity_score: float = -1.0
    search_appearance_30d: int = 0
    saved_by_recruiters_30d: int = 0
    interview_completion_rate: float = 0.0
    offer_acceptance_rate: float = -1.0
    verified_email: bool = False
    verified_phone: bool = False
    linkedin_connected: bool = False
