"""Core fit-score formula.

The fit score is the main measure of whether a candidate fits the JD before
availability and suspiciousness adjustments are applied.
"""

from __future__ import annotations

from typing import Mapping


def compute_experience_band_score(
    years_of_experience: float,
    *,
    min_years: float = 5.0,
    max_years: float = 9.0,
    work_maturity_score: float = 0.0,
    evidence_channel_count: int = 0,
) -> float:
    """Score experience against the JD band without ignoring early maturity."""

    if min_years <= years_of_experience <= max_years:
        return 1.0
    if years_of_experience < min_years:
        shortfall = min_years - years_of_experience
        base = max(0.0, 1.0 - shortfall / max(min_years, 1.0))
        if work_maturity_score >= 0.70 and evidence_channel_count >= 3:
            base = max(base, 0.80)
        elif work_maturity_score >= 0.55 and evidence_channel_count >= 2:
            base = max(base, 0.65)
        return round(base, 6)
    overage = years_of_experience - max_years
    return round(max(0.35, 1.0 - overage / max(max_years, 1.0)), 6)


def _value(row: Mapping[str, float], key: str, default: float = 0.0) -> float:
    return float(row.get(key, default) or default)


def compute_role_specific_signal(row: Mapping[str, float]) -> float:
    """Reward broad JD coverage instead of one saturated keyword channel."""

    keyword_signal = max(_value(row, "job_keyword_score"), _value(row, "job_skill_score"))
    requirement_signal = _value(row, "requirement_coverage_score")
    search_system_signal = (
        _value(row, "retrieval_score")
        + _value(row, "retrieval_skill_score")
        + _value(row, "vector_score")
    ) / 3.0
    strongest_search_signal = max(
        _value(row, "retrieval_score"),
        _value(row, "retrieval_skill_score"),
        _value(row, "vector_score"),
    )
    return round(
        min(
            1.0,
            0.35 * keyword_signal
            + 0.25 * requirement_signal
            + 0.25 * search_system_signal
            + 0.15 * strongest_search_signal,
        ),
        6,
    )


def compute_fit_score(row: Mapping[str, float]) -> float:
    """Blend structural and role-specific signals into a 0-1 fit score."""

    # This blended signal adapts to different JDs while avoiding the old max()
    # saturation problem where one perfect channel hid weak coverage elsewhere.
    role_specific_signal = compute_role_specific_signal(row)
    evaluation_signal = max(
        _value(row, "evaluation_score"),
        _value(row, "evaluation_skill_score"),
    )
    consensus_signal = _value(row, "evidence_consensus_score")
    work_maturity_signal = _value(row, "work_maturity_score")
    seniority_signal = _value(row, "seniority_alignment_score")
    experience_score = max(
        _value(row, "experience_alignment_score"),
        compute_experience_band_score(
            _value(row, "years_of_experience"),
            min_years=_value(row, "jd_min_years", 5.0),
            max_years=_value(row, "jd_max_years", 9.0),
            work_maturity_score=work_maturity_signal,
            evidence_channel_count=int(_value(row, "evidence_channel_count")),
        ),
    )
    fit_score = (
        0.20 * _value(row, "title_score")
        + 0.20 * _value(row, "career_score")
        + 0.17 * role_specific_signal
        + 0.12 * consensus_signal
        + 0.08 * seniority_signal
        + 0.07 * work_maturity_signal
        + 0.06 * evaluation_signal
        + 0.04 * _value(row, "python_score")
        + 0.03 * _value(row, "product_background_score")
        + 0.03 * experience_score
    )
    return round(min(fit_score, 1.0), 6)
