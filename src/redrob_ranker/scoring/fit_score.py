"""Core fit-score formula.

The fit score is the main measure of whether a candidate fits the JD before
availability and suspiciousness adjustments are applied.
"""

from __future__ import annotations

from typing import Mapping


def compute_experience_band_score(years_of_experience: float) -> float:
    """Score experience against the expected senior-but-hands-on band."""

    if 5.0 <= years_of_experience <= 9.0:
        return 1.0
    if 4.0 <= years_of_experience <= 10.0:
        return 0.8
    if 3.0 <= years_of_experience <= 12.0:
        return 0.6
    if years_of_experience > 0:
        return 0.3
    return 0.0


def _value(row: Mapping[str, float], key: str) -> float:
    return float(row.get(key, 0.0) or 0.0)


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
    experience_score = compute_experience_band_score(_value(row, "years_of_experience"))
    fit_score = (
        0.22 * _value(row, "title_score")
        + 0.22 * _value(row, "career_score")
        + 0.18 * role_specific_signal
        + 0.12 * consensus_signal
        + 0.08 * evaluation_signal
        + 0.06 * _value(row, "python_score")
        + 0.06 * _value(row, "product_background_score")
        + 0.06 * experience_score
    )
    return round(min(fit_score, 1.0), 6)
