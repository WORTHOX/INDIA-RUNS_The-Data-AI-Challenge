"""Suspiciousness penalties for noisy or unsupported profiles.

This layer subtracts score for profiles that look attractive through keywords
but weak under structural checks: unrelated titles, summary-only matches,
title-description mismatch, experience gaps, or skill stuffing.
"""

from __future__ import annotations

from collections.abc import Mapping

from redrob_ranker.jd_contract import DEFAULT_JD_CONTRACT, JobContract


def _value(candidate: Mapping[str, object] | object, key: str, default: float | int = 0.0) -> object:
    if isinstance(candidate, Mapping):
        return candidate.get(key, default)
    return getattr(candidate, key, default)


def compute_suspiciousness_penalty(
    candidate: Mapping[str, object] | object,
    *,
    title_features: dict,
    skill_features: dict,
    honeypot_features: dict,
    contract: JobContract = DEFAULT_JD_CONTRACT,
) -> float:
    """Return a capped penalty for unsupported or contradictory profiles."""

    penalty = 0.0

    # Wrong-title candidates with many AI-like skills are classic keyword
    # stuffers. Wrong-title candidates without stuffing still get a smaller hit.
    if (
        title_features["negative_title_score"] >= 0.6
        and title_features["title_score"] == 0.0
        and skill_features["ai_like_skill_count"] >= 4
    ):
        penalty += 0.35
    elif title_features["negative_title_score"] >= 0.6 and title_features["title_score"] == 0.0:
        penalty += 0.20

    if skill_features["expert_skill_count"] >= 10 and skill_features["short_duration_expert_count"] >= 10:
        penalty += 0.40

    if honeypot_features["claimed_experience_gap_years"] > 3.0:
        penalty += 0.30

    if honeypot_features["research_only_score"] >= 1.0:
        penalty += 0.25

    # Consensus features supply dataset-specific noise penalties.
    summary_only_match_score = float(_value(candidate, "summary_only_match_score", 0.0) or 0.0)
    if summary_only_match_score >= 0.40:
        penalty += 0.20 * summary_only_match_score

    title_description_mismatch_score = float(_value(candidate, "title_description_mismatch_score", 0.0) or 0.0)
    if title_description_mismatch_score:
        penalty += 0.20 * title_description_mismatch_score

    service_background_score = float(_value(candidate, "service_background_score", 0.0) or 0.0)
    total_roles = int(_value(candidate, "total_roles", 0) or 0)
    if total_roles and service_background_score >= 0.95:
        penalty += 0.20
    elif not total_roles and service_background_score >= 0.95:
        penalty += 0.20

    return round(min(0.50, penalty), 4)
