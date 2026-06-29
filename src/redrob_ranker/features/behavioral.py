"""Convert Redrob platform signals into availability features.

These features do not decide whether a candidate can do the job. They adjust
ranking for practical recruiting likelihood: recent activity, response rate,
notice period, relocation, and preferred locations.
"""

from __future__ import annotations

from datetime import date
from typing import Iterable

from redrob_ranker.jd_contract import DEFAULT_JD_CONTRACT, JobContract
from redrob_ranker.types import NormalizedCandidate


def compute_reference_date(last_active_dates: Iterable[str]) -> date:
    """Use the newest activity date in the dataset as the recency reference."""

    parsed_dates = [date.fromisoformat(value) for value in last_active_dates if value]
    if not parsed_dates:
        raise ValueError("At least one last_active_date is required to compute a reference date")
    return max(parsed_dates)


def extract_behavioral_features(
    candidate: NormalizedCandidate,
    reference_date: date,
    contract: JobContract = DEFAULT_JD_CONTRACT,
) -> dict[str, float | int]:
    last_active = date.fromisoformat(candidate.last_active_date)
    days_since_active = (reference_date - last_active).days
    location_text = candidate.location.lower()
    country_text = candidate.country.lower()
    preferred_location_score = 1.0 if any(city in location_text for city in contract.preferred_locations) else 0.0
    preferred_countries = getattr(contract, "preferred_countries", ())
    preferred_country_score = (
        1.0
        if preferred_countries and any(country in country_text for country in preferred_countries)
        else 0.0
    )

    # Start near neutral, then nudge up/down for recruiting likelihood. This is
    # a multiplier, not a replacement for JD fit.
    multiplier = 0.85
    if candidate.open_to_work_flag:
        multiplier += 0.05
    if days_since_active <= 30:
        multiplier += 0.05
    elif days_since_active > 180:
        multiplier -= 0.15
    if candidate.recruiter_response_rate > 0.50:
        multiplier += 0.05
    elif candidate.recruiter_response_rate < 0.10:
        multiplier -= 0.10
    if candidate.notice_period_days <= 30:
        multiplier += 0.05
    elif candidate.notice_period_days > 60:
        multiplier -= 0.05
    if candidate.interview_completion_rate > 0.80:
        multiplier += 0.03
    if candidate.github_activity_score > 50:
        multiplier += 0.03
    if preferred_location_score:
        multiplier += 0.03
    if preferred_countries:
        if preferred_country_score:
            multiplier += 0.01
        elif candidate.willing_to_relocate:
            multiplier += 0.01
        elif candidate.preferred_work_mode == "remote":
            multiplier -= 0.05
        else:
            multiplier -= 0.10
    elif candidate.willing_to_relocate:
        multiplier += 0.02
    elif candidate.preferred_work_mode == "remote":
        multiplier -= 0.03

    return {
        "days_since_active": days_since_active,
        "preferred_location_score": preferred_location_score,
        "preferred_country_score": preferred_country_score,
        "availability_multiplier": round(max(0.55, min(multiplier, 1.10)), 4),
    }
