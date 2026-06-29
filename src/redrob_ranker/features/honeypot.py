"""Detect internal consistency problems and honeypot-style signals.

The rules here only use fields present in the candidate dataset. They avoid
external company facts and focus on contradictions such as impossible experience
claims or research-only profiles with no shipping evidence.
"""

from __future__ import annotations

from redrob_ranker.jd_contract import DEFAULT_JD_CONTRACT, JobContract
from redrob_ranker.types import NormalizedCandidate


def extract_honeypot_features(
    candidate: NormalizedCandidate,
    contract: JobContract = DEFAULT_JD_CONTRACT,
) -> dict[str, float | int]:
    total_role_months = sum(role.duration_months for role in candidate.roles)
    claimed_months = int(round(candidate.years_of_experience * 12))
    claimed_gap_years = max(0.0, (claimed_months - total_role_months) / 12.0)

    expert_skill_count = sum(1 for skill in candidate.skills if skill.proficiency == "expert")
    short_duration_expert_count = sum(
        1 for skill in candidate.skills if skill.proficiency == "expert" and skill.duration_months < 12
    )
    shipping_hits = sum(
        1 for role in candidate.roles for keyword in contract.shipping_keywords if keyword in role.description_text
    )
    research_hits = sum(
        1 for role in candidate.roles for keyword in contract.research_keywords if keyword in role.description_text
    )

    return {
        "total_role_months": total_role_months,
        "claimed_experience_gap_years": round(claimed_gap_years, 4),
        "expert_skill_count": expert_skill_count,
        "short_duration_expert_count": short_duration_expert_count,
        "research_only_score": 1.0 if research_hits and not shipping_hits else 0.0,
    }
