"""Score seniority fit without reducing maturity to years alone.

Recruiters rarely think "years == seniority" in a straight line. A candidate
with 3-4 years can be senior-like if recent work shows ownership, production
shipping, architecture, evaluation, mentoring, or end-to-end delivery. The
opposite is also true: a very senior manager may be risky for a lower-level
hands-on IC role even if keywords match.
"""

from __future__ import annotations

from collections.abc import Mapping

from redrob_ranker.jd_contract import DEFAULT_JD_CONTRACT, JobContract
from redrob_ranker.types import NormalizedCandidate

OWNERSHIP_TERMS = (
    "owned",
    "ownership",
    "led",
    "lead",
    "architected",
    "designed",
    "built",
    "launched",
    "deployed",
    "production",
    "scaled",
    "optimized",
    "improved",
    "implemented",
    "end-to-end",
    "end to end",
    "from scratch",
    "mentored",
    "reviewed",
    "roadmap",
)

MANAGEMENT_TITLE_TERMS = (
    "manager",
    "director",
    "head",
    "vp",
    "vice president",
    "cto",
    "chief",
)

HANDS_ON_TERMS = (
    "built",
    "implemented",
    "coded",
    "developed",
    "deployed",
    "production",
    "debugged",
    "architected",
    "designed",
)


def _value(features: Mapping[str, object], key: str, default: float = 0.0) -> float:
    return float(features.get(key, default) or default)


def _title_level(title: str) -> float:
    """Map title language to a rough career level scale."""

    padded = f" {title.lower()} "
    if any(term in padded for term in (" chief ", " cto ", " vp ", " vice president ")):
        return 6.0
    if any(term in padded for term in (" director ", " head ")):
        return 5.0
    if " principal " in padded:
        return 4.5
    if " staff " in padded:
        return 4.2
    if " manager " in padded:
        return 4.0
    if " lead " in padded:
        return 3.8
    if any(term in padded for term in (" senior ", " sr ")):
        return 3.2
    if any(term in padded for term in (" junior ", " jr ", " associate ", " trainee ", " intern ")):
        return 1.2
    if any(term in padded for term in (" engineer ", " scientist ", " developer ", " analyst ", " specialist ")):
        return 2.4
    return 2.0


def _has_management_title(title: str) -> bool:
    padded = f" {title.lower()} "
    return any(term in padded for term in MANAGEMENT_TITLE_TERMS)


def _weighted_ownership_score(candidate: NormalizedCandidate) -> float:
    """Measure senior-like ownership in recent role descriptions."""

    if not candidate.roles:
        return 0.0
    if candidate.total_roles == 1:
        weights = [1.0]
    elif candidate.total_roles == 2:
        weights = [0.65, 0.35]
    else:
        weights = [0.5, 0.3]
        weights.extend([0.2 / max(candidate.total_roles - 2, 1)] * (candidate.total_roles - 2))

    score = 0.0
    for role, weight in zip(candidate.roles, weights, strict=False):
        role_text = f"{role.title_text} {role.description_text}"
        matches = {term for term in OWNERSHIP_TERMS if term in role_text}
        score += min(1.0, len(matches) / 8.0) * weight
    return round(min(score, 1.0), 4)


def _peak_title_level(candidate: NormalizedCandidate) -> float:
    levels = [_title_level(candidate.current_title)]
    levels.extend(_title_level(role.title) for role in candidate.roles)
    return max(levels) if levels else 0.0


def _experience_alignment_score(years: float, min_years: float, max_years: float, work_maturity_score: float) -> float:
    """Score tenure against the JD band while allowing early maturity evidence."""

    if min_years <= years <= max_years:
        return 1.0
    if years < min_years:
        shortfall = min_years - years
        base = max(0.0, 1.0 - shortfall / max(min_years, 1.0))
        if work_maturity_score >= 0.70:
            base = max(base, 0.78)
        elif work_maturity_score >= 0.55:
            base = max(base, 0.62)
        return round(base, 4)

    overage = years - max_years
    return round(max(0.35, 1.0 - overage / max(max_years, 1.0)), 4)


def extract_seniority_alignment_features(
    candidate: NormalizedCandidate,
    contract: JobContract = DEFAULT_JD_CONTRACT,
    *,
    career_features: Mapping[str, object],
    skill_features: Mapping[str, object],
    consensus_features: Mapping[str, object],
) -> dict[str, float | str]:
    """Return hierarchy, maturity, and title-level risk features."""

    target_level = float(contract.target_seniority_level)
    current_level = _title_level(candidate.current_title)
    peak_level = _peak_title_level(candidate)
    ownership_score = _weighted_ownership_score(candidate)
    evidence_channel_count = int(consensus_features.get("evidence_channel_count", 0) or 0)
    consensus_score = _value(consensus_features, "evidence_consensus_score")
    career_score = _value(career_features, "career_score")
    shipping_score = _value(career_features, "shipping_score")
    requirement_score = _value(consensus_features, "requirement_coverage_score")
    skill_score = _value(skill_features, "job_skill_score")

    work_maturity_score = round(
        min(
            1.0,
            0.42 * ownership_score
            + 0.18 * shipping_score
            + 0.16 * career_score
            + 0.14 * consensus_score
            + 0.10 * max(requirement_score, skill_score),
        ),
        4,
    )
    experience_alignment_score = _experience_alignment_score(
        candidate.years_of_experience,
        contract.min_years,
        contract.max_years,
        work_maturity_score,
    )

    early_maturity_signal = 0.0
    if candidate.years_of_experience < contract.min_years and evidence_channel_count >= 3 and work_maturity_score >= 0.62:
        early_maturity_signal = min(
            1.0,
            0.62 * work_maturity_score + 0.23 * consensus_score + 0.15 * career_score,
        )

    level_shortfall = max(0.0, target_level - current_level)
    tenure_shortfall = max(0.0, contract.min_years - candidate.years_of_experience) / max(contract.min_years, 1.0)
    underqualified_seniority_risk = min(1.0, 0.55 * min(1.0, level_shortfall / 1.5) + 0.45 * tenure_shortfall)
    if early_maturity_signal:
        underqualified_seniority_risk *= max(0.25, 1.0 - early_maturity_signal)

    excess_level = max(0.0, peak_level - target_level)
    overqualified_downgrade_risk = 0.0
    if excess_level > 1.0:
        overqualified_downgrade_risk = min(1.0, (excess_level - 1.0) / 1.6)
        if candidate.open_to_work_flag:
            overqualified_downgrade_risk *= 0.90
        if contract.target_role_mode in {"hybrid", "management"}:
            overqualified_downgrade_risk *= 0.50

    is_management_candidate = _has_management_title(candidate.current_title)
    has_hands_on_evidence = any(term in candidate.career_text for term in HANDS_ON_TERMS)
    management_mismatch_score = 0.0
    if contract.target_role_mode == "ic" and is_management_candidate:
        management_mismatch_score = 0.65
        if has_hands_on_evidence or work_maturity_score >= 0.65:
            management_mismatch_score = 0.30
        if career_score < 0.25:
            management_mismatch_score = max(management_mismatch_score, 0.55)

    level_gap = abs(current_level - target_level)
    seniority_alignment_score = max(0.0, 1.0 - min(1.0, level_gap / 2.6))
    if early_maturity_signal:
        seniority_alignment_score = max(seniority_alignment_score, 0.72 + 0.20 * early_maturity_signal)
    seniority_alignment_score -= 0.18 * overqualified_downgrade_risk
    seniority_alignment_score -= 0.18 * underqualified_seniority_risk
    seniority_alignment_score -= 0.18 * management_mismatch_score

    return {
        "jd_min_years": round(contract.min_years, 2),
        "jd_max_years": round(contract.max_years, 2),
        "target_seniority_level": round(target_level, 2),
        "target_seniority_label": contract.target_seniority_label,
        "target_role_mode": contract.target_role_mode,
        "current_seniority_level": round(current_level, 2),
        "peak_seniority_level": round(peak_level, 2),
        "ownership_signal_score": round(ownership_score, 4),
        "work_maturity_score": round(work_maturity_score, 4),
        "experience_alignment_score": round(experience_alignment_score, 4),
        "early_maturity_signal": round(early_maturity_signal, 4),
        "underqualified_seniority_risk": round(min(underqualified_seniority_risk, 1.0), 4),
        "overqualified_downgrade_risk": round(min(overqualified_downgrade_risk, 1.0), 4),
        "management_mismatch_score": round(min(management_mismatch_score, 1.0), 4),
        "seniority_alignment_score": round(min(max(seniority_alignment_score, 0.0), 1.0), 4),
    }
