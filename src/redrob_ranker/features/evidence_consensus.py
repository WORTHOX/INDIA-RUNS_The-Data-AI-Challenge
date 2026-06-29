"""Measure agreement across independent evidence channels.

The dataset contains noisy summaries and mismatched role descriptions. This
module rewards candidates whose title, career, skills, and requirement coverage
all point to the same JD fit, while flagging summary-only or mismatched evidence.
"""

from __future__ import annotations

from collections.abc import Mapping

from redrob_ranker.jd_contract import DEFAULT_JD_CONTRACT, JobContract
from redrob_ranker.types import NormalizedCandidate


def _unique_matches(text: str, terms: tuple[str, ...]) -> list[str]:
    return sorted({term for term in terms if term and term in text})


def _score_matches(matches: list[str], terms: tuple[str, ...]) -> float:
    if not terms:
        return 0.0
    return min(1.0, len(set(matches)) / len(set(terms)))


def _float_value(features: Mapping[str, object], key: str) -> float:
    return float(features.get(key, 0.0) or 0.0)


def _role_weights(total_roles: int) -> list[float]:
    if total_roles <= 0:
        return []
    if total_roles == 1:
        return [1.0]
    if total_roles == 2:
        return [0.65, 0.35]
    weights = [0.5, 0.3]
    weights.extend([0.2 / max(total_roles - 2, 1)] * (total_roles - 2))
    return weights


def _title_matches_positive(title_text: str, contract: JobContract) -> bool:
    return bool(_unique_matches(title_text, contract.positive_title_terms + contract.positive_title_hints))


def _title_matches_negative(title_text: str, contract: JobContract) -> bool:
    return bool(_unique_matches(title_text, contract.negative_title_terms + contract.negative_title_hints))


def extract_evidence_consensus_features(
    candidate: NormalizedCandidate,
    contract: JobContract = DEFAULT_JD_CONTRACT,
    *,
    title_features: Mapping[str, object],
    career_features: Mapping[str, object],
    skill_features: Mapping[str, object],
) -> dict[str, float | int | str]:
    """Return cross-channel agreement and noisy-match risk features."""

    fallback_terms = contract.evidence_keywords or contract.job_keywords
    evidence_terms = contract.must_have_keywords or fallback_terms[:12]
    nice_terms = contract.nice_to_have_keywords
    role_specific_terms = tuple(dict.fromkeys([*evidence_terms, *nice_terms, *fallback_terms]))

    structured_text = " ".join([candidate.title_text, candidate.career_text, candidate.skills_text])
    summary_text = " ".join([candidate.headline.lower(), candidate.summary.lower()])

    must_matches = _unique_matches(structured_text, evidence_terms)
    nice_matches = _unique_matches(structured_text, nice_terms)
    summary_matches = _unique_matches(summary_text, role_specific_terms)

    must_have_coverage_score = _score_matches(must_matches, evidence_terms)
    nice_to_have_coverage_score = _score_matches(nice_matches, nice_terms)
    if evidence_terms and nice_terms:
        requirement_coverage_score = 0.75 * must_have_coverage_score + 0.25 * nice_to_have_coverage_score
    elif evidence_terms:
        requirement_coverage_score = must_have_coverage_score
    else:
        requirement_coverage_score = max(
            _float_value(career_features, "job_keyword_score"),
            _float_value(skill_features, "job_skill_score"),
        )

    # A candidate is more trustworthy when multiple independent channels agree.
    # Four possible channels: title, career, skills, and requirement coverage.
    title_channel = _float_value(title_features, "title_score") >= 0.35
    career_channel = max(
        _float_value(career_features, "job_keyword_score"),
        _float_value(career_features, "retrieval_score"),
        _float_value(career_features, "evaluation_score"),
    ) >= 0.25
    skill_channel = max(
        _float_value(skill_features, "job_skill_score"),
        _float_value(skill_features, "retrieval_skill_score"),
        _float_value(skill_features, "python_score"),
    ) >= 0.25
    requirement_channel = requirement_coverage_score >= 0.50
    evidence_channel_count = sum([title_channel, career_channel, skill_channel, requirement_channel])
    evidence_consensus_score = evidence_channel_count / 4.0

    summary_keyword_score = _score_matches(summary_matches, role_specific_terms)
    # Summary-only matches are risky because many noisy profiles mention trendy
    # terms without matching title/career/skill evidence.
    summary_only_match_score = 0.0
    if summary_keyword_score >= 0.40 and evidence_channel_count <= 1:
        summary_only_match_score = summary_keyword_score * (1.0 - evidence_consensus_score)

    # If an unrelated title has a JD-heavy description, treat it as a possible
    # template/noise mismatch instead of blindly rewarding the description.
    title_description_mismatch_score = 0.0
    for role, weight in zip(candidate.roles, _role_weights(candidate.total_roles), strict=False):
        role_has_wrong_title = _title_matches_negative(role.title_text, contract) and not _title_matches_positive(
            role.title_text, contract
        )
        description_has_jd_evidence = bool(_unique_matches(role.description_text, role_specific_terms))
        if role_has_wrong_title and description_has_jd_evidence and _float_value(title_features, "title_score") == 0.0:
            title_description_mismatch_score += weight

    channel_names = []
    if title_channel:
        channel_names.append("title")
    if career_channel:
        channel_names.append("career")
    if skill_channel:
        channel_names.append("skills")
    if requirement_channel:
        channel_names.append("requirements")

    consensus_evidence_terms = sorted(set([*must_matches, *nice_matches]))[:6]
    consensus_evidence = ", ".join(consensus_evidence_terms)

    return {
        "must_have_coverage_score": round(must_have_coverage_score, 4),
        "nice_to_have_coverage_score": round(nice_to_have_coverage_score, 4),
        "requirement_coverage_score": round(min(requirement_coverage_score, 1.0), 4),
        "evidence_channel_count": evidence_channel_count,
        "evidence_consensus_score": round(evidence_consensus_score, 4),
        "summary_keyword_score": round(summary_keyword_score, 4),
        "summary_only_match_score": round(min(summary_only_match_score, 1.0), 4),
        "title_description_mismatch_score": round(min(title_description_mismatch_score, 1.0), 4),
        "evidence_channels": ", ".join(channel_names),
        "consensus_evidence": consensus_evidence,
    }
