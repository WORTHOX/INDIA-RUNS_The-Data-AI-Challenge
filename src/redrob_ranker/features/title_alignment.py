"""Score title alignment against the target JD role.

Titles are the strongest structural signal in this noisy dataset. This module
rewards current/recent titles that match the JD and flags unrelated title
families only when there is no positive title evidence.
"""

from __future__ import annotations

from redrob_ranker.jd_contract import DEFAULT_JD_CONTRACT, JobContract
from redrob_ranker.types import NormalizedCandidate

ADJACENT_TECH_TITLE_WORDS = (
    "engineer",
    "scientist",
    "developer",
    "architect",
    "specialist",
    "researcher",
)


def _match_terms(text: str, terms: tuple[str, ...]) -> list[str]:
    return [term for term in terms if term in text]


def _has_adjacent_technical_title_family(title: str) -> bool:
    return any(word in title for word in ADJACENT_TECH_TITLE_WORDS)


def _weighted_role_titles(candidate: NormalizedCandidate) -> list[tuple[float, str]]:
    """Return unique titles weighted by recency/current-role importance."""

    titles = [candidate.current_title.lower(), *(role.title.lower() for role in candidate.roles)]
    unique_titles: list[str] = []
    for title in titles:
        if title and title not in unique_titles:
            unique_titles.append(title)

    if not unique_titles:
        return []
    if len(unique_titles) == 1:
        return [(1.0, unique_titles[0])]
    if len(unique_titles) == 2:
        return [(0.6, unique_titles[0]), (0.4, unique_titles[1])]
    weights = [0.5, 0.3]
    remainder = 0.2 / max(len(unique_titles) - 2, 1)
    weights.extend([remainder] * (len(unique_titles) - 2))
    return list(zip(weights, unique_titles, strict=True))


def extract_title_features(
    candidate: NormalizedCandidate,
    contract: JobContract = DEFAULT_JD_CONTRACT,
) -> dict[str, float | str]:
    title_score = 0.0
    negative_title_score = 0.0
    positive_evidence: list[str] = []
    negative_evidence: list[str] = []

    for weight, title in _weighted_role_titles(candidate):
        matched_positive = _match_terms(title, contract.positive_title_terms)
        matched_positive_hints = _match_terms(title, contract.positive_title_hints)
        matched_negative = _match_terms(title, contract.negative_title_terms)
        matched_negative_hints = _match_terms(title, contract.negative_title_hints)

        if matched_positive:
            title_score += weight
            positive_evidence.append(title)
        elif matched_positive_hints:
            hint_multiplier = 0.8 if _has_adjacent_technical_title_family(title) else 0.65
            title_score += weight * hint_multiplier
            positive_evidence.append(title)

        if matched_negative:
            negative_title_score += weight
            negative_evidence.append(title)
        elif matched_negative_hints:
            negative_title_score += weight * 0.7
            negative_evidence.append(title)

    # A positive title match wins over older unrelated titles. This avoids
    # punishing career switchers whose current title clearly matches the JD.
    if title_score > 0.0:
        negative_title_score = 0.0
        negative_evidence = []

    return {
        "title_score": round(min(title_score, 1.0), 4),
        "negative_title_score": round(min(negative_title_score, 1.0), 4),
        "title_evidence": ", ".join(positive_evidence),
        "negative_title_evidence": ", ".join(negative_evidence),
    }
