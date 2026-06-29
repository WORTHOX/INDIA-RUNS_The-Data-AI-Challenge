"""Extract career-history evidence from role descriptions.

Career evidence answers: has this person actually done work related to the JD,
especially in recent roles, and does that work look like production/product
experience rather than only generic keywords?
"""

from __future__ import annotations

from redrob_ranker.jd_contract import DEFAULT_JD_CONTRACT, JobContract
from redrob_ranker.types import NormalizedCandidate


def _unique_matches(text: str, terms: tuple[str, ...]) -> list[str]:
    return sorted({term for term in terms if term in text})


def _role_weights(total_roles: int) -> list[float]:
    """Weight recent/current roles more heavily than older history."""

    if total_roles <= 0:
        return []
    if total_roles == 1:
        return [1.0]
    if total_roles == 2:
        return [0.65, 0.35]
    weights = [0.5, 0.3]
    remainder = 0.2 / max(total_roles - 2, 1)
    weights.extend([remainder] * (total_roles - 2))
    return weights


def extract_career_features(
    candidate: NormalizedCandidate,
    contract: JobContract = DEFAULT_JD_CONTRACT,
) -> dict[str, float | str]:
    retrieval_score = 0.0
    evaluation_score = 0.0
    shipping_score = 0.0
    product_background_score = 0.0
    service_background_score = 0.0
    job_keyword_score = 0.0
    matched_terms: set[str] = set()

    weights = _role_weights(candidate.total_roles)
    for role, weight in zip(candidate.roles, weights, strict=False):
        role_text = f"{role.title_text} {role.description_text}"
        role_retrieval = _unique_matches(role_text, contract.retrieval_keywords)
        role_eval = _unique_matches(role_text, contract.evaluation_keywords)
        role_shipping = _unique_matches(role_text, contract.shipping_keywords)
        role_job_keywords = _unique_matches(role_text, contract.evidence_keywords or contract.job_keywords)

        retrieval_score += min(1.0, len(role_retrieval) / 4.0) * weight
        evaluation_score += min(1.0, len(role_eval) / 2.0) * weight
        shipping_score += min(1.0, len(role_shipping) / 2.0) * weight
        job_keyword_score += min(1.0, len(role_job_keywords) / 5.0) * weight

        matched_terms.update(role_retrieval)
        matched_terms.update(role_eval)
        matched_terms.update(role_job_keywords[:5])

        company = role.company.lower()
        industry = role.industry.lower()
        is_service_industry = any(name in industry for name in contract.service_industries)
        is_service_company = any(name in company for name in contract.service_companies)

        if is_service_industry:
            service_background_score += weight
        elif is_service_company and not industry:
            service_background_score += weight
        else:
            product_background_score += weight

    # For AI/search JDs, retrieval_score carries the signal. For other JDs,
    # job_keyword_score lets the same career formula follow the supplied role.
    role_specific_score = max(retrieval_score, job_keyword_score)
    career_score = min(
        1.0,
        0.45 * role_specific_score + 0.25 * shipping_score + 0.15 * evaluation_score + 0.15 * product_background_score,
    )

    return {
        "career_score": round(career_score, 4),
        "retrieval_score": round(min(retrieval_score, 1.0), 4),
        "evaluation_score": round(min(evaluation_score, 1.0), 4),
        "shipping_score": round(min(shipping_score, 1.0), 4),
        "job_keyword_score": round(min(job_keyword_score, 1.0), 4),
        "product_background_score": round(min(product_background_score, 1.0), 4),
        "service_background_score": round(min(service_background_score, 1.0), 4),
        "career_evidence": ", ".join(sorted(matched_terms)),
    }
