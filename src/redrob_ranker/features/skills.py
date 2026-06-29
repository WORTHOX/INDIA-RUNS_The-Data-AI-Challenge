"""Extract skill evidence and skill-stuffing indicators.

Skills support the JD match, but they are easier to exaggerate than career
history. This module scores direct skill coverage and also records suspicious
patterns such as many short-duration expert skills.
"""

from __future__ import annotations

from redrob_ranker.jd_contract import DEFAULT_JD_CONTRACT, JobContract
from redrob_ranker.types import NormalizedCandidate


def _match_skill_names(candidate: NormalizedCandidate, terms: tuple[str, ...]) -> list[str]:
    matched: list[str] = []
    for skill in candidate.skills:
        if any(term == skill.name or term in skill.name for term in terms):
            matched.append(skill.name)
    return sorted(set(matched))


def extract_skill_features(
    candidate: NormalizedCandidate,
    contract: JobContract = DEFAULT_JD_CONTRACT,
) -> dict[str, float | str | int]:
    python_matches = _match_skill_names(candidate, contract.python_keywords)
    vector_matches = _match_skill_names(candidate, contract.vector_keywords)
    retrieval_matches = _match_skill_names(candidate, contract.retrieval_keywords)
    evaluation_matches = _match_skill_names(candidate, contract.evaluation_keywords)
    llm_matches = _match_skill_names(candidate, contract.llm_keywords)
    nlp_matches = _match_skill_names(candidate, contract.nlp_keywords)
    job_skill_matches = _match_skill_names(candidate, contract.evidence_keywords or contract.job_keywords)

    expert_skill_count = sum(1 for skill in candidate.skills if skill.proficiency == "expert")
    short_duration_expert_count = sum(
        1 for skill in candidate.skills if skill.proficiency == "expert" and skill.duration_months < 12
    )

    python_score = 1.0 if python_matches else 0.0
    vector_score = min(1.0, len(vector_matches) / 2.0 + (0.25 if "embeddings" in retrieval_matches else 0.0))
    retrieval_skill_score = min(1.0, len(retrieval_matches) / 4.0)
    evaluation_skill_score = min(1.0, len(evaluation_matches) / 3.0)
    llm_skill_score = min(1.0, len(llm_matches) / 3.0)
    nlp_skill_score = min(1.0, len(nlp_matches) / 2.0)
    job_skill_score = min(1.0, len(job_skill_matches) / 4.0)

    evidence = sorted(set(python_matches + vector_matches + retrieval_matches + evaluation_matches + job_skill_matches))
    return {
        "python_score": round(python_score, 4),
        "vector_score": round(vector_score, 4),
        "retrieval_skill_score": round(retrieval_skill_score, 4),
        "evaluation_skill_score": round(evaluation_skill_score, 4),
        "llm_skill_score": round(llm_skill_score, 4),
        "nlp_skill_score": round(nlp_skill_score, 4),
        "job_skill_score": round(job_skill_score, 4),
        "expert_skill_count": expert_skill_count,
        "short_duration_expert_count": short_duration_expert_count,
        "ai_like_skill_count": len(set(retrieval_matches + llm_matches + nlp_matches + vector_matches)),
        "skill_evidence": ", ".join(evidence),
    }
