"""Build deterministic evidence-only reasoning strings.

The final CSV needs human-readable explanations, but generated claims are risky.
This module formats feature evidence already extracted from the candidate data;
it does not infer new facts or call an LLM.
"""

from __future__ import annotations

from typing import Mapping

TERM_PRIORITY = {
    "embeddings": 0,
    "pinecone": 1,
    "qdrant": 2,
    "milvus": 3,
    "weaviate": 4,
    "faiss": 5,
    "opensearch": 6,
    "elasticsearch": 7,
    "bm25": 8,
    "hybrid retrieval": 9,
    "semantic search": 10,
    "rag": 11,
    "retrieval": 12,
    "ranking": 13,
    "recommendation": 14,
    "learning-to-rank": 15,
    "python": 16,
    "llm": 17,
    "llms": 18,
    "fine-tuning": 19,
    "ndcg": 20,
    "mrr": 21,
    "map": 22,
    "a/b": 23,
    "ai": 50,
    "ml": 51,
}

UPPERCASE_TERMS = {"ai", "bm25", "faiss", "llm", "nlp", "rag", "ndcg", "mrr", "map"}


def _format_years(years: float) -> str:
    return f"{years:.1f}".rstrip("0").rstrip(".")


def _format_term(term: str) -> str:
    normalized = term.lower()
    if normalized in UPPERCASE_TERMS:
        return normalized.upper()
    if normalized == "llms":
        return "LLMs"
    return term.title()


def _pretty_terms(evidence: str) -> str:
    """Format comma-separated evidence terms for a readable CSV sentence."""

    if not evidence:
        return ""
    unique_terms = []
    for term in evidence.split(","):
        cleaned = term.strip()
        if cleaned and cleaned.lower() not in {term.lower() for term in unique_terms}:
            unique_terms.append(cleaned)

    ranked_terms = sorted(unique_terms, key=lambda term: (TERM_PRIORITY.get(term.lower(), 40), term.lower()))
    return ", ".join(_format_term(term) for term in ranked_terms[:3])


def _fit_band(row: Mapping[str, object], rank: int, evidence_channel_count: int) -> str:
    """Classify explanation tone from measured fit, falling back to rank for tests/fixtures."""

    has_scored_fields = "score" in row or "fit_score" in row
    if not has_scored_fields:
        if rank <= 10:
            return "strong"
        if rank <= 50:
            return "solid"
        return "borderline"

    score = float(row.get("score", 0.0) or 0.0)
    fit_score = float(row.get("fit_score", 0.0) or 0.0)
    penalty = float(row.get("suspiciousness_penalty", 0.0) or 0.0)
    consistency_risk = max(
        float(row.get("summary_only_match_score", 0.0) or 0.0),
        float(row.get("title_description_mismatch_score", 0.0) or 0.0),
    )

    if (
        (score >= 0.80 or fit_score >= 0.76)
        and evidence_channel_count >= 3
        and penalty <= 0.08
        and consistency_risk < 0.50
    ):
        return "strong"
    if (score >= 0.65 or fit_score >= 0.62) and evidence_channel_count >= 2 and penalty <= 0.20:
        return "solid"
    return "borderline"


def build_reasoning(row: Mapping[str, object], rank: int) -> str:
    """Create the final reasoning cell for one ranked candidate."""

    title = str(row.get("current_title", "Candidate"))
    company = str(row.get("current_company", "their current company"))
    years = float(row.get("years_of_experience", 0.0))
    title_evidence = _pretty_terms(str(row.get("title_evidence", "")))
    career_evidence = _pretty_terms(str(row.get("career_evidence", "")))
    skill_evidence = _pretty_terms(str(row.get("skill_evidence", "")))
    consensus_evidence = _pretty_terms(str(row.get("consensus_evidence", "")))
    evidence_channel_count = int(row.get("evidence_channel_count", 0) or 0)

    primary_evidence = career_evidence or consensus_evidence or skill_evidence or title_evidence
    channel_phrase = (
        f" across {evidence_channel_count} evidence channels"
        if evidence_channel_count
        else ""
    )
    fit_band = _fit_band(row, rank, evidence_channel_count)
    if fit_band == "strong":
        opener = (
            f"{title} at {company} with {_format_years(years)} years of experience{channel_phrase} and hands-on "
            f"{primary_evidence or 'role-specific evidence'}; this is a strong match for the JD."
        )
    elif fit_band == "solid":
        opener = (
            f"{title} at {company} with {_format_years(years)} years of experience{channel_phrase} and relevant "
            f"{primary_evidence or 'role-specific evidence'}; the fit is solid for the JD."
        )
    else:
        opener = (
            f"{title} at {company} with {_format_years(years)} years of experience{channel_phrase} shows "
            f"{primary_evidence or 'adjacent technical evidence'}; this profile looks more borderline for the JD."
        )

    support_parts: list[str] = []
    if float(row.get("early_maturity_signal", 0.0) or 0.0) >= 0.75:
        support_parts.append("early high-ownership evidence offsets lighter tenure")
    if support_parts:
        opener = opener.rstrip(".") + "; " + ", ".join(support_parts[:1]) + "."

    concern_parts: list[str] = []
    if float(row.get("summary_only_match_score", 0.0) or 0.0) >= 0.50:
        concern_parts.append("summary-only JD keywords")
    if float(row.get("title_description_mismatch_score", 0.0) or 0.0) >= 0.50:
        concern_parts.append("title-description mismatch")
    if float(row.get("overqualified_downgrade_risk", 0.0) or 0.0) >= 0.45:
        concern_parts.append("possible title-level downgrade risk")
    if float(row.get("underqualified_seniority_risk", 0.0) or 0.0) >= 0.45:
        concern_parts.append("seniority evidence is lighter than the JD")
    if float(row.get("management_mismatch_score", 0.0) or 0.0) >= 0.45:
        concern_parts.append("manager-to-hands-on mismatch risk")
    # Concerns are included only when measured features found them. No free-form
    # claims are added here.
    if concern_parts:
        opener = opener.rstrip(".") + "; consistency concern: " + ", ".join(concern_parts[:2]) + "."

    availability_parts: list[str] = []
    if row.get("open_to_work_flag"):
        availability_parts.append("open to work")
    response_rate = float(row.get("recruiter_response_rate", 0.0))
    if response_rate:
        availability_parts.append(f"{int(round(response_rate * 100))}% recruiter response rate")
    notice_period_days = int(row.get("notice_period_days", 0) or 0)
    if notice_period_days:
        availability_parts.append(f"{notice_period_days}-day notice period")
    days_since_active = int(row.get("days_since_active", 0) or 0)
    if days_since_active:
        availability_parts.append(f"active {days_since_active} days ago")

    if availability_parts:
        closer = "Availability signals include " + ", ".join(availability_parts[:3]) + "."
    else:
        closer = "Availability signals are neutral."

    return f"{opener} {closer}"
