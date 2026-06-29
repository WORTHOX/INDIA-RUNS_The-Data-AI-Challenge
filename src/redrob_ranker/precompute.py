"""Build reusable ranking artifacts from candidates and a supplied JD.

Precompute is the heavier phase. It streams the JSONL candidate file, extracts
all deterministic features, computes semantic scores, and writes the parquet
artifact consumed later by the fast ranking command.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from redrob_ranker.features.behavioral import compute_reference_date, extract_behavioral_features
from redrob_ranker.features.career_history import extract_career_features
from redrob_ranker.features.evidence_consensus import extract_evidence_consensus_features
from redrob_ranker.features.honeypot import extract_honeypot_features
from redrob_ranker.features.normalize import build_normalized_candidate
from redrob_ranker.features.skills import extract_skill_features
from redrob_ranker.features.title_alignment import extract_title_features
from redrob_ranker.job_description import build_job_contract, compute_job_text_hash, read_job_description_text
from redrob_ranker.scoring.semantic import (
    build_candidate_semantic_text,
    build_query_text,
    compute_dense_scores,
    compute_tfidf_scores,
)
from redrob_ranker.utils.hash_guard import compute_dataset_hash, write_dataset_hash
from redrob_ranker.utils.hash_guard import write_text_hash
from redrob_ranker.utils.io import iter_jsonl_candidates, write_feature_frame


def build_feature_frame(
    candidates_path: str | Path,
    *,
    job_description_text: str | None = None,
    enable_dense: bool = True,
) -> pd.DataFrame:
    """Return one feature row per candidate for the supplied candidate file."""

    contract = build_job_contract(job_description_text)
    # Use the dataset's own latest activity date as "today"; this keeps recency
    # features stable even when the code is run months later.
    reference_date = compute_reference_date(
        record.get("redrob_signals", {}).get("last_active_date", "") for record in iter_jsonl_candidates(candidates_path)
    )

    records: list[dict] = []
    semantic_texts: list[str] = []
    # Stream the full dataset. Each loop iteration becomes one parquet row.
    for candidate in iter_jsonl_candidates(candidates_path):
        normalized = build_normalized_candidate(candidate)
        title_features = extract_title_features(normalized, contract)
        career_features = extract_career_features(normalized, contract)
        skill_features = extract_skill_features(normalized, contract)
        consensus_features = extract_evidence_consensus_features(
            normalized,
            contract,
            title_features=title_features,
            career_features=career_features,
            skill_features=skill_features,
        )
        behavioral_features = extract_behavioral_features(normalized, reference_date, contract)
        honeypot_features = extract_honeypot_features(normalized, contract)

        # The record combines identity fields, searchable text, and all feature
        # dictionaries. Ranking later reads only this parquet output.
        record = {
            "candidate_id": normalized.candidate_id,
            "current_title": normalized.current_title,
            "current_company": normalized.current_company,
            "location": normalized.location,
            "country": normalized.country,
            "years_of_experience": normalized.years_of_experience,
            "current_industry": normalized.current_industry,
            "headline": normalized.headline,
            "summary": normalized.summary,
            "open_to_work_flag": normalized.open_to_work_flag,
            "recruiter_response_rate": normalized.recruiter_response_rate,
            "notice_period_days": normalized.notice_period_days,
            "github_activity_score": normalized.github_activity_score,
            "interview_completion_rate": normalized.interview_completion_rate,
            "preferred_work_mode": normalized.preferred_work_mode,
            "willing_to_relocate": normalized.willing_to_relocate,
            "last_active_date": normalized.last_active_date,
            "profile_text": normalized.profile_text,
            "career_text": normalized.career_text,
            "skills_text": normalized.skills_text,
            **title_features,
            **career_features,
            **skill_features,
            **consensus_features,
            **behavioral_features,
            **honeypot_features,
        }
        records.append(record)
        semantic_texts.append(build_candidate_semantic_text(normalized))

    # Semantic scores are computed after collecting candidate texts so TF-IDF
    # can compare the JD against the whole corpus at once.
    query_text = build_query_text(contract)
    tfidf_scores = compute_tfidf_scores(semantic_texts, query_text)
    dense_scores = compute_dense_scores(semantic_texts, query_text, enable_dense=enable_dense)

    for index, record in enumerate(records):
        record["tfidf_score"] = round(float(tfidf_scores[index]), 6)
        record["dense_score"] = round(float(dense_scores[index]), 6)

    return pd.DataFrame(records)


def run_precompute(
    candidates_path: str | Path,
    artifacts_dir: str | Path,
    *,
    job_description_path: str | Path | None = None,
    enable_dense: bool = True,
) -> tuple[Path, Path]:
    """Write feature parquet plus dataset/JD hash guards."""

    artifacts_path = Path(artifacts_dir)
    artifacts_path.mkdir(parents=True, exist_ok=True)

    job_description_text = read_job_description_text(job_description_path)
    feature_frame = build_feature_frame(
        candidates_path,
        job_description_text=job_description_text,
        enable_dense=enable_dense,
    )
    feature_path = artifacts_path / "candidate_features.parquet"
    hash_path = artifacts_path / "dataset.hash"
    job_hash_path = artifacts_path / "job.hash"

    write_feature_frame(feature_frame, feature_path)
    write_dataset_hash(hash_path, compute_dataset_hash(candidates_path))
    write_text_hash(job_hash_path, compute_job_text_hash(job_description_text))
    return feature_path, hash_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute Redrob candidate feature artifacts.")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    parser.add_argument(
        "--job-description",
        required=False,
        help="Path to the JD file. Supports .txt/.md and .docx. Defaults to ./job_description.docx when present.",
    )
    parser.add_argument("--artifacts", required=True, help="Directory to write parquet artifacts")
    parser.add_argument(
        "--skip-dense",
        action="store_true",
        help="Disable dense embedding similarity during precompute",
    )
    args = parser.parse_args()

    run_precompute(
        args.candidates,
        args.artifacts,
        job_description_path=args.job_description,
        enable_dense=not args.skip_dense,
    )


if __name__ == "__main__":
    main()
