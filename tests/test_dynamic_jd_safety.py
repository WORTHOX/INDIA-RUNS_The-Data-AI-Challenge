from __future__ import annotations

import json

import pytest

from redrob_ranker.job_description import build_job_contract
from redrob_ranker.precompute import run_precompute
from redrob_ranker.rank import run_ranking


def test_title_extraction_ignores_negative_examples_later_in_jd():
    jd_text = """
    Job Description: Senior AI Engineer

    We need retrieval and ranking experience.
    A candidate with keyword-stuffed AI skills but a Marketing Manager title is not a fit.
    """

    contract = build_job_contract(jd_text)

    assert "senior ai engineer" in contract.positive_title_terms
    assert "marketing manager" not in contract.positive_title_terms


def test_ranking_rejects_artifacts_created_for_a_different_jd(tmp_path, sample_candidates):
    candidates_path = tmp_path / "candidates.jsonl"
    with candidates_path.open("w", encoding="utf-8") as handle:
        for row in sample_candidates[:5]:
            handle.write(json.dumps(row) + "\n")

    original_jd = tmp_path / "ai_jd.txt"
    original_jd.write_text("Job Description: Senior AI Engineer\nNeed retrieval and ranking.", encoding="utf-8")
    wrong_jd = tmp_path / "marketing_jd.txt"
    wrong_jd.write_text("Job Description: Marketing Manager\nNeed SEO and campaigns.", encoding="utf-8")

    artifacts_dir = tmp_path / "artifacts"
    output_path = tmp_path / "submission.csv"
    run_precompute(candidates_path, artifacts_dir, job_description_path=original_jd, enable_dense=False)

    with pytest.raises(ValueError, match="Text hash mismatch"):
        run_ranking(candidates_path, artifacts_dir, output_path, job_description_path=wrong_jd, top_n=5)


def test_ranking_accepts_same_jd_when_text_needs_normalization(tmp_path, sample_candidates):
    candidates_path = tmp_path / "candidates.jsonl"
    with candidates_path.open("w", encoding="utf-8") as handle:
        for row in sample_candidates[:5]:
            handle.write(json.dumps(row) + "\n")

    jd_path = tmp_path / "ai_jd.txt"
    jd_path.write_text("Job Description: Senior AI Engineer — Founding Team\nNeed retrieval and ranking.", encoding="utf-8")

    artifacts_dir = tmp_path / "artifacts"
    output_path = tmp_path / "submission.csv"
    run_precompute(candidates_path, artifacts_dir, job_description_path=jd_path, enable_dense=False)

    submission = run_ranking(candidates_path, artifacts_dir, output_path, job_description_path=jd_path, top_n=5)

    assert submission.shape[0] == 5
