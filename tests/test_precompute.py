from __future__ import annotations

import json

from redrob_ranker.precompute import run_precompute
from redrob_ranker.utils.hash_guard import verify_dataset_hash, verify_text_hash
from redrob_ranker.utils.io import read_feature_frame


def test_run_precompute_builds_feature_artifacts(tmp_path, sample_candidates):
    candidates_path = tmp_path / "candidates.jsonl"
    with candidates_path.open("w", encoding="utf-8") as handle:
        for row in sample_candidates[:5]:
            handle.write(json.dumps(row) + "\n")

    artifacts_dir = tmp_path / "artifacts"
    feature_path, hash_path = run_precompute(candidates_path, artifacts_dir, enable_dense=False)
    features = read_feature_frame(feature_path)

    assert feature_path.exists()
    assert hash_path.exists()
    assert verify_dataset_hash(candidates_path, hash_path)
    assert features.shape[0] == 5
    assert {"candidate_id", "title_score", "tfidf_score", "availability_multiplier"} <= set(features.columns)


def test_run_precompute_uses_supplied_job_description_for_title_contract(tmp_path, sample_candidates):
    ml_candidate = json.loads(json.dumps(sample_candidates[0]))
    ml_candidate["candidate_id"] = "CAND_9000001"
    ml_candidate["profile"]["current_title"] = "ML Engineer"
    ml_candidate["career_history"][0]["title"] = "ML Engineer"

    marketing_candidate = json.loads(json.dumps(sample_candidates[1]))
    marketing_candidate["candidate_id"] = "CAND_9000002"
    marketing_candidate["profile"]["current_title"] = "Marketing Manager"
    marketing_candidate["career_history"][0]["title"] = "Marketing Manager"
    marketing_candidate["career_history"][0]["description"] = (
        "Owned SEO, brand campaigns, content strategy, funnel conversion, and marketing analytics."
    )
    marketing_candidate["skills"] = [
        {"name": "SEO", "proficiency": "advanced", "endorsements": 12, "duration_months": 36},
        {"name": "Content Strategy", "proficiency": "advanced", "endorsements": 8, "duration_months": 30},
    ]

    candidates_path = tmp_path / "candidates.jsonl"
    with candidates_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(ml_candidate) + "\n")
        handle.write(json.dumps(marketing_candidate) + "\n")

    jd_path = tmp_path / "marketing_jd.txt"
    jd_text = "Job Description: Marketing Manager\nNeed SEO, content strategy, campaigns, and analytics."
    jd_path.write_text(jd_text, encoding="utf-8")

    artifacts_dir = tmp_path / "artifacts"
    feature_path, _ = run_precompute(candidates_path, artifacts_dir, job_description_path=jd_path, enable_dense=False)
    features = read_feature_frame(feature_path).set_index("candidate_id")

    assert verify_text_hash(jd_text, artifacts_dir / "job.hash")
    assert features.loc["CAND_9000002", "title_score"] > features.loc["CAND_9000001", "title_score"]
    assert features.loc["CAND_9000002", "negative_title_score"] == 0.0
