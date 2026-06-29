from __future__ import annotations

import json

import pandas as pd

from redrob_ranker.utils.hash_guard import (
    compute_dataset_hash,
    verify_dataset_hash,
    write_dataset_hash,
)
from redrob_ranker.utils.io import (
    iter_jsonl_candidates,
    read_feature_frame,
    write_feature_frame,
)


def test_iter_jsonl_candidates_streams_records(tmp_path, sample_candidates):
    jsonl_path = tmp_path / "candidates.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in sample_candidates[:2]:
            handle.write(json.dumps(row) + "\n")

    records = list(iter_jsonl_candidates(jsonl_path))

    assert [record["candidate_id"] for record in records] == ["CAND_0000001", "CAND_0000002"]


def test_write_and_read_feature_frame_round_trip(tmp_path):
    frame = pd.DataFrame(
        [
            {"candidate_id": "CAND_0000001", "title_score": 0.7, "retrieval_score": 0.2},
            {"candidate_id": "CAND_0000002", "title_score": 0.0, "retrieval_score": 0.1},
        ]
    )
    output_path = tmp_path / "candidate_features.parquet"

    write_feature_frame(frame, output_path)
    loaded = read_feature_frame(output_path)

    assert loaded.to_dict("records") == frame.to_dict("records")


def test_dataset_hash_round_trip(tmp_path, sample_candidates):
    jsonl_path = tmp_path / "candidates.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in sample_candidates[:3]:
            handle.write(json.dumps(row) + "\n")

    hash_path = tmp_path / "dataset.hash"
    dataset_hash = compute_dataset_hash(jsonl_path)
    write_dataset_hash(hash_path, dataset_hash)

    assert verify_dataset_hash(jsonl_path, hash_path) == dataset_hash
