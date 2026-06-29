"""Compare ranking variants against manually labeled relevance.

Given a `labels.csv`, this script reports how keyword-only, title-only,
dense-only, and hybrid/evidence-consensus rankings perform on NDCG and
precision metrics.
"""

from __future__ import annotations

import argparse
from math import log2

import pandas as pd

from redrob_ranker.rank import score_feature_frame
from redrob_ranker.utils.io import read_feature_frame


def ndcg_at_k(relevance: list[float], k: int) -> float:
    gains = relevance[:k]
    dcg = sum((2**rel - 1) / log2(index + 2) for index, rel in enumerate(gains))
    ideal = sorted(relevance, reverse=True)[:k]
    idcg = sum((2**rel - 1) / log2(index + 2) for index, rel in enumerate(ideal))
    return 0.0 if idcg == 0 else dcg / idcg


def precision_at_k(relevance: list[float], k: int, *, relevant_threshold: float = 3.0) -> float:
    gains = relevance[:k]
    if not gains:
        return 0.0
    hits = sum(1 for rel in gains if rel >= relevant_threshold)
    return hits / min(k, len(gains))


def evaluate_ranker(scored_frame: pd.DataFrame, labels: pd.DataFrame, score_column: str) -> dict[str, float]:
    """Score one ranking column against human labels."""

    merged = scored_frame.merge(labels, on="candidate_id", how="inner")
    ordered = merged.sort_values(by=[score_column, "candidate_id"], ascending=[False, True])
    relevance = ordered["relevance"].astype(float).tolist()
    return {
        "ndcg@10": round(ndcg_at_k(relevance, 10), 4),
        "ndcg@50": round(ndcg_at_k(relevance, 50), 4),
        "precision@20": round(precision_at_k(relevance, 20), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate baseline rankers against labeled relevance judgments.")
    parser.add_argument("--features", required=True, help="Path to candidate_features.parquet")
    parser.add_argument("--labels", required=True, help="CSV with candidate_id and relevance columns")
    args = parser.parse_args()

    feature_frame = read_feature_frame(args.features)
    labels = pd.read_csv(args.labels)
    scored = score_feature_frame(feature_frame)

    baseline_columns = {
        "keyword_only": "tfidf_score",
        "title_only": "title_score",
        "dense_only": "dense_score",
        "hybrid": "score",
    }

    rows = []
    for name, column in baseline_columns.items():
        metrics = evaluate_ranker(scored, labels, column)
        rows.append({"ranker": name, **metrics})

    report = pd.DataFrame(rows)
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
