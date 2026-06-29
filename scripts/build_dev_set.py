"""Create a stratified manual-labeling set from precomputed features.

This is optional evaluation tooling. It samples obvious positives, hidden gems,
keyword stuffers, unavailable candidates, and suspicious profiles so manual
labels cover more than just the top of the ranking.
"""

from __future__ import annotations

import argparse

import pandas as pd

from redrob_ranker.rank import score_feature_frame
from redrob_ranker.utils.io import read_feature_frame


def _sample_group(frame: pd.DataFrame, size: int, seed: int) -> pd.DataFrame:
    if frame.empty:
        return frame
    if len(frame) <= size:
        return frame.copy()
    return frame.sample(n=size, random_state=seed)


def build_dev_set(features: pd.DataFrame, *, per_stratum: int = 30, seed: int = 7) -> pd.DataFrame:
    """Return a CSV-ready candidate list for human relevance labeling."""

    scored = score_feature_frame(features)
    strata = {
        "obvious_positive": scored[
            (scored["title_score"] >= 0.65)
            & (scored["retrieval_score"] >= 0.35)
            & (scored["availability_multiplier"] >= 0.95)
        ],
        "hidden_gem": scored[
            (scored["title_score"].between(0.10, 0.60))
            & (scored["career_score"] >= 0.45)
            & (scored["retrieval_score"] >= 0.20)
        ],
        "keyword_stuffer": scored[
            (scored["negative_title_score"] >= 0.60)
            & (scored["ai_like_skill_count"] >= 4)
        ],
        "unavailable": scored[
            ((scored["days_since_active"] > 180) | (scored["recruiter_response_rate"] < 0.10))
            & (scored["title_score"] >= 0.50)
        ],
        "suspicious": scored[
            (scored["claimed_experience_gap_years"] > 3.0)
            | (scored["short_duration_expert_count"] >= 10)
            | (scored["research_only_score"] >= 1.0)
        ],
    }

    selections = []
    for index, (stratum, frame) in enumerate(strata.items()):
        sampled = _sample_group(frame, per_stratum, seed + index)
        if sampled.empty:
            continue
        output = sampled[["candidate_id"]].copy()
        output["stratum"] = stratum
        output["label"] = ""
        selections.append(output)

    if not selections:
        return pd.DataFrame(columns=["candidate_id", "stratum", "label"])
    return pd.concat(selections, ignore_index=True).drop_duplicates(subset=["candidate_id"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a stratified manual-labeling dev set.")
    parser.add_argument("--features", required=True, help="Path to candidate_features.parquet")
    parser.add_argument("--out", required=True, help="CSV path to write the dev set template")
    parser.add_argument("--per-stratum", type=int, default=30, help="Max candidates per stratum")
    args = parser.parse_args()

    feature_frame = read_feature_frame(args.features)
    dev_set = build_dev_set(feature_frame, per_stratum=args.per_stratum)
    dev_set.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
