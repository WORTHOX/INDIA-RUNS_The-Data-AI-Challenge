"""File IO helpers for candidate input and feature artifacts.

The full candidate dataset is large, so JSONL input is streamed. Feature output
is stored as parquet because ranking needs fast columnar reads.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import orjson
import pandas as pd


def iter_jsonl_candidates(path: str | Path) -> Iterator[dict]:
    """Yield candidates one line at a time to avoid loading the full dataset."""

    dataset_path = Path(path)
    with dataset_path.open("rb") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            yield orjson.loads(line)


def load_json_candidates(path: str | Path) -> list[dict]:
    json_path = Path(path)
    return orjson.loads(json_path.read_bytes())


def write_feature_frame(frame: pd.DataFrame, path: str | Path) -> None:
    """Persist feature rows in the format consumed by ranking."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)


def read_feature_frame(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(Path(path))
