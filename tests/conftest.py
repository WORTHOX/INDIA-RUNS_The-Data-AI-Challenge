from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def sample_candidates() -> list[dict]:
    path = Path(__file__).resolve().parents[1] / "sample_candidates.json"
    return json.loads(path.read_text())


@pytest.fixture()
def sample_candidate(sample_candidates: list[dict]) -> dict:
    return sample_candidates[0]
