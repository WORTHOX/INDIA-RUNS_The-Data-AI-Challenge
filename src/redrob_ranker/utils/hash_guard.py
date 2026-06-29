"""Hash helpers that prevent artifact/data/JD mismatches.

Precompute writes hashes beside the parquet artifact. Ranking verifies those
hashes before producing a CSV, so stale artifacts cannot silently be reused with
a different candidate file or job description.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def compute_file_hash(path: str | Path) -> str:
    """Compute a streaming SHA-256 hash for large files."""

    file_path = Path(path)
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_text_hash(text: str) -> str:
    """Hash text after whitespace normalization."""

    normalized = " ".join(text.strip().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def compute_dataset_hash(path: str | Path) -> str:
    return compute_file_hash(path)


def write_dataset_hash(path: str | Path, dataset_hash: str) -> None:
    hash_path = Path(path)
    hash_path.parent.mkdir(parents=True, exist_ok=True)
    hash_path.write_text(dataset_hash.strip() + "\n", encoding="utf-8")


def read_dataset_hash(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8").strip()


def verify_dataset_hash(dataset_path: str | Path, hash_path: str | Path) -> str:
    """Raise if the current candidate file does not match the artifact hash."""

    expected_hash = read_dataset_hash(hash_path)
    actual_hash = compute_dataset_hash(dataset_path)
    if actual_hash != expected_hash:
        raise ValueError(
            f"Dataset hash mismatch: expected {expected_hash}, found {actual_hash}"
        )
    return actual_hash


def write_text_hash(path: str | Path, text_hash: str) -> None:
    write_dataset_hash(path, text_hash)


def verify_text_hash(text: str, hash_path: str | Path) -> str:
    """Raise if the current text does not match a stored text hash."""

    expected_hash = read_dataset_hash(hash_path)
    actual_hash = compute_text_hash(text)
    if actual_hash != expected_hash:
        raise ValueError(f"Text hash mismatch: expected {expected_hash}, found {actual_hash}")
    return actual_hash
