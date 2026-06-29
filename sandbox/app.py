"""Small Streamlit demo for the same ranking pipeline.

The sandbox is for small samples and presentation, not the full 465 MB dataset.
It writes uploaded data into a temporary folder, runs precompute, then runs the
same ranking code used for final submission generation.
"""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import streamlit as st

from redrob_ranker.precompute import run_precompute
from redrob_ranker.rank import run_ranking


def _write_uploaded_candidates(upload, target_path: Path) -> None:
    """Normalize uploaded JSON or JSONL-like content into JSONL."""

    suffix = Path(upload.name).suffix.lower()
    content = upload.getvalue()
    if suffix == ".json":
        rows = json.loads(content.decode("utf-8"))
        with target_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        return

    target_path.write_bytes(content)


st.set_page_config(page_title="Redrob Ranker Sandbox", layout="wide")
st.title("Redrob Ranker Sandbox")
st.caption("Upload a small JSON or JSONL sample and run the same ranking pipeline used for submission generation.")

uploaded = st.file_uploader("Candidate sample", type=["json", "jsonl"])
enable_dense = st.checkbox("Enable dense embeddings during sandbox precompute", value=False)

if uploaded and st.button("Run ranker", type="primary"):
    # TemporaryDirectory keeps sandbox artifacts isolated from real submission
    # artifacts in ./artifacts.
    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        candidates_path = temp_path / "sandbox_candidates.jsonl"
        artifacts_dir = temp_path / "artifacts"
        output_path = temp_path / "sandbox_submission.csv"

        _write_uploaded_candidates(uploaded, candidates_path)
        run_precompute(candidates_path, artifacts_dir, enable_dense=enable_dense)
        submission = run_ranking(candidates_path, artifacts_dir, output_path, top_n=100)

        st.subheader("Ranked output")
        st.dataframe(submission, use_container_width=True)
        st.download_button(
            "Download CSV",
            data=output_path.read_bytes(),
            file_name="sandbox_submission.csv",
            mime="text/csv",
        )
elif not uploaded:
    st.info("Upload up to 100 candidates as `.json` or `.jsonl` to exercise the ranker end to end.")
