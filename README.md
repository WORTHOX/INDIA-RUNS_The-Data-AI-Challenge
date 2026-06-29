# Redrob Ranker

Evidence-consensus candidate ranking system for the Redrob hackathon bundle in this workspace. The code accepts a job description as an input, builds a dynamic JD contract from it, and ranks candidates from `candidates.jsonl` against that JD.

## What the repo does

- Streams `candidates.jsonl`
- Reads `.txt`, `.md`, or `.docx` job descriptions
- Normalizes profile, career, skill, and behavioral fields
- Precomputes reusable feature artifacts tied to both the candidate file and JD hash
- Scores candidates with structural fit, JD requirement coverage, evidence consensus, semantic support, availability, and suspiciousness penalties
- Emits a submission CSV with deterministic evidence-backed reasoning
- Provides a small Streamlit sandbox for sample runs

## Methodology

- [Beginner guide](docs/BEGINNER_GUIDE.md) explains the project from zero technical background: what was built, which technology is used, which file does what, and why the ranking/reasoning works.
- [Technical deep dive](docs/TECHNICAL_DEEP_DIVE.md) explains architecture, system design, frontend-to-backend flow, file call graph, and what happens when a new JD is supplied.
- [Methodology](docs/METHODOLOGY.md) explains the ranking approach, formulas, and reproducibility commands.
- [Candidate data analysis](docs/CANDIDATE_DATA_ANALYSIS.md) documents the dataset noise patterns that shaped the design.
- [Execution flow and hardships](docs/PROJECT_EXECUTION_FLOW.md) explains what runs when, what each file returns, and how the final CSV is deduced.

## Environment

```bash
python3.11 -m venv .venv311
. .venv311/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Precompute

Dense embeddings are optional at precompute time. The ranking step itself does not require network access.

```bash
python -m redrob_ranker.precompute \
  --candidates ./candidates.jsonl \
  --job-description ./job_description.docx \
  --artifacts ./artifacts
```

Skip dense embeddings:

```bash
python -m redrob_ranker.precompute \
  --candidates ./candidates.jsonl \
  --job-description ./job_description.docx \
  --artifacts ./artifacts \
  --skip-dense
```

## Ranking

This is the single reproduce command for submission generation once artifacts exist:

```bash
python rank.py \
  --candidates ./candidates.jsonl \
  --job-description ./job_description.docx \
  --artifacts ./artifacts \
  --participant-id TEAM_ID \
  --out ./TEAM_ID.csv
```

Expected output columns:

```text
candidate_id,rank,score,reasoning
```

## Validation

```bash
python validate_submission.py TEAM_ID.csv
```

## Sandbox

```bash
streamlit run sandbox/app.py
```

The sandbox accepts a small `.json`, `.jsonl`, or extensionless JSONL candidate sample plus an optional `.txt`, `.md`, or `.docx` JD, runs precompute plus ranking on that sample, and provides a downloadable CSV.

## Dev-set tooling

Build a manual-labeling template:

```bash
python scripts/build_dev_set.py \
  --features ./artifacts/candidate_features.parquet \
  --out ./dev_set.csv
```

Evaluate baselines against labeled relevance:

```bash
python scripts/evaluate_baselines.py \
  --features ./artifacts/candidate_features.parquet \
  --labels ./labels.csv
```

`labels.csv` must contain:

```text
candidate_id,relevance
```

## Ranking-time guarantees

- Ranking uses only local artifacts, the candidate file, and the supplied JD file
- No hosted API calls are needed during the final ranking step
- The code path for submission generation is deterministic given the same artifacts, candidate file, and JD file
- Ranking verifies both `artifacts/dataset.hash` and `artifacts/job.hash` before producing a CSV
