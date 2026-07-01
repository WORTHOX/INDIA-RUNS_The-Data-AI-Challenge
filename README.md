# Redrob Ranker

Hybrid candidate ranker for the India Runs Data and AI Challenge.

The system ranks LinkedIn-style candidate profiles against a supplied job description and emits top-100 CSV and XLSX outputs with deterministic scores and evidence-backed reasoning.

## What It Does

- Parses `.docx`, `.txt`, or `.md` job descriptions into a dynamic JD contract.
- Processes large JSONL candidate files into reusable feature artifacts.
- Scores title fit, career evidence, skill evidence, seniority fit, availability, semantic support, and suspiciousness risk.
- Rewards candidates whose title, career history, skills, and JD requirement coverage agree.
- Handles seniority more realistically: strong ownership can offset lighter tenure, while downgrade or manager-to-hands-on mismatch can reduce fit.
- Produces `HACKATHON_TEAM.csv` in the required submission format and `HACKATHON_TEAM.xlsx` for spreadsheet review.

## Key Files

```text
rank.py                         Root CLI wrapper for final ranking
src/redrob_ranker/precompute.py  Builds feature artifacts from candidates + JD
src/redrob_ranker/rank.py        Scores candidates and writes top-100 CSV/XLSX
src/redrob_ranker/job_description.py  Extracts JD requirements and target role profile
METHODOLOGY.md                  Concise explanation of the ranking method
HACKATHON_TEAM.csv              Current generated CSV submission
HACKATHON_TEAM.xlsx             Current generated spreadsheet output
validate_submission.py          Submission format validator
```

## Setup

```bash
python3.11 -m venv .venv311
source .venv311/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Reproduce Submission

Precompute features:

```bash
python -m redrob_ranker.precompute \
  --candidates candidates.jsonl \
  --job-description job_description.docx \
  --artifacts artifacts \
  --skip-dense
```

Generate ranked CSV and XLSX:

```bash
python rank.py \
  --candidates candidates.jsonl \
  --job-description job_description.docx \
  --artifacts artifacts \
  --participant-id HACKATHON_TEAM \
  --out HACKATHON_TEAM.csv
```

Validate:

```bash
python validate_submission.py HACKATHON_TEAM.csv
```

## Output Format

```text
candidate_id,rank,score,reasoning
```

The CSV is the official validator-compatible file. The XLSX contains the same columns and rows in a readable spreadsheet. The final ranking path is deterministic for the same candidate file, JD, and generated artifacts. No hosted API is required for ranking.

## Optional Sandbox

```bash
streamlit run sandbox/app.py
```

The sandbox is a lightweight local UI for testing small candidate samples with the project default JD.
