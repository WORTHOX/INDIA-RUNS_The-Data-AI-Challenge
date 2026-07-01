# Methodology: Evidence-Consensus JD Ranker

## Objective

The system ranks LinkedIn-style professional profiles against a supplied job description. The output is a top-100 CSV with deterministic scores and human-readable reasoning. The design goal is not just keyword relevance; it is to identify candidates whose title, career history, skills, and platform activity agree with the JD.

## Dataset Finding That Drives The Method

Before modeling, we inspected the full 100,000-candidate JSONL dataset. The data is intentionally noisy:

- many non-technical profiles mention AI in summaries;
- repeated role-description templates appear thousands of times;
- role titles and role descriptions often describe different domains;
- some profiles contain skill stuffing or weakly supported expertise claims.

Because of this, semantic similarity alone is unsafe. The ranker uses semantic support only as a capped boost after structural evidence exists.

## Pipeline

1. **JD contract extraction**
   - Reads `.docx`, `.txt`, or `.md` JD files.
   - Extracts target title, experience band, must-have requirements, nice-to-have requirements, exclusions, preferred locations/countries, and high-signal evidence keywords.
   - Hashes the JD so precomputed artifacts cannot be reused with a different role.

2. **Streaming candidate normalization**
   - Streams the 465 MB JSONL file line by line.
   - Flattens profile, career history, skills, and Redrob platform signals into normalized records.

3. **Evidence feature extraction**
   - Title alignment: current/recent titles against the JD target role.
   - Career evidence: production, retrieval/ranking, evaluation, product/service background, and JD-specific keywords.
   - Skill evidence: direct skill support, proficiency sanity, duration sanity, Python/vector/evaluation/JD coverage.
   - Seniority evidence: target JD level vs candidate title level, early high-ownership maturity, downgrade risk, and manager-to-hands-on mismatch.
   - Behavioral evidence: recency, open-to-work, response rate, notice period, relocation/location/country fit.
   - Evidence consensus: how many independent channels support the match: title, career, skills, and JD requirement coverage.
   - Noise checks: summary-only keyword matches and title-description mismatch risk.

4. **Hybrid scoring**
   - Structural fit is the base.
   - Semantic similarity is capped and gated.
   - Availability adjusts the score.
   - Suspiciousness penalties reduce noisy or unsupported profiles.
   - Final scores are clamped to `0.0-1.0` and sorted deterministically.

## Scoring Formula

```text
final_score = clamp_0_1(
  (fit_score + semantic_support) * availability_multiplier
  - suspiciousness_penalty
)
```

```text
fit_score =
  0.20 * title_score +
  0.20 * career_score +
  0.17 * role_specific_signal +
  0.12 * evidence_consensus_score +
  0.08 * seniority_alignment_score +
  0.07 * work_maturity_score +
  0.06 * evaluation_signal +
  0.04 * python_score +
  0.03 * product_background_score +
  0.03 * experience_score
```

```text
role_specific_signal =
  0.35 * max(job_keyword_score, job_skill_score) +
  0.25 * requirement_coverage_score +
  0.25 * mean(retrieval_score, retrieval_skill_score, vector_score) +
  0.15 * max(retrieval_score, retrieval_skill_score, vector_score)
)
```

This avoids the old single-channel saturation problem: a candidate with one perfect keyword hit should not beat another candidate with broad JD evidence across career, skills, and requirements.

Seniority is not treated as years alone. The ranker estimates the JD's target career level and compares it with the candidate's current and peak titles. It then uses role descriptions to detect mature work such as ownership, architecture, production launch, scaling, end-to-end delivery, and mentoring. This allows a lower-tenure candidate with real senior-level project ownership to stay competitive, while adding risk for profiles that look overqualified for a lower-level role or too management-heavy for a hands-on IC role.

Semantic support is limited to `0.15` and is disabled for profiles that only match the JD in their summary without supporting title/career/skill evidence.

## Explainability

The reasoning column is generated from stored feature evidence only. It does not call an LLM, so it cannot invent unsupported claims. Each explanation mentions:

- current role and company;
- experience;
- number of supporting evidence channels;
- strongest matched evidence terms;
- availability signals;
- consistency concerns when present.

Reasoning tone is based on score, fit score, evidence-channel count, and measured consistency risk. It is not based on a hard rank cutoff, so rank 10 and rank 11 do not get artificially different wording when their evidence is nearly identical.

## Reproducibility

Precompute:

```bash
python -m redrob_ranker.precompute \
  --candidates ./candidates.jsonl \
  --job-description ./job_description.docx \
  --artifacts ./artifacts \
  --skip-dense
```

Rank:

```bash
python rank.py \
  --candidates ./candidates.jsonl \
  --job-description ./job_description.docx \
  --artifacts ./artifacts \
  --participant-id HACKATHON_TEAM \
  --out ./HACKATHON_TEAM.csv
```

Validate:

```bash
python validate_submission.py HACKATHON_TEAM.csv
```

The final ranking command uses only local files and precomputed artifacts. No network calls are required.
