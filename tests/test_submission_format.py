from __future__ import annotations

import pandas as pd

from redrob_ranker.rank import build_ranked_submission, write_submission_csv
from validate_submission import validate_submission


def test_generated_submission_passes_validator(tmp_path):
    rows = []
    for index in range(100):
        rows.append(
            {
                "candidate_id": f"CAND_{index:07d}",
                "current_title": "ML Engineer",
                "current_company": "ProdCo",
                "location": "Pune, Maharashtra",
                "years_of_experience": 6.0,
                "title_score": 0.85,
                "negative_title_score": 0.0,
                "career_score": 0.75,
                "retrieval_score": 0.65,
                "retrieval_skill_score": 0.55,
                "evaluation_score": 0.35,
                "evaluation_skill_score": 0.25,
                "python_score": 1.0,
                "product_background_score": 0.85,
                "service_background_score": 0.0,
                "vector_score": 0.7,
                "tfidf_score": 0.4 - index * 0.001,
                "dense_score": 0.3 - index * 0.001,
                "availability_multiplier": 1.01,
                "open_to_work_flag": True,
                "recruiter_response_rate": 0.71,
                "notice_period_days": 30,
                "days_since_active": 7,
                "title_evidence": "ml engineer",
                "career_evidence": "elasticsearch, faiss",
                "skill_evidence": "python, pinecone",
                "claimed_experience_gap_years": 0.0,
                "expert_skill_count": 3,
                "short_duration_expert_count": 0,
                "ai_like_skill_count": 4,
                "research_only_score": 0.0,
            }
        )

    ranked = build_ranked_submission(pd.DataFrame(rows), top_n=100)
    output_path = tmp_path / "team_redrob.csv"
    write_submission_csv(ranked, output_path)

    assert validate_submission(output_path) == []
