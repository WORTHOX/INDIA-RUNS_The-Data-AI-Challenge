from __future__ import annotations

import pandas as pd

from redrob_ranker.rank import build_ranked_submission


def test_build_ranked_submission_returns_deterministic_top_100():
    rows = []
    for index in range(105):
        rows.append(
            {
                "candidate_id": f"CAND_{index:07d}",
                "current_title": "ML Engineer" if index % 2 == 0 else "Marketing Manager",
                "current_company": "ProdCo",
                "location": "Pune, Maharashtra",
                "years_of_experience": 6.0,
                "title_score": 0.9 if index % 2 == 0 else 0.0,
                "negative_title_score": 0.0 if index % 2 == 0 else 0.9,
                "career_score": 0.8 if index % 2 == 0 else 0.1,
                "retrieval_score": 0.7 if index % 2 == 0 else 0.0,
                "retrieval_skill_score": 0.6 if index % 2 == 0 else 0.0,
                "evaluation_score": 0.4 if index % 2 == 0 else 0.0,
                "evaluation_skill_score": 0.3 if index % 2 == 0 else 0.0,
                "python_score": 1.0 if index % 2 == 0 else 0.0,
                "product_background_score": 0.8,
                "service_background_score": 0.0 if index % 2 == 0 else 1.0,
                "vector_score": 0.6 if index % 2 == 0 else 0.0,
                "tfidf_score": 0.4 if index % 2 == 0 else 0.05,
                "dense_score": 0.3 if index % 2 == 0 else 0.0,
                "availability_multiplier": 1.02 if index % 2 == 0 else 0.82,
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
                "ai_like_skill_count": 4 if index % 2 == 0 else 5,
                "research_only_score": 0.0,
            }
        )

    ranked = build_ranked_submission(pd.DataFrame(rows), top_n=100)

    assert ranked.shape[0] == 100
    assert ranked["rank"].tolist() == list(range(1, 101))
    assert ranked["score"].tolist() == sorted(ranked["score"].tolist(), reverse=True)
    assert ranked.iloc[0]["candidate_id"].startswith("CAND_")
    assert ranked["score"].max() <= 1.0
