from __future__ import annotations

from redrob_ranker.reasoning.evidence_builder import build_reasoning


def test_build_reasoning_uses_feature_evidence_without_hallucination():
    row = {
        "candidate_id": "CAND_1234567",
        "current_title": "ML Engineer",
        "current_company": "PhonePe",
        "years_of_experience": 6.5,
        "title_evidence": "ml engineer, search engineer",
        "career_evidence": "elasticsearch, faiss, ndcg",
        "skill_evidence": "python, pinecone, embeddings",
        "open_to_work_flag": True,
        "recruiter_response_rate": 0.74,
        "notice_period_days": 30,
        "days_since_active": 5,
        "evidence_channel_count": 4,
        "consensus_evidence": "python, pinecone, embeddings",
        "summary_only_match_score": 0.0,
        "title_description_mismatch_score": 0.0,
    }

    reasoning = build_reasoning(row, rank=3)

    assert "PhonePe" in reasoning
    assert "ML Engineer" in reasoning
    assert "4 evidence channels" in reasoning
    assert "FAISS" in reasoning
    assert "30-day notice period" in reasoning


def test_build_reasoning_surfaces_consistency_concerns():
    row = {
        "candidate_id": "CAND_1234568",
        "current_title": "Marketing Manager",
        "current_company": "Acme Corp",
        "years_of_experience": 6.5,
        "title_evidence": "",
        "career_evidence": "python, spark",
        "skill_evidence": "",
        "open_to_work_flag": False,
        "recruiter_response_rate": 0.20,
        "notice_period_days": 90,
        "days_since_active": 80,
        "evidence_channel_count": 1,
        "consensus_evidence": "",
        "summary_only_match_score": 0.8,
        "title_description_mismatch_score": 0.7,
    }

    reasoning = build_reasoning(row, rank=80)

    assert "consistency concern" in reasoning
    assert "summary-only JD keywords" in reasoning


def test_build_reasoning_uses_fit_strength_instead_of_rank_cutoff():
    row = {
        "candidate_id": "CAND_7654321",
        "current_title": "NLP Engineer",
        "current_company": "Aganitha",
        "years_of_experience": 6.6,
        "title_evidence": "nlp engineer",
        "career_evidence": "a/b, ai, embeddings, pinecone, rag, ranking",
        "skill_evidence": "python, faiss, embeddings",
        "open_to_work_flag": True,
        "recruiter_response_rate": 0.88,
        "notice_period_days": 30,
        "days_since_active": 60,
        "evidence_channel_count": 3,
        "consensus_evidence": "embeddings, faiss, llm",
        "summary_only_match_score": 0.0,
        "title_description_mismatch_score": 0.0,
        "fit_score": 0.79,
        "score": 0.835,
        "suspiciousness_penalty": 0.0,
    }

    reasoning = build_reasoning(row, rank=11)

    assert "strong match" in reasoning
    assert "not top-tier" not in reasoning


def test_build_reasoning_prioritizes_specific_evidence_terms_over_generic_ai():
    row = {
        "candidate_id": "CAND_7654322",
        "current_title": "AI Engineer",
        "current_company": "Ola",
        "years_of_experience": 5.3,
        "title_evidence": "ai engineer",
        "career_evidence": "a/b, ai, embeddings, pinecone, rag, ranking",
        "skill_evidence": "bm25, embeddings, opensearch",
        "open_to_work_flag": True,
        "recruiter_response_rate": 0.81,
        "notice_period_days": 60,
        "days_since_active": 10,
        "evidence_channel_count": 3,
        "consensus_evidence": "embeddings, opensearch, pinecone",
        "summary_only_match_score": 0.0,
        "title_description_mismatch_score": 0.0,
        "fit_score": 0.79,
        "score": 0.835,
        "suspiciousness_penalty": 0.0,
    }

    reasoning = build_reasoning(row, rank=10)

    assert "Embeddings" in reasoning
    assert "Pinecone" in reasoning
    assert "RAG" in reasoning
    assert "A/B, AI, Embeddings" not in reasoning
