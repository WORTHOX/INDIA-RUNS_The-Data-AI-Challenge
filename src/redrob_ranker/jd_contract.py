"""Shared job-description contract.

Feature extractors do not read the raw JD directly. They receive a JobContract,
which contains the parsed target title, requirements, exclusions, and keyword
families needed for scoring.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class JobContract:
    job_query_text: str
    job_keywords: tuple[str, ...]
    evidence_keywords: tuple[str, ...]
    must_have_keywords: tuple[str, ...]
    nice_to_have_keywords: tuple[str, ...]
    exclusion_keywords: tuple[str, ...]
    min_years: float
    max_years: float
    target_seniority_level: float
    target_seniority_label: str
    target_role_mode: str
    positive_title_terms: tuple[str, ...]
    positive_title_hints: tuple[str, ...]
    negative_title_terms: tuple[str, ...]
    negative_title_hints: tuple[str, ...]
    service_companies: tuple[str, ...]
    service_industries: tuple[str, ...]
    preferred_locations: tuple[str, ...]
    preferred_countries: tuple[str, ...]
    retrieval_keywords: tuple[str, ...]
    vector_keywords: tuple[str, ...]
    evaluation_keywords: tuple[str, ...]
    python_keywords: tuple[str, ...]
    llm_keywords: tuple[str, ...]
    nlp_keywords: tuple[str, ...]
    cv_keywords: tuple[str, ...]
    speech_keywords: tuple[str, ...]
    shipping_keywords: tuple[str, ...]
    research_keywords: tuple[str, ...]


DEFAULT_JD_CONTRACT = JobContract(
    job_query_text=(
        "senior ai engineer founding team retrieval ranking recommendation embeddings vector databases "
        "python evaluation ndcg mrr map production machine learning product company"
    ),
    job_keywords=(
        "senior",
        "ai",
        "engineer",
        "retrieval",
        "ranking",
        "recommendation",
        "embeddings",
        "vector",
        "python",
        "evaluation",
        "production",
        "machine learning",
    ),
    evidence_keywords=(
        "python",
        "retrieval",
        "ranking",
        "recommendation",
        "embeddings",
        "vector",
        "evaluation",
        "machine learning",
        "ndcg",
        "mrr",
        "map",
    ),
    must_have_keywords=(
        "python",
        "retrieval",
        "ranking",
        "embeddings",
        "evaluation",
    ),
    nice_to_have_keywords=(
        "vector",
        "recommendation",
        "ndcg",
        "mrr",
        "map",
    ),
    exclusion_keywords=(),
    min_years=5.0,
    max_years=9.0,
    target_seniority_level=3.2,
    target_seniority_label="senior",
    target_role_mode="ic",
    positive_title_terms=(
        "ai engineer",
        "senior ai engineer",
        "ml engineer",
        "machine learning engineer",
        "senior machine learning engineer",
        "junior ml engineer",
        "nlp engineer",
        "search engineer",
        "recommendation engineer",
        "ranking engineer",
        "data scientist",
        "applied scientist",
        "research engineer",
        "ai specialist",
    ),
    positive_title_hints=(
        "retrieval",
        "ranking",
        "recommendation",
        "search",
        "machine learning",
        " ml ",
        "nlp",
        "data science",
        "applied ai",
    ),
    negative_title_terms=(
        "marketing manager",
        "hr manager",
        "content writer",
        "sales executive",
        "customer support",
        "graphic designer",
        "accountant",
        "operations manager",
        "civil engineer",
        "mechanical engineer",
        "project manager",
    ),
    negative_title_hints=(
        "marketing",
        "recruiter",
        "support",
        "content",
        "sales",
        "account",
        "design",
        "hr",
    ),
    service_companies=(
        "tcs",
        "infosys",
        "wipro",
        "accenture",
        "cognizant",
        "capgemini",
        "mindtree",
        "tech mahindra",
        "hcl",
        "ltimindtree",
    ),
    service_industries=("it services", "consulting"),
    preferred_locations=(
        "noida",
        "pune",
        "mumbai",
        "delhi",
        "hyderabad",
        "bangalore",
        "bengaluru",
    ),
    preferred_countries=("india",),
    retrieval_keywords=(
        "retrieval",
        "search",
        "ranking",
        "ranker",
        "recommendation",
        "recommender",
        "matching",
        "hybrid retrieval",
        "vector search",
        "semantic search",
        "information retrieval",
        "bm25",
        "elasticsearch",
        "opensearch",
        "pinecone",
        "weaviate",
        "qdrant",
        "milvus",
        "faiss",
        "pgvector",
        "embeddings",
    ),
    vector_keywords=("pinecone", "weaviate", "qdrant", "milvus", "faiss", "pgvector", "vector search", "embeddings"),
    evaluation_keywords=("ndcg", "mrr", "map", "a/b", "ab test", "offline evaluation", "benchmark", "precision@k", "recall@k"),
    python_keywords=("python", "pytorch", "pandas", "numpy", "scikit-learn", "fastapi", "flask", "spark", "pyspark"),
    llm_keywords=("llm", "lora", "qlora", "peft", "fine-tuning", "reranker", "rag"),
    nlp_keywords=("nlp", "transformers", "sentence transformers", "language model", "information retrieval"),
    cv_keywords=("computer vision", "image classification", "object detection", "opencv", "cnn", "yolo", "diffusion"),
    speech_keywords=("speech recognition", "asr", "tts"),
    shipping_keywords=("built", "shipped", "deployed", "launched", "production", "owned", "operated", "scaled"),
    research_keywords=("research", "paper", "academic", "publication", "thesis", "lab"),
)
