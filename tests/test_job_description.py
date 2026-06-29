from __future__ import annotations

from redrob_ranker.job_description import build_job_contract, compute_job_text_hash, read_job_description_text


def test_read_job_description_text_reads_plain_text_file(tmp_path):
    jd_path = tmp_path / "marketing_jd.txt"
    jd_path.write_text("Job Description: Marketing Manager\nOwn SEO, campaigns, and brand strategy.", encoding="utf-8")

    text = read_job_description_text(jd_path)

    assert "Marketing Manager" in text
    assert "brand strategy" in text


def test_build_job_contract_uses_the_given_role_instead_of_static_ai_terms():
    text = "Job Description: Marketing Manager\nNeed SEO, content strategy, campaigns, and brand leadership."

    contract = build_job_contract(text)

    assert "marketing manager" in contract.positive_title_terms
    assert "marketing manager" not in contract.negative_title_terms
    assert "seo" in contract.job_keywords
    assert "content strategy" in contract.job_query_text


def test_build_job_contract_extracts_unseen_title_without_ai_fallback():
    text = "Role: Compliance Analyst\nOwn regulatory audits, risk controls, policy reviews, and reporting."

    contract = build_job_contract(text)

    assert "compliance analyst" in contract.positive_title_terms
    assert "ai engineer" not in contract.positive_title_terms
    assert contract.python_keywords == ()
    assert "regulatory" in contract.job_keywords


def test_build_job_contract_extracts_requirements_and_experience_band():
    text = """
    Role: Senior Data Engineer
    Requirements:
    Must have: Python, Spark, Airflow, data pipelines.
    Nice to have: dbt, Snowflake, Kafka.
    Looking for 5-8 years of experience.
    Not a fit: sales executive or content writer profiles.
    """

    contract = build_job_contract(text)

    assert {"python", "spark", "airflow", "data pipelines"} <= set(contract.must_have_keywords)
    assert {"dbt", "snowflake", "kafka"} <= set(contract.nice_to_have_keywords)
    assert contract.min_years == 5.0
    assert contract.max_years == 8.0
    assert "sales executive" in contract.exclusion_keywords


def test_build_job_contract_extracts_narrative_requirement_sections():
    text = """
    Role: Senior AI Engineer
    Things you absolutely need
    Production experience with embeddings-based retrieval systems.
    Strong Python.
    Hands-on experience designing evaluation frameworks for ranking systems.
    Things we'd like you to have but won't reject you for
    LLM fine-tuning experience and learning-to-rank models.
    Things we explicitly do NOT want
    People whose primary expertise is computer vision without significant NLP/IR exposure.
    """

    contract = build_job_contract(text)

    assert {"embeddings", "retrieval", "python", "evaluation frameworks", "ranking systems"} <= set(
        contract.must_have_keywords
    )
    assert {"fine-tuning", "learning-to-rank"} <= set(contract.nice_to_have_keywords)
    assert "computer vision" in contract.exclusion_keywords
    assert "redrob" not in contract.evidence_keywords


def test_build_job_contract_extracts_preferred_country_from_location_text():
    text = """
    Role: Senior AI Engineer
    Location: Pune/Noida, India. Candidates in Hyderabad, Mumbai, Delhi NCR welcome.
    Outside India case-by-case; no visa sponsorship.
    """

    contract = build_job_contract(text)

    assert "india" in contract.preferred_countries


def test_job_text_hash_changes_with_jd_text():
    first = compute_job_text_hash("Job Description: Marketing Manager")
    second = compute_job_text_hash("Job Description: Senior AI Engineer")

    assert first != second
