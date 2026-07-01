"""Read and parse job descriptions into scoring contracts.

This file is the JD-adaptation layer. It supports plain text/Markdown/DOCX,
extracts role requirements from both clean labels and narrative sections, and
separates broad semantic query text from high-signal scoring evidence.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree

from redrob_ranker.jd_contract import DEFAULT_JD_CONTRACT, JobContract
from redrob_ranker.utils.hash_guard import compute_text_hash

DEFAULT_JOB_DESCRIPTION_TEXT = DEFAULT_JD_CONTRACT.job_query_text

KNOWN_TITLE_TERMS = tuple(
    sorted(
        set(
            DEFAULT_JD_CONTRACT.positive_title_terms
            + DEFAULT_JD_CONTRACT.negative_title_terms
            + (
                "backend engineer",
                "software engineer",
                "frontend engineer",
                "full stack developer",
                "cloud engineer",
                "devops engineer",
                "java developer",
                ".net developer",
                "mobile developer",
                "business analyst",
                "data analyst",
                "analytics engineer",
                "data engineer",
                "product manager",
            )
        ),
        key=len,
        reverse=True,
    )
)

# Known phrases are high-signal terms we trust for structured scoring. The full
# JD text is still kept for semantic search, but generic words are not allowed to
# become evidence just because they appeared in a long narrative JD.
KNOWN_JOB_PHRASES = tuple(
    sorted(
        set(
            DEFAULT_JD_CONTRACT.retrieval_keywords
            + DEFAULT_JD_CONTRACT.vector_keywords
            + DEFAULT_JD_CONTRACT.evaluation_keywords
            + DEFAULT_JD_CONTRACT.python_keywords
            + DEFAULT_JD_CONTRACT.llm_keywords
            + DEFAULT_JD_CONTRACT.nlp_keywords
            + DEFAULT_JD_CONTRACT.cv_keywords
            + DEFAULT_JD_CONTRACT.speech_keywords
            + (
                "seo",
                "content strategy",
                "distributed systems",
                "embedding drift",
                "embeddings-based retrieval",
                "evaluation frameworks",
                "hybrid search",
                "index refresh",
                "large-scale inference",
                "learning-to-rank",
                "marketplace products",
                "offline-to-online",
                "production code",
                "production deployment",
                "ranking systems",
                "recruiting tech",
                "retrieval-quality",
                "vector databases",
                "campaigns",
                "brand strategy",
                "marketing analytics",
                "sales pipeline",
                "customer support",
                "accounting",
                "financial reporting",
                "project management",
                "operations",
                "civil engineering",
                "mechanical engineering",
                "graphic design",
            )
        ),
        key=len,
        reverse=True,
    )
)

STOPWORDS = {
    "and",
    "are",
    "but",
    "for",
    "from",
    "have",
    "into",
    "need",
    "needs",
    "our",
    "role",
    "that",
    "the",
    "this",
    "with",
    "work",
    "will",
    "you",
    "your",
    "years",
    "experience",
    "candidate",
    "company",
    "team",
    "job",
    "description",
    "have",
    "has",
    "must",
    "required",
    "requirements",
    "preferred",
    "prefer",
    "nice",
    "bonus",
    "fit",
    "profiles",
    "profile",
}

GENERIC_TITLE_WORDS = {
    "analyst",
    "architect",
    "consultant",
    "developer",
    "designer",
    "director",
    "engineer",
    "executive",
    "head",
    "lead",
    "manager",
    "officer",
    "owner",
    "scientist",
    "specialist",
}

SENIORITY_WORDS = {
    "associate",
    "founding",
    "jr",
    "junior",
    "lead",
    "mid",
    "midlevel",
    "principal",
    "senior",
    "staff",
    "sr",
}

LEVEL_WORDS = {"i", "ii", "iii", "iv", "v", "l1", "l2", "l3", "l4", "l5", "level"}

COUNTRY_ALIASES = {
    "india": ("india", "indian"),
    "united states": ("united states", "usa", "u.s."),
    "canada": ("canada", "canadian"),
    "united kingdom": ("united kingdom", "uk", "u.k."),
    "germany": ("germany", "german"),
    "singapore": ("singapore",),
}

AI_TITLE_HINT_TRIGGERS = (
    " ai ",
    " ml ",
    "machine learning",
    "applied ml",
    "retrieval",
    "ranking",
    "recommendation",
    "recommender",
    "embeddings",
    "vector",
    "nlp",
    "llm",
    "rag",
    "search",
)

TITLE_LINE_PATTERNS = (
    re.compile(r"^(?:job description|job title|role|title|position)\s*[:\-]\s*(?P<title>.+)$"),
    re.compile(r"^(?:hiring|we are hiring)\s+(?:for\s+)?(?:an?\s+|the\s+)?(?P<title>.+)$"),
)

MUST_HAVE_LABELS = ("must have", "required", "requirements", "requirement")
NICE_TO_HAVE_LABELS = ("nice to have", "preferred", "bonus", "good to have")
EXCLUSION_LABELS = ("not a fit", "not fit", "exclude", "excluding", "avoid")
MUST_SECTION_MARKERS = ("things you absolutely need", "must have", "required skills", "requirements")
NICE_SECTION_MARKERS = ("things we'd like", "things we would like", "nice to have", "preferred skills")
EXCLUSION_SECTION_MARKERS = ("things we explicitly do not want", "not a fit")
SECTION_STOP_MARKERS = (
    "things you absolutely need",
    "things we'd like",
    "things we would like",
    "things we explicitly do not want",
    "on location",
    "notice period",
    "the vibe",
    "comp",
    "logistics",
)

LOW_SIGNAL_KEYWORDS = STOPWORDS | GENERIC_TITLE_WORDS | SENIORITY_WORDS | LEVEL_WORDS | {
    "about",
    "actually",
    "before",
    "below",
    "building",
    "cadence",
    "candidates",
    "career",
    "changes",
    "checklist",
    "cities",
    "differently",
    "employment",
    "flexible",
    "full-time",
    "going",
    "honest",
    "hybrid",
    "india",
    "indian",
    "intelligence",
    "kind",
    "ladder",
    "let",
    "location",
    "mean",
    "months",
    "new",
    "open",
    "platform",
    "pune",
    "raised",
    "relocation",
    "scratch",
    "see",
    "spent",
    "team",
    "tier-1",
    "type",
    "want",
    "well-scoped",
    "where",
    "write",
}

SHORT_SIGNAL_KEYWORDS = {"ai", "ml", "nlp", "llm", "rag", "mrr", "map", "dbt", "a/b"}
TECHNICAL_SINGLE_KEYWORDS = set(
    term
    for terms in (
        DEFAULT_JD_CONTRACT.retrieval_keywords,
        DEFAULT_JD_CONTRACT.vector_keywords,
        DEFAULT_JD_CONTRACT.evaluation_keywords,
        DEFAULT_JD_CONTRACT.python_keywords,
        DEFAULT_JD_CONTRACT.llm_keywords,
        DEFAULT_JD_CONTRACT.nlp_keywords,
        DEFAULT_JD_CONTRACT.cv_keywords,
        DEFAULT_JD_CONTRACT.speech_keywords,
    )
    for term in terms
    if " " not in term
) | {"campaigns", "seo", "accounting", "operations"}


def normalize_job_text(text: str) -> str:
    normalized = text.replace("\u2014", " ").replace("\u2013", " ")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def compute_job_text_hash(text: str) -> str:
    return compute_text_hash(normalize_job_text(text))


def read_job_description_text(path: str | Path | None = None) -> str:
    if path is None:
        default_path = Path("job_description.docx")
        if default_path.exists():
            path = default_path
        else:
            return DEFAULT_JOB_DESCRIPTION_TEXT

    jd_path = Path(path)
    if jd_path.suffix.lower() == ".docx":
        return _read_docx_text(jd_path)
    return jd_path.read_text(encoding="utf-8")


def _read_docx_text(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        xml_bytes = archive.read("word/document.xml")

    root = ElementTree.fromstring(xml_bytes)
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    parts = [node.text for node in root.findall(".//w:t", namespace) if node.text]
    return "\n".join(parts)


def _title_search_region(text: str) -> str:
    lines = [line.strip().lower() for line in text.splitlines() if line.strip()]
    priority_lines = [
        line
        for line in lines[:12]
        if re.search(r"^(job description|role|title|position|hiring|we are hiring)\b", line)
        or "job description:" in line
    ]
    if priority_lines:
        return " ".join(priority_lines)
    return " ".join(lines[:3])


def _clean_header_title(value: str) -> str:
    value = re.sub(r"[\u2013\u2014]", " - ", value)
    value = re.split(r"\s+-\s+|[|()/]", value, maxsplit=1)[0]
    value = re.split(r"\s+(?:at|with|for|in)\s+", value, maxsplit=1)[0]
    normalized = normalize_job_text(value).lower()
    normalized = re.sub(r"^(?:a|an|the)\s+", "", normalized)
    normalized = re.sub(r"\b(?:full[- ]time|part[- ]time|remote|hybrid|onsite|contract)\b", " ", normalized)
    normalized = normalize_job_text(normalized)

    if not normalized or normalized.startswith(("we ", "looking ", "need ", "needs ", "responsibilities", "requirements")):
        return ""

    words = [
        word
        for word in re.findall(r"[a-z][a-z0-9.+#-]*", normalized)
        if word not in {"job", "description"}
    ]
    if not words or len(words) > 6:
        return ""
    return " ".join(words)


def _header_title_candidates(text: str) -> tuple[str, ...]:
    candidates: list[str] = []
    lines = [line.strip().lower() for line in text.splitlines() if line.strip()]
    for line in lines[:12]:
        for pattern in TITLE_LINE_PATTERNS:
            match = pattern.search(line)
            if not match:
                continue
            candidate = _clean_header_title(match.group("title"))
            if candidate:
                candidates.append(candidate)
            break
    return tuple(dict.fromkeys(candidates))


def _title_variants(title: str) -> tuple[str, ...]:
    words = title.split()
    variants = [title]
    without_level = [word for word in words if word not in SENIORITY_WORDS and word not in LEVEL_WORDS]
    if without_level and without_level != words:
        variants.append(" ".join(without_level))
    return tuple(dict.fromkeys(variants))


def _extract_title_terms(text: str) -> tuple[str, ...]:
    normalized = f" {_title_search_region(text)} "
    matched = [term for term in KNOWN_TITLE_TERMS if term in normalized]
    # Header parsing lets unseen titles like "Compliance Analyst" work without
    # being manually added to the title lexicon first.
    parsed = [variant for title in _header_title_candidates(text) for variant in _title_variants(title)]
    return tuple(dict.fromkeys([*matched, *parsed]))


def _extract_job_keywords(text: str) -> tuple[str, ...]:
    normalized = text.lower()
    phrase_matches = [phrase for phrase in KNOWN_JOB_PHRASES if phrase in normalized]
    token_matches = [
        token
        for token in re.findall(r"[a-z][a-z0-9.+#-]{2,}", normalized)
        if token not in STOPWORDS and not token.isdigit()
    ]
    return tuple(dict.fromkeys([*phrase_matches, *token_matches]))


def _clean_requirement_term(value: str) -> str:
    cleaned = normalize_job_text(value.lower())
    cleaned = re.sub(r"^(?:and|or|with|in|on|using|strong|hands[- ]on|experience(?: with| in)?|profiles?)\s+", "", cleaned)
    cleaned = re.sub(r"\b(?:profiles?|candidates?|experience|required|preferred|must|have|nice|bonus)\b", " ", cleaned)
    cleaned = re.sub(r"[^a-z0-9.+# -]", " ", cleaned)
    cleaned = normalize_job_text(cleaned)
    cleaned = cleaned[:-1] if cleaned.endswith(".") else cleaned
    words = [word for word in cleaned.split() if word not in STOPWORDS]
    if not words or len(words) > 5:
        return ""
    return " ".join(words)


def _split_requirement_fragment(fragment: str) -> list[str]:
    parts = re.split(r",|;|/|\bor\b|\band\b", fragment, flags=re.IGNORECASE)
    terms = [_clean_requirement_term(part) for part in parts]
    return [term for term in terms if term]


def _extract_labeled_keywords(text: str, labels: tuple[str, ...]) -> tuple[str, ...]:
    matches: list[str] = []
    for line in text.splitlines():
        normalized_line = line.strip().lower()
        if not normalized_line:
            continue
        for label in labels:
            pattern = rf"\b{re.escape(label)}\b\s*[:\-]\s*(?P<body>.+)$"
            found = re.search(pattern, normalized_line)
            if found:
                matches.extend(_split_requirement_fragment(found.group("body")))
                break
    return tuple(dict.fromkeys(matches))


def _capture_section_text(
    text: str,
    start_markers: tuple[str, ...],
    stop_markers: tuple[str, ...] = SECTION_STOP_MARKERS,
) -> str:
    active = False
    captured: list[str] = []
    for line in text.splitlines():
        normalized_line = line.strip().lower()
        if not normalized_line:
            continue
        if any(marker in normalized_line for marker in start_markers):
            active = True
            continue
        if active and any(normalized_line.startswith(marker) for marker in stop_markers):
            break
        if active:
            captured.append(normalized_line)
    return " ".join(captured)


def _extract_section_keywords(text: str, start_markers: tuple[str, ...]) -> tuple[str, ...]:
    section_text = _capture_section_text(text, start_markers)
    if not section_text:
        return ()

    phrase_matches = [phrase for phrase in KNOWN_JOB_PHRASES if phrase in section_text]
    token_matches = [
        token
        for token in re.findall(r"[a-z][a-z0-9.+#-]{1,}", section_text)
        if _is_high_signal_keyword(token)
    ]
    return tuple(dict.fromkeys([*phrase_matches, *token_matches]))


def _is_high_signal_keyword(term: str) -> bool:
    normalized = term.strip().lower()
    if not normalized or normalized in LOW_SIGNAL_KEYWORDS:
        return False
    if normalized in SHORT_SIGNAL_KEYWORDS:
        return True
    if normalized in KNOWN_JOB_PHRASES:
        return True
    if " " in normalized:
        return False
    return normalized in TECHNICAL_SINGLE_KEYWORDS


def _extract_evidence_keywords(
    job_keywords: tuple[str, ...],
    must_have_keywords: tuple[str, ...],
    nice_to_have_keywords: tuple[str, ...],
    exclusion_keywords: tuple[str, ...] = (),
) -> tuple[str, ...]:
    # Evidence keywords are the terms used by structured career/skill scoring.
    # Keep this conservative; semantic search already sees the whole JD.
    selected = [
        *must_have_keywords,
        *nice_to_have_keywords,
        *(keyword for keyword in job_keywords if _is_high_signal_keyword(keyword)),
    ]
    exclusions = set(exclusion_keywords)
    return tuple(keyword for keyword in dict.fromkeys(selected) if keyword not in exclusions)[:80]


def _extract_experience_band(text: str) -> tuple[float, float]:
    normalized = normalize_job_text(text.lower())
    range_match = re.search(
        r"(?P<min>\d+(?:\.\d+)?)\s*(?:-|to|–|—)\s*(?P<max>\d+(?:\.\d+)?)\+?\s*(?:years|yrs)",
        normalized,
    )
    if range_match:
        return float(range_match.group("min")), float(range_match.group("max"))

    min_match = re.search(r"(?P<min>\d+(?:\.\d+)?)\+?\s*(?:years|yrs)", normalized)
    if min_match:
        return float(min_match.group("min")), 99.0
    return 0.0, 99.0


def _infer_target_seniority(
    text: str,
    positive_title_terms: tuple[str, ...],
    min_years: float,
    max_years: float,
) -> tuple[float, str, str]:
    """Infer the JD's target career level without treating tenure as destiny."""

    title_text = " ".join(positive_title_terms).lower()
    normalized = f" {normalize_job_text(text).lower()} "
    search_text = f" {title_text} {normalized} "

    if any(term in search_text for term in (" chief ", " cto ", " vp ", " vice president ")):
        level, label = 6.0, "executive"
    elif any(term in search_text for term in (" director ", " head of ", " head ")):
        level, label = 5.0, "director"
    elif " principal " in search_text:
        level, label = 4.5, "principal"
    elif " staff " in search_text:
        level, label = 4.2, "staff"
    elif " manager " in search_text or " engineering manager " in search_text:
        level, label = 4.0, "manager"
    elif " lead " in search_text:
        level, label = 3.8, "lead"
    elif any(term in search_text for term in (" senior ", " sr ", " founding ")):
        level, label = 3.2, "senior"
    elif any(term in search_text for term in (" junior ", " jr ", " associate ", " intern ", " trainee ")):
        level, label = 1.2, "junior"
    elif min_years >= 8.0:
        level, label = 4.0, "lead"
    elif min_years >= 5.0:
        level, label = 3.2, "senior"
    elif min_years >= 3.0:
        level, label = 2.5, "mid"
    else:
        level, label = 2.0, "early"

    management_terms = (" manager ", " director ", " head ", " vp ", " vice president ", " cto ", " chief ")
    hands_on_terms = (
        " hands-on ",
        " hands on ",
        " build ",
        " building ",
        " code ",
        " coding ",
        " engineer ",
        " developer ",
        " individual contributor ",
        " ic ",
    )
    if any(term in search_text for term in management_terms) and not any(term in search_text for term in hands_on_terms):
        role_mode = "management"
    elif any(term in search_text for term in management_terms) and any(term in search_text for term in hands_on_terms):
        role_mode = "hybrid"
    else:
        role_mode = "ic"

    if max_years < min_years:
        max_years = min_years
    return level, label, role_mode


def _mentions_any(text: str, terms: tuple[str, ...]) -> bool:
    padded = f" {text.lower()} "
    return any(term in padded for term in terms)


def _positive_title_hints(
    positive_title_terms: tuple[str, ...],
    query_text: str,
    *,
    is_default_contract: bool,
) -> tuple[str, ...]:
    hints: set[str] = set()
    if is_default_contract or _mentions_any(query_text, AI_TITLE_HINT_TRIGGERS):
        hints.update(DEFAULT_JD_CONTRACT.positive_title_hints)
    for title in positive_title_terms:
        hints.update(
            part
            for part in title.split()
            if len(part) > 2 and part not in GENERIC_TITLE_WORDS and part not in SENIORITY_WORDS
        )
    return tuple(sorted(hints))


def _extract_preferred_countries(query_text: str) -> tuple[str, ...]:
    countries = []
    for country, aliases in COUNTRY_ALIASES.items():
        if any(re.search(rf"\b{re.escape(alias)}\b", query_text) for alias in aliases):
            countries.append(country)
    return tuple(countries)


def build_job_contract(job_description_text: str | None = None) -> JobContract:
    """Parse raw JD text into the contract consumed by every feature module."""

    is_default_contract = job_description_text is None or not job_description_text.strip()
    raw_text = DEFAULT_JOB_DESCRIPTION_TEXT if is_default_contract else job_description_text
    query_text = normalize_job_text(raw_text).lower()
    extracted_titles = _extract_title_terms(raw_text)
    positive_title_terms = extracted_titles or (DEFAULT_JD_CONTRACT.positive_title_terms if is_default_contract else ())
    job_keywords = _extract_job_keywords(query_text)
    must_have_keywords = (
        _extract_labeled_keywords(raw_text, MUST_HAVE_LABELS)
        or _extract_section_keywords(raw_text, MUST_SECTION_MARKERS)
        or (DEFAULT_JD_CONTRACT.must_have_keywords if is_default_contract else ())
    )
    nice_to_have_keywords = (
        _extract_labeled_keywords(raw_text, NICE_TO_HAVE_LABELS)
        or _extract_section_keywords(raw_text, NICE_SECTION_MARKERS)
        or (DEFAULT_JD_CONTRACT.nice_to_have_keywords if is_default_contract else ())
    )
    exclusion_keywords = _extract_labeled_keywords(raw_text, EXCLUSION_LABELS) or _extract_section_keywords(
        raw_text, EXCLUSION_SECTION_MARKERS
    )
    exclusion_keywords = tuple(
        keyword
        for keyword in exclusion_keywords
        if keyword not in {"nlp", "retrieval", "ranking", "search", "information retrieval", "ir"}
    )
    evidence_keywords = (
        _extract_evidence_keywords(job_keywords, must_have_keywords, nice_to_have_keywords, exclusion_keywords)
        or (DEFAULT_JD_CONTRACT.evidence_keywords if is_default_contract else ())
    )
    min_years, max_years = _extract_experience_band(raw_text)
    if is_default_contract:
        min_years = DEFAULT_JD_CONTRACT.min_years
        max_years = DEFAULT_JD_CONTRACT.max_years
    target_seniority_level, target_seniority_label, target_role_mode = _infer_target_seniority(
        raw_text,
        positive_title_terms,
        min_years,
        max_years,
    )
    if is_default_contract:
        target_seniority_level = DEFAULT_JD_CONTRACT.target_seniority_level
        target_seniority_label = DEFAULT_JD_CONTRACT.target_seniority_label
        target_role_mode = DEFAULT_JD_CONTRACT.target_role_mode

    negative_title_terms = tuple(
        term
        for term in dict.fromkeys([*DEFAULT_JD_CONTRACT.negative_title_terms, *exclusion_keywords])
        if term not in positive_title_terms and not any(term in positive or positive in term for positive in positive_title_terms)
    )
    negative_title_hints = tuple(
        hint
        for hint in DEFAULT_JD_CONTRACT.negative_title_hints
        if not any(hint in positive for positive in positive_title_terms)
    )

    def selected(terms: tuple[str, ...]) -> tuple[str, ...]:
        matches = tuple(term for term in terms if term in query_text)
        return matches

    return JobContract(
        job_query_text=query_text,
        job_keywords=job_keywords,
        evidence_keywords=evidence_keywords,
        must_have_keywords=must_have_keywords,
        nice_to_have_keywords=nice_to_have_keywords,
        exclusion_keywords=exclusion_keywords,
        min_years=min_years,
        max_years=max_years,
        target_seniority_level=target_seniority_level,
        target_seniority_label=target_seniority_label,
        target_role_mode=target_role_mode,
        positive_title_terms=positive_title_terms,
        positive_title_hints=_positive_title_hints(
            positive_title_terms,
            query_text,
            is_default_contract=is_default_contract,
        ),
        negative_title_terms=negative_title_terms,
        negative_title_hints=negative_title_hints,
        service_companies=DEFAULT_JD_CONTRACT.service_companies,
        service_industries=DEFAULT_JD_CONTRACT.service_industries,
        preferred_locations=(
            tuple(city for city in DEFAULT_JD_CONTRACT.preferred_locations if city in query_text)
            or (DEFAULT_JD_CONTRACT.preferred_locations if is_default_contract else ())
        ),
        preferred_countries=(
            _extract_preferred_countries(query_text)
            or (DEFAULT_JD_CONTRACT.preferred_countries if is_default_contract else ())
        ),
        retrieval_keywords=selected(DEFAULT_JD_CONTRACT.retrieval_keywords),
        vector_keywords=selected(DEFAULT_JD_CONTRACT.vector_keywords),
        evaluation_keywords=selected(DEFAULT_JD_CONTRACT.evaluation_keywords),
        python_keywords=selected(DEFAULT_JD_CONTRACT.python_keywords)
        or (DEFAULT_JD_CONTRACT.python_keywords if is_default_contract else ()),
        llm_keywords=selected(DEFAULT_JD_CONTRACT.llm_keywords),
        nlp_keywords=selected(DEFAULT_JD_CONTRACT.nlp_keywords),
        cv_keywords=selected(DEFAULT_JD_CONTRACT.cv_keywords),
        speech_keywords=selected(DEFAULT_JD_CONTRACT.speech_keywords),
        shipping_keywords=DEFAULT_JD_CONTRACT.shipping_keywords,
        research_keywords=DEFAULT_JD_CONTRACT.research_keywords,
    )
