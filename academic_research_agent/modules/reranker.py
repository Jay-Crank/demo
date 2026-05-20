"""Reranker — TF-IDF relevance, freshness, citation, source, quality, composite scoring.

Composite formula with configurable weights:
  Score = w_rel*R + w_fresh*T + w_cite*C + w_src*S + w_qual*Q

Default weights (user customisable):
  w_rel=0.40  w_fresh=0.20  w_cite=0.20  w_src=0.10  w_qual=0.10
"""

import math
from datetime import datetime, timezone

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Preference keyword sets (retained for backward compat) ─────────────

_PREFERENCE_KEYWORDS = {
    "survey": [
        "survey", "review", "overview", "comprehensive", "systematic",
        "state of the art", "literature review", "meta-analysis",
        "taxonomy", "bibliometric",
    ],
    "method": [
        "novel method", "new method", "framework", "algorithm",
        "architecture", "propose", "introduce", "approach",
        "technique", "design", "paradigm",
    ],
    "application": [
        "application", "system", "dataset", "benchmark",
        "real-world", "deploy", "practical", "implementation",
        "tool", "platform", "empirical study", "case study",
    ],
}

_PREFERENCE_MAP = {
    "关注综述论文": "survey",
    "关注方法创新": "method",
    "关注应用系统": "application",
}


def _build_text(paper: dict) -> str:
    title = paper.get("title", "") or ""
    abstract = paper.get("abstract", "") or paper.get("summary", "") or ""
    return f"{title} {abstract}"


def _parse_published(p) -> datetime | None:
    import pandas as pd

    if p is None:
        return None
    if isinstance(p, datetime):
        return p
    if isinstance(p, str):
        try:
            return pd.Timestamp(p).to_pydatetime()
        except Exception:
            return None
    return None


# ── Individual score components ───────────────────────────────────────

def compute_relevance(
    papers: list[dict],
    query: str,
) -> np.ndarray:
    """TF-IDF cosine similarity between each paper (title+abstract) and query.

    Returns shape (len(papers),) in [0, 1].
    """
    if not papers:
        return np.array([])

    paper_texts = [_build_text(p) for p in papers]
    if all(not t.strip() for t in paper_texts):
        return np.zeros(len(papers))

    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    try:
        tfidf_matrix = vectorizer.fit_transform(paper_texts + [query])
    except ValueError:
        return np.zeros(len(papers))

    paper_vecs = tfidf_matrix[:-1]
    query_vec = tfidf_matrix[-1]
    sims = cosine_similarity(paper_vecs, query_vec).flatten()
    return np.clip(sims, 0.0, 1.0)


def compute_freshness(
    papers: list[dict],
    current_date: datetime | None = None,
    decay_days: float = 30.0,
) -> np.ndarray:
    """Freshness score: exp(-delta_t / decay_days).  Newer → closer to 1."""
    if current_date is None:
        current_date = datetime.now(timezone.utc)

    scores = []
    for p in papers:
        pub = _parse_published(p.get("published") or p.get("published_date"))
        if pub is None:
            scores.append(0.5)
            continue
        if pub.tzinfo is None:
            pub = pub.replace(tzinfo=timezone.utc)
        delta_t = (current_date - pub).days
        delta_t = max(delta_t, 0)
        scores.append(math.exp(-delta_t / decay_days))
    return np.array(scores)


def compute_citation_score(papers: list[dict]) -> np.ndarray:
    """Log-normalised citation score in [0, 1].

    score = log(1 + citation_count) / log(1 + max_citation_count)

    If every paper has citation_count == 0 the array is all zeros.
    """
    if not papers:
        return np.array([])

    raw = np.array([float(p.get("citation_count", 0) or 0) for p in papers])
    max_cite = raw.max()
    if max_cite <= 0:
        return np.zeros(len(papers))

    scores = np.log1p(raw) / math.log1p(max_cite)
    return np.clip(scores, 0.0, 1.0)


def compute_source_score(papers: list[dict]) -> np.ndarray:
    """Source-diversity score based on how many data sources returned the paper.

    Each paper dict may carry:
      - ``sources`` : list[str]  — all source names
      - ``source``  : str        — primary source

    Score = min(1.0, count / 3)   — 1 source=0.33, 2=0.67, 3=1.0
    """
    if not papers:
        return np.array([])

    scores = []
    for p in papers:
        src_list = p.get("sources") or []
        if not src_list:
            src = p.get("source", "")
            n_src = 1 if src else 0
        else:
            n_src = len(src_list)
        scores.append(min(1.0, n_src / 3.0))
    return np.array(scores)


def compute_quality(papers: list[dict]) -> np.ndarray:
    """Heuristic quality score (0–1) based on metadata completeness.

    - Has non-empty abstract → +0.3
    - Has DOI              → +0.25
    - Has venue            → +0.25
    - Has url              → +0.2
    """
    if not papers:
        return np.array([])

    scores = []
    for p in papers:
        s = 0.0
        abstract = p.get("abstract") or p.get("summary") or ""
        if abstract.strip():
            s += 0.3
        if p.get("doi", "").strip():
            s += 0.25
        if p.get("venue", "").strip():
            s += 0.25
        url = p.get("url") or p.get("link") or ""
        if url.strip():
            s += 0.2
        scores.append(s)
    return np.array(scores)


def compute_preference_score(
    papers: list[dict],
    selected_preferences: list[str],
) -> np.ndarray:
    """Keyword-based preference score (retained for personalised ranking)."""
    if not papers or not selected_preferences:
        return np.zeros(len(papers))

    scores = np.zeros(len(papers))
    for i, p in enumerate(papers):
        title = p.get("title", "") or ""
        summary = p.get("summary", "") or p.get("abstract", "") or ""
        text = (title + " " + summary).lower()
        cat_scores = []
        for pref_key in selected_preferences:
            keywords = _PREFERENCE_KEYWORDS.get(pref_key, [])
            if not keywords:
                continue
            hits = sum(1 for kw in keywords if kw.lower() in text)
            cat_scores.append(hits / len(keywords))
        scores[i] = max(cat_scores) if cat_scores else 0.0

    max_s = scores.max()
    if max_s > 0:
        scores = scores / max_s
    return scores


# ── Composite scoring ─────────────────────────────────────────────────

def compute_composite(
    papers: list[dict],
    query: str,
    current_date: datetime | None = None,
    w_rel: float = 0.40,
    w_fresh: float = 0.20,
    w_cite: float = 0.20,
    w_src: float = 0.10,
    w_qual: float = 0.10,
    selected_preferences: list[str] | None = None,
) -> list[dict]:
    """Compute composite score and attach all sub-scores.

    Score = w_rel*R + w_fresh*T + w_cite*C + w_src*S + w_qual*Q

    Papers are sorted by composite score descending.

    Each returned paper dict gains:
      relevance_score, freshness_score, citation_score,
      source_score, quality_score, composite_score
    """
    if not papers:
        return []

    rel = compute_relevance(papers, query)
    fresh = compute_freshness(papers, current_date)
    cite = compute_citation_score(papers)
    src = compute_source_score(papers)
    qual = compute_quality(papers)

    composite = w_rel * rel + w_fresh * fresh + w_cite * cite + w_src * src + w_qual * qual

    scored = []
    for i, p in enumerate(papers):
        p = dict(p)
        p["relevance_score"] = round(float(rel[i]), 4)
        p["freshness_score"] = round(float(fresh[i]), 4)
        p["citation_score"] = round(float(cite[i]), 4)
        p["source_score"] = round(float(src[i]), 4)
        p["quality_score"] = round(float(qual[i]), 4)
        p["composite_score"] = round(float(composite[i]), 4)
        scored.append(p)

    scored.sort(key=lambda x: x["composite_score"], reverse=True)
    return scored


# ── Utility ───────────────────────────────────────────────────────────

def get_preference_keys(selected_labels: list[str]) -> list[str]:
    return [_PREFERENCE_MAP[label] for label in selected_labels if label in _PREFERENCE_MAP]


def get_default_weights() -> dict:
    return {
        "w_rel": 0.40,
        "w_fresh": 0.20,
        "w_cite": 0.20,
        "w_src": 0.10,
        "w_qual": 0.10,
    }


def normalize_weights(
    w_rel: float,
    w_fresh: float,
    w_cite: float = 0.0,
    w_src: float = 0.0,
    w_qual: float = 0.0,
) -> tuple:
    """Normalize five weights so they sum to 1.0."""
    total = w_rel + w_fresh + w_cite + w_src + w_qual
    if total == 0:
        d = get_default_weights()
        return (d["w_rel"], d["w_fresh"], d["w_cite"], d["w_src"], d["w_qual"])
    return (
        w_rel / total,
        w_fresh / total,
        w_cite / total,
        w_src / total,
        w_qual / total,
    )
