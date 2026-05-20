"""Unified paper data schema — single source of truth for paper fields.

All source clients and downstream modules should produce/consume dicts
that conform to this schema.  The normalizer fills in defaults and adds
backward-compatible aliases so old code referencing ``summary``,
``link``, or ``published`` continues to work.
"""

from datetime import datetime


# ── Field definitions ──────────────────────────────────────────────────

PAPER_FIELDS: dict[str, object] = {
    "paper_id": "",
    "title": "",
    "authors": "",
    "abstract": "",
    "year": None,
    "published_date": None,
    "doi": "",
    "arxiv_id": "",
    "source": "",
    "source_id": "",
    "url": "",
    "pdf_url": "",
    "venue": "",
    "publisher": "",
    "citation_count": 0,
    "influential_citation_count": 0,
    "fields_of_study": [],
    "keywords": [],
    "is_open_access": False,
    "sources": [],
    "raw_data": {},
}


def normalize_paper(raw: dict) -> dict:
    """Normalize a raw paper dict so every standard field is present.

    Also populates backward-compatible aliases:
      ``summary``   → mirrors ``abstract``
      ``link``      → mirrors ``url``
      ``published`` → mirrors ``published_date``

    Parameters
    ----------
    raw : dict
        A paper dict from any source.  May use old or new field names.

    Returns
    -------
    dict
        A paper dict with all PAPER_FIELDS keys present.
    """
    # ── Core fields ──
    title = raw.get("title") or ""
    authors = raw.get("authors") or ""
    abstract = raw.get("abstract") or raw.get("summary") or ""
    pub_date = _coerce_date(raw.get("published_date") or raw.get("published"))
    year = raw.get("year") or (pub_date.year if pub_date else None)

    sources = raw.get("sources") or []
    source = raw.get("source") or ""
    if source and source not in sources:
        sources = list(dict.fromkeys([source] + sources))  # unique, order-preserving

    paper = {
        # ── New canonical names ──
        "paper_id": raw.get("paper_id") or raw.get("arxiv_id") or raw.get("source_id") or "",
        "title": title.strip() if title else "Untitled",
        "authors": authors.strip() if authors else "Unknown",
        "abstract": abstract.strip().replace("\n", " "),
        "year": year,
        "published_date": pub_date,
        "doi": raw.get("doi") or "",
        "arxiv_id": raw.get("arxiv_id") or "",
        "source": sources[0] if sources else source,
        "source_id": raw.get("source_id") or "",
        "url": raw.get("url") or raw.get("link") or "",
        "pdf_url": raw.get("pdf_url") or "",
        "venue": raw.get("venue") or "",
        "publisher": raw.get("publisher") or "",
        "citation_count": int(raw.get("citation_count") or 0),
        "influential_citation_count": int(raw.get("influential_citation_count") or 0),
        "fields_of_study": raw.get("fields_of_study") or [],
        "keywords": raw.get("keywords") or [],
        "is_open_access": bool(raw.get("is_open_access") or False),
        "sources": sources,
        "raw_data": raw.get("raw_data") or {},
    }

    # ── Backward-compatible aliases ──
    paper["summary"] = paper["abstract"]
    paper["link"] = paper["url"]
    paper["published"] = paper["published_date"]

    return paper


def merge_papers(kept: dict, incoming: dict) -> dict:
    """Merge two paper dicts representing the same work.

    Rules:
    - ``citation_count``: keep the max.
    - ``influential_citation_count``: keep the max.
    - ``abstract``: keep the longer one.
    - ``sources``: union (unique, order-preserving).
    - ``source``: set to the first (primary) source.
    - Other scalar fields: prefer non-empty value from *incoming* if
      *kept* is empty.

    Parameters
    ----------
    kept : dict
        The paper currently retained.
    incoming : dict
        The paper being merged in.

    Returns
    -------
    dict — merged paper.
    """
    merged = dict(kept)

    # citation counts → max
    merged["citation_count"] = max(kept.get("citation_count", 0), incoming.get("citation_count", 0))
    merged["influential_citation_count"] = max(
        kept.get("influential_citation_count", 0),
        incoming.get("influential_citation_count", 0),
    )

    # abstract → longest
    kept_abs = kept.get("abstract", "") or kept.get("summary", "") or ""
    inc_abs = incoming.get("abstract", "") or incoming.get("summary", "") or ""
    if len(inc_abs) > len(kept_abs):
        merged["abstract"] = inc_abs
        merged["summary"] = inc_abs

    # sources → union
    merged_sources = list(dict.fromkeys(
        (kept.get("sources") or []) + (incoming.get("sources") or [])
    ))
    merged["sources"] = merged_sources
    merged["source"] = merged_sources[0] if merged_sources else ""

    # Fill empty fields from incoming
    for key in ("doi", "arxiv_id", "url", "link", "pdf_url", "venue", "publisher",
                 "source_id", "paper_id"):
        if not merged.get(key):
            merged[key] = incoming.get(key) or ""

    for key in ("year", "published_date", "published"):
        if not merged.get(key):
            merged[key] = incoming.get(key)

    for key in ("fields_of_study", "keywords"):
        if not merged.get(key):
            merged[key] = incoming.get(key) or []

    if not merged.get("is_open_access"):
        merged["is_open_access"] = bool(incoming.get("is_open_access"))

    # Update raw_data
    if incoming.get("raw_data"):
        merged["raw_data"] = {**merged.get("raw_data", {}), **incoming.get("raw_data", {})}

    return merged


# ── Internal helpers ───────────────────────────────────────────────────

def _coerce_date(val) -> datetime | None:
    """Try to convert *val* to a datetime.  Returns None on failure."""
    if val is None:
        return None
    if isinstance(val, datetime):
        return val
    if isinstance(val, str) and val.strip():
        for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ",
                     "%Y/%m/%d", "%Y", "%Y-%m"):
            try:
                return datetime.strptime(val.strip()[:19], fmt)
            except ValueError:
                continue
    return None
