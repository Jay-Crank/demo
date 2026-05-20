"""arXiv API client — search papers and return standardized paper dicts."""

import time
import arxiv
import pandas as pd
from datetime import datetime, timezone
from typing import Optional


MAX_RESULTS_PER_QUERY = 50
CACHE_PATH = "data/cache.csv"

_client = arxiv.Client(
    delay_seconds=5.0,
    num_retries=3,
)

# Global error state that callers can inspect
_last_error: str | None = None
_last_api_error: bool = False


def get_last_error() -> str | None:
    """Return the last error message, or None if last call succeeded."""
    return _last_error


def had_api_error() -> bool:
    """Return True if the most recent search call hit an API/network error."""
    return _last_api_error


def _paper_to_dict(paper: arxiv.Result) -> dict:
    """Convert arxiv Result to a safe dict with all required fields."""
    title = (paper.title or "").strip()
    summary = (paper.summary or "").strip().replace("\n", " ")
    authors = ", ".join(a.name for a in (paper.authors or [])) if paper.authors else "Unknown"
    arxiv_id = (paper.entry_id or "unknown").split("/")[-1]
    return {
        "title": title or "Untitled",
        "authors": authors or "Unknown",
        "summary": summary or "No abstract available.",
        "published": paper.published,
        "updated": paper.updated,
        "arxiv_id": arxiv_id,
        "link": paper.entry_id or "",
        "pdf_url": paper.pdf_url or "",
    }


def search_papers(
    query: str,
    max_results: int = 30,
    sort_by: str = "relevance",
) -> list[dict]:
    """Search arXiv and return a list of standardized paper dicts.

    Sets global _last_error / _last_api_error so callers can check
    whether the call succeeded or failed.
    """
    global _last_error, _last_api_error
    _last_error = None
    _last_api_error = False

    if not query or not query.strip():
        _last_error = "Query is empty."
        return []

    sort = (
        arxiv.SortCriterion.Relevance
        if sort_by == "relevance"
        else arxiv.SortCriterion.SubmittedDate
    )
    search = arxiv.Search(
        query=query.strip(),
        max_results=min(max_results, MAX_RESULTS_PER_QUERY),
        sort_by=sort,
    )
    papers = []
    try:
        for result in _client.results(search):
            papers.append(_paper_to_dict(result))
    except arxiv.ArxivError as e:
        _last_error = f"arXiv API error: {e}"
        _last_api_error = True
    except Exception as e:
        _last_error = f"Unexpected error: {e}"
        _last_api_error = True

    if not papers and not _last_error:
        _last_error = f"No papers found for query '{query}'."

    return papers


def search_multi_keywords(
    keywords: list[str],
    papers_per_keyword: int = 15,
) -> list[dict]:
    """Search with multiple keyword strings, merging results."""
    seen_ids = set()
    all_papers = []
    error_count = 0

    for kw in keywords:
        if not kw or not kw.strip():
            continue
        papers = search_papers(kw.strip(), max_results=papers_per_keyword)
        if had_api_error():
            error_count += 1
        for p in papers:
            if p["arxiv_id"] not in seen_ids:
                seen_ids.add(p["arxiv_id"])
                all_papers.append(p)
        # Rate-limit courtesy delay between keyword searches
        time.sleep(1.0)

    if not all_papers and error_count >= len(keywords):
        global _last_error
        _last_error = "All keyword queries failed. arXiv may be rate-limiting or unreachable."
        _last_api_error = True

    return all_papers


def load_cache() -> pd.DataFrame:
    """Load cached papers from CSV if available."""
    import os

    if os.path.exists(CACHE_PATH):
        try:
            return pd.read_csv(CACHE_PATH, parse_dates=["published", "updated"])
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def save_cache(df: pd.DataFrame) -> None:
    """Save papers DataFrame to CSV cache."""
    import os

    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    df.to_csv(CACHE_PATH, index=False)
