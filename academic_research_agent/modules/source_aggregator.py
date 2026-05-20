"""Multi-source aggregator — parallel/serial search across data sources.

Orchestrates queries to multiple sources, merges results, deduplicates,
and returns a unified, deduplicated paper list.
"""

import concurrent.futures
from typing import Protocol

from modules.deduplicator import deduplicate as dedup_fn
from modules.sources.base import BaseSource


MAX_WORKERS = 3


class SourceLike(Protocol):
    """Minimal protocol for a data source — duck-typing, not inheritance."""

    name: str

    def search(self, query: str, max_results: int = 20, **kwargs) -> list[dict]:
        ...

    @property
    def last_error(self) -> str | None:
        ...

    @property
    def had_error(self) -> bool:
        ...


def aggregate_search(
    sources: list[SourceLike],
    query: str,
    max_results: int = 20,
    parallel: bool = True,
    **source_kwargs,
) -> dict:
    """Search across multiple sources, merge, and deduplicate.

    Parameters
    ----------
    sources : list
        Source instances (each with ``name``, ``search()``, ``last_error``, ``had_error``).
    query : str
        Search query string.
    max_results : int
        Max results requested *per source*.
    parallel : bool
        If True, query all sources concurrently via ThreadPoolExecutor.
    **source_kwargs
        Extra keyword arguments forwarded to each source's ``search()``.

    Returns
    -------
    dict with keys:
      papers : list[dict]
          Unified, deduplicated paper list.
      source_stats : dict
          {source_name: {"count": n, "error": str|None}, ...}
      total_raw : int
          Total papers before dedup.
      total_deduped : int
          Papers after dedup.
    """
    if not sources:
        return {"papers": [], "source_stats": {}, "total_raw": 0, "total_deduped": 0}

    # ── Query each source ──
    if parallel and len(sources) > 1:
        raw_by_source = _search_parallel(sources, query, max_results, source_kwargs)
    else:
        raw_by_source = _search_serial(sources, query, max_results, source_kwargs)

    # ── Gather stats ──
    source_stats: dict[str, dict] = {}
    for src in sources:
        papers = raw_by_source.get(src.name, [])
        err = src.last_error if src.had_error else None
        source_stats[src.name] = {"count": len(papers), "error": err}

    # ── Merge all papers ──
    all_papers: list[dict] = []
    for papers in raw_by_source.values():
        all_papers.extend(papers)

    total_raw = len(all_papers)

    # ── Deduplicate ──
    if all_papers:
        deduped = dedup_fn(all_papers, threshold=0.92)
    else:
        deduped = []

    return {
        "papers": deduped,
        "source_stats": source_stats,
        "total_raw": total_raw,
        "total_deduped": len(deduped),
    }


# ── Internal ───────────────────────────────────────────────────────────

def _search_serial(
    sources: list[SourceLike],
    query: str,
    max_results: int,
    kwargs: dict,
) -> dict[str, list[dict]]:
    """Query sources one at a time."""
    result: dict[str, list[dict]] = {}
    for src in sources:
        try:
            papers = src.search(query, max_results=max_results, **kwargs)
        except Exception:
            papers = []
        result[src.name] = papers or []
    return result


def _search_parallel(
    sources: list[SourceLike],
    query: str,
    max_results: int,
    kwargs: dict,
) -> dict[str, list[dict]]:
    """Query sources concurrently."""
    result: dict[str, list[dict]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(sources), MAX_WORKERS)) as ex:
        futures = {}
        for src in sources:
            fut = ex.submit(_safe_search, src, query, max_results, kwargs)
            futures[fut] = src.name

        for fut in concurrent.futures.as_completed(futures):
            name = futures[fut]
            try:
                papers = fut.result(timeout=60)
            except Exception:
                papers = []
            result[name] = papers or []
    return result


def _safe_search(
    src: SourceLike,
    query: str,
    max_results: int,
    kwargs: dict,
) -> list[dict]:
    """Call src.search() and return [] on any exception."""
    try:
        return src.search(query, max_results=max_results, **kwargs) or []
    except Exception:
        return []
