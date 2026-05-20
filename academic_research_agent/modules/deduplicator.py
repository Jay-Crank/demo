"""Paper deduplication — DOI, arXiv ID, and title-similarity based merging.

Dedup rules (applied in order):
1. Same DOI → merge
2. Same arXiv ID → merge
3. Normalized-title similarity >= 0.92 → merge

On merge, fields are combined via ``paper_schema.merge_papers`` which
keeps the longer abstract, max citation counts, and unions source lists.
"""

import re
from difflib import SequenceMatcher

from modules.paper_schema import merge_papers


def _normalize_title(title: str) -> str:
    """Lowercase, remove punctuation, collapse whitespace."""
    if not title:
        return ""
    t = title.lower()
    t = re.sub(r"[^a-z0-9\s]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def title_similarity(a: str, b: str) -> float:
    """Compute similarity ratio between two normalized titles."""
    na = _normalize_title(a or "")
    nb = _normalize_title(b or "")
    if not na or not nb:
        return 0.0
    return SequenceMatcher(None, na, nb).ratio()


def deduplicate(
    papers: list[dict],
    threshold: float = 0.92,
) -> list[dict]:
    """Deduplicate papers by DOI, arXiv ID, and title similarity.

    Strategy: greedy — for each incoming paper, check if it matches any
    already-kept paper.  If a match is found, merge the two.  Otherwise
    append.

    Parameters
    ----------
    papers : list[dict]
        Paper dicts conforming to ``paper_schema``.
    threshold : float
        Title similarity threshold (0.0–1.0).  Default 0.92.

    Returns
    -------
    list[dict] — deduplicated (and merged) paper list.
    """
    kept: list[dict] = []

    for paper in papers:
        merged_idx = _find_duplicate(paper, kept, threshold)
        if merged_idx >= 0:
            # Merge into existing
            kept[merged_idx] = merge_papers(kept[merged_idx], paper)
        else:
            kept.append(paper)

    return kept


def _find_duplicate(
    paper: dict,
    kept: list[dict],
    threshold: float,
) -> int:
    """Return the index of a duplicate in *kept*, or -1 if none match."""
    doi = paper.get("doi", "")
    arxiv_id = paper.get("arxiv_id", "")
    title = paper.get("title", "")

    for i, k in enumerate(kept):
        # ── Rule 1: DOI match ──
        if doi and k.get("doi") and doi.lower() == k["doi"].lower():
            return i

        # ── Rule 2: arXiv ID match ──
        if arxiv_id and k.get("arxiv_id") and arxiv_id == k["arxiv_id"]:
            return i

        # ── Rule 3: Title similarity ──
        k_title = k.get("title", "")
        if title and k_title:
            sim = title_similarity(title, k_title)
            if sim >= threshold:
                return i

    return -1
