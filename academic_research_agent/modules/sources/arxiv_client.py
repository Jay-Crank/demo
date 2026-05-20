"""arXiv source client — wraps the arXiv API."""

import time
import arxiv

from modules.paper_schema import normalize_paper
from modules.sources.base import BaseSource


MAX_RESULTS_PER_QUERY = 50


class ArxivSource(BaseSource):
    """arXiv API data source."""

    def __init__(self, delay_seconds: float = 5.0, num_retries: int = 3):
        self._client = arxiv.Client(
            delay_seconds=delay_seconds,
            num_retries=num_retries,
        )
        self._last_error: str | None = None
        self._had_api_error: bool = False

    @property
    def name(self) -> str:
        return "arxiv"

    @property
    def last_error(self) -> str | None:
        return self._last_error

    @property
    def had_error(self) -> bool:
        return self._had_api_error

    def search(
        self,
        query: str,
        max_results: int = 20,
        **kwargs,
    ) -> list[dict]:
        self._last_error = None
        self._had_api_error = False

        if not query or not query.strip():
            self._last_error = "Query is empty."
            return []

        sort_by = kwargs.get("sort_by", "relevance")
        sort_criterion = (
            arxiv.SortCriterion.Relevance
            if sort_by == "relevance"
            else arxiv.SortCriterion.SubmittedDate
        )

        search = arxiv.Search(
            query=query.strip(),
            max_results=min(max_results, MAX_RESULTS_PER_QUERY),
            sort_by=sort_criterion,
        )

        papers = []
        try:
            for result in self._client.results(search):
                papers.append(self._to_dict(result))
        except arxiv.ArxivError as e:
            self._last_error = f"arXiv API error: {e}"
            self._had_api_error = True
        except Exception as e:
            self._last_error = f"arXiv unexpected error: {e}"
            self._had_api_error = True

        if not papers and not self._last_error:
            self._last_error = f"No papers found for query '{query}'."

        return papers

    def search_multi_keywords(
        self,
        keywords: list[str],
        papers_per_keyword: int = 15,
    ) -> list[dict]:
        """Search with multiple keywords, merging results."""
        seen_ids: set[str] = set()
        all_papers: list[dict] = []
        error_count = 0

        for kw in keywords:
            if not kw or not kw.strip():
                continue
            papers = self.search(kw.strip(), max_results=papers_per_keyword)
            if self._had_api_error:
                error_count += 1
            for p in papers:
                aid = p.get("arxiv_id", p.get("paper_id", ""))
                if aid and aid not in seen_ids:
                    seen_ids.add(aid)
                    all_papers.append(p)
                elif not aid:
                    all_papers.append(p)
            time.sleep(1.0)

        if not all_papers and error_count >= max(len(keywords), 1):
            self._last_error = (
                "All keyword queries failed. arXiv may be rate-limiting or unreachable."
            )
            self._had_api_error = True

        return all_papers

    # ── Internal ───────────────────────────────────────────────────────

    def _to_dict(self, result: arxiv.Result) -> dict:
        title = (result.title or "").strip()
        abstract = (result.summary or "").strip().replace("\n", " ")
        authors_list = [a.name for a in (result.authors or [])]
        arxiv_id = (result.entry_id or "unknown").split("/")[-1]

        raw = {
            "title": title or "Untitled",
            "authors": ", ".join(authors_list) if authors_list else "Unknown",
            "abstract": abstract or "No abstract available.",
            "summary": abstract or "No abstract available.",
            "published_date": result.published,
            "published": result.published,
            "arxiv_id": arxiv_id,
            "paper_id": arxiv_id,
            "url": result.entry_id or "",
            "link": result.entry_id or "",
            "pdf_url": result.pdf_url or "",
            "source": "arxiv",
            "source_id": arxiv_id,
            "sources": ["arxiv"],
        }
        return normalize_paper(raw)
