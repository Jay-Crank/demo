"""Semantic Scholar source client — queries the S2 Graph API.

Docs: https://api.semanticscholar.org/api-docs/graph
Base URL: https://api.semanticscholar.org/graph/v1/paper/search
"""

import time
import urllib.request
import urllib.parse
import json

from modules.paper_schema import normalize_paper
from modules.sources.base import BaseSource


S2_BASE = "https://api.semanticscholar.org/graph/v1/paper/search"
REQUEST_TIMEOUT = 15  # seconds
BATCH_SIZE = 100  # S2 default batch size


# Fields to request from Semantic Scholar
_REQUEST_FIELDS = (
    "paperId,title,authors,abstract,year,url,venue,"
    "citationCount,influentialCitationCount,fieldsOfStudy,"
    "publicationTypes,externalIds,openAccessPdf,"
    "publicationDate,journal"
)


class SemanticScholarSource(BaseSource):
    """Semantic Scholar Graph API data source."""

    def __init__(self, api_key: str = ""):
        self._api_key = api_key
        self._last_error: str | None = None
        self._had_error: bool = False

    @property
    def name(self) -> str:
        return "semantic_scholar"

    @property
    def last_error(self) -> str | None:
        return self._last_error

    @property
    def had_error(self) -> bool:
        return self._had_error

    def search(
        self,
        query: str,
        max_results: int = 20,
        **kwargs,
    ) -> list[dict]:
        """Search Semantic Scholar for papers.

        Parameters
        ----------
        query : str
            Search query string.
        max_results : int
            Max papers to return (S2 caps at 100 per request, paginated).
        """
        self._last_error = None
        self._had_error = False

        if not query or not query.strip():
            self._last_error = "Query is empty."
            return []

        papers: list[dict] = []
        offset = 0
        limit = min(max_results, BATCH_SIZE)

        while len(papers) < max_results:
            params = {
                "query": query.strip(),
                "limit": str(limit),
                "offset": str(offset),
                "fields": _REQUEST_FIELDS,
            }
            url = S2_BASE + "?" + urllib.parse.urlencode(params)
            data = self._fetch_json(url)
            if data is None:
                break

            results = data.get("data") or []
            for item in results:
                papers.append(self._to_dict(item))
                if len(papers) >= max_results:
                    break

            # Pagination — Semantic Scholar uses offset-based pagination
            next_offset = data.get("next")
            if next_offset is not None:
                offset = next_offset
            else:
                break

            if offset > 500:  # safety cap — S2 allows 1000 max, we cap at 500
                break
            time.sleep(0.5)  # courtesy delay

        if not papers and not self._last_error:
            self._last_error = f"No papers found for query '{query}'."

        return papers

    # ── Internal ───────────────────────────────────────────────────────

    def _fetch_json(self, url: str) -> dict | None:
        """Fetch a URL and parse the JSON response.  Returns None on failure."""
        req = urllib.request.Request(url)
        req.add_header("Accept", "application/json")
        if self._api_key:
            req.add_header("x-api-key", self._api_key)
        else:
            req.add_header("User-Agent", "AcademicResearchAgent/1.0")

        try:
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                body = resp.read().decode("utf-8")
                return json.loads(body)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace") if e.fp else ""
            self._last_error = f"Semantic Scholar HTTP {e.code}: {e.reason}"
            if body:
                # Try to extract a more descriptive message
                try:
                    detail = json.loads(body).get("message", "")
                    if detail:
                        self._last_error += f" — {detail}"
                except Exception:
                    pass
            self._had_error = True
        except urllib.error.URLError as e:
            self._last_error = f"Semantic Scholar network error: {e.reason}"
            self._had_error = True
        except json.JSONDecodeError as e:
            self._last_error = f"Semantic Scholar JSON parse error: {e}"
            self._had_error = True
        except Exception as e:
            self._last_error = f"Semantic Scholar error: {e}"
            self._had_error = True
        return None

    def _to_dict(self, item: dict) -> dict:
        """Convert a Semantic Scholar paper item to a unified paper dict."""
        # ── Title ──
        title = item.get("title") or "Untitled"

        # ── Authors ──
        authors_list = item.get("authors") or []
        author_names = [a.get("name", "") for a in authors_list if a.get("name")]
        authors = ", ".join(author_names) if author_names else "Unknown"

        # ── Abstract ──
        abstract = item.get("abstract") or ""

        # ── Year / Date ──
        year = item.get("year")
        pub_date = item.get("publicationDate") or ""

        # ── External IDs ──
        ext_ids = item.get("externalIds") or {}
        doi = ext_ids.get("DOI", "") or ""
        arxiv_id = ext_ids.get("ArXiv", "") or ""

        # ── URL ──
        url = item.get("url") or ""
        if not url and doi:
            url = f"https://doi.org/{doi}"

        # ── PDF ──
        oa_pdf = item.get("openAccessPdf") or {}
        pdf_url = oa_pdf.get("url", "") if oa_pdf else ""

        # ── Venue ──
        venue = ""
        journal = item.get("journal") or {}
        if journal:
            venue = journal.get("name", "") or ""
        if not venue:
            v_info = item.get("venue") or ""
            if isinstance(v_info, dict):
                venue = v_info.get("name", "") or v_info.get("displayName", "") or ""
            elif isinstance(v_info, str):
                venue = v_info

        # ── Publisher ──
        publisher = ""
        pub_types = item.get("publicationTypes") or []
        if pub_types:
            publisher = ", ".join(pub_types)

        # ── Citations ──
        citation_count = item.get("citationCount") or 0
        influential_citation_count = item.get("influentialCitationCount") or 0

        # ── Fields of study ──
        fos_list = item.get("fieldsOfStudy") or []
        fields_of_study = [f for f in fos_list if f]

        # ── Paper ID ──
        paper_id = item.get("paperId") or ""

        raw = {
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "summary": abstract,
            "year": year,
            "published_date": pub_date,
            "published": pub_date,
            "doi": doi,
            "arxiv_id": arxiv_id,
            "paper_id": arxiv_id or doi or paper_id,
            "url": url,
            "link": url,
            "pdf_url": pdf_url,
            "venue": venue,
            "publisher": publisher,
            "citation_count": citation_count,
            "influential_citation_count": influential_citation_count,
            "fields_of_study": fields_of_study,
            "keywords": [],
            "is_open_access": bool(pdf_url),
            "source": "semantic_scholar",
            "source_id": paper_id,
            "sources": ["semantic_scholar"],
            "raw_data": {
                "paperId": paper_id,
                "externalIds": ext_ids,
                "publicationTypes": pub_types,
            },
        }
        return normalize_paper(raw)
