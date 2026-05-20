"""OpenAlex source client — queries the OpenAlex REST API.

Docs: https://docs.openalex.org/api-entities/works
Base URL: https://api.openalex.org/works
"""

import time
import urllib.request
import urllib.parse
import json

from modules.paper_schema import normalize_paper
from modules.sources.base import BaseSource


OPENALEX_BASE = "https://api.openalex.org/works"
REQUEST_TIMEOUT = 15  # seconds


class OpenAlexSource(BaseSource):
    """OpenAlex scholarly-works data source."""

    def __init__(self, email: str = ""):
        self._email = email
        self._last_error: str | None = None
        self._had_error: bool = False

    @property
    def name(self) -> str:
        return "openalex"

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
        from_date: str | None = None,
        to_date: str | None = None,
        **kwargs,
    ) -> list[dict]:
        """Search OpenAlex works.

        Parameters
        ----------
        query : str
            Free-text search query.
        max_results : int
            Max papers to return (OpenAlex caps at 200 per page).
        from_date, to_date : str | None
            ISO-format date strings, e.g. ``"2024-01-01"``.
        """
        self._last_error = None
        self._had_error = False

        if not query or not query.strip():
            self._last_error = "Query is empty."
            return []

        papers: list[dict] = []
        per_page = min(max_results, 200)
        page = 1

        while len(papers) < max_results:
            params = {
                "search": query.strip(),
                "per_page": str(per_page),
                "page": str(page),
            }
            # Date filter
            date_filter_parts = []
            if from_date:
                date_filter_parts.append(f"from_publication_date:{from_date}")
            if to_date:
                date_filter_parts.append(f"to_publication_date:{to_date}")
            if date_filter_parts:
                params["filter"] = ",".join(date_filter_parts)

            url = OPENALEX_BASE + "?" + urllib.parse.urlencode(params)
            data = self._fetch_json(url)
            if data is None:
                break

            results = data.get("results") or []
            for w in results:
                papers.append(self._to_dict(w))
                if len(papers) >= max_results:
                    break

            # Pagination
            meta = data.get("meta", {})
            total = meta.get("count", 0)
            if page * per_page >= total:
                break
            page += 1
            if page > 5:  # safety cap — 5 pages max
                break
            time.sleep(0.3)  # courtesy delay

        if not papers and not self._last_error:
            self._last_error = f"No papers found for query '{query}'."

        return papers

    # ── Internal ───────────────────────────────────────────────────────

    def _fetch_json(self, url: str) -> dict | None:
        """Fetch a URL and parse the JSON response.  Returns None on failure."""
        req = urllib.request.Request(url)
        req.add_header("Accept", "application/json")
        if self._email:
            req.add_header("User-Agent", f"mailto:{self._email}")
        else:
            req.add_header("User-Agent", "AcademicResearchAgent/1.0")

        try:
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                body = resp.read().decode("utf-8")
                return json.loads(body)
        except urllib.error.HTTPError as e:
            self._last_error = f"OpenAlex HTTP {e.code}: {e.reason}"
            self._had_error = True
        except urllib.error.URLError as e:
            self._last_error = f"OpenAlex network error: {e.reason}"
            self._had_error = True
        except json.JSONDecodeError as e:
            self._last_error = f"OpenAlex JSON parse error: {e}"
            self._had_error = True
        except Exception as e:
            self._last_error = f"OpenAlex error: {e}"
            self._had_error = True
        return None

    def _to_dict(self, work: dict) -> dict:
        """Convert an OpenAlex Work object to a unified paper dict."""
        # ── Title ──
        title = work.get("title") or "Untitled"

        # ── Authors ──
        authorships = work.get("authorships") or []
        author_names = []
        for a in authorships:
            auth = a.get("author", {})
            name = auth.get("display_name", "")
            if name:
                author_names.append(name)
        authors = ", ".join(author_names) if author_names else "Unknown"

        # ── Abstract (inverted index decode) ──
        inv_idx = work.get("abstract_inverted_index")
        abstract = _decode_inverted_index(inv_idx) if inv_idx else ""
        if not abstract:
            # Fallback: sometimes OpenAlex puts abstract directly (rare)
            abstract = work.get("abstract", "") or ""

        # ── Date / year ──
        pub_date_str = work.get("publication_date") or ""
        pub_year = work.get("publication_year")

        # ── DOI ──
        doi = work.get("doi") or ""
        if doi:
            doi = doi.removeprefix("https://doi.org/")

        # ── Venue / publisher ──
        primary_loc = work.get("primary_location") or {}
        source_info = primary_loc.get("source") or {}
        venue = source_info.get("display_name", "") or ""
        publisher = source_info.get("host_organization_name", "") or ""

        # ── URL ──
        url = primary_loc.get("landing_page_url", "") or work.get("id", "") or ""

        # ── Citations ──
        cited_by = work.get("cited_by_count", 0) or 0

        # ── Open access ──
        oa_info = work.get("open_access") or {}
        is_oa = oa_info.get("is_oa", False) or False

        # ── Fields of study ──
        concepts = work.get("concepts") or []
        fos = []
        for c in concepts:
            if c.get("level", 0) == 0:  # top-level concepts
                fos.append(c.get("display_name", ""))
        if not fos:
            fos = [c.get("display_name", "") for c in concepts[:3]]

        # ── Keywords ──
        kw_list = work.get("keywords") or []
        keywords = [k.get("display_name", "") for k in kw_list if k.get("display_name")]

        # ── Type ──
        work_type = work.get("type", "")

        raw = {
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "summary": abstract,
            "year": pub_year,
            "published_date": pub_date_str,
            "published": pub_date_str,
            "doi": doi,
            "arxiv_id": "",
            "paper_id": doi or work.get("id", "").split("/")[-1],
            "url": url,
            "link": url,
            "pdf_url": oa_info.get("oa_url", ""),
            "venue": venue,
            "publisher": publisher,
            "citation_count": cited_by,
            "influential_citation_count": 0,
            "fields_of_study": fos,
            "keywords": keywords,
            "is_open_access": is_oa,
            "source": "openalex",
            "source_id": work.get("id", ""),
            "sources": ["openalex"],
            "raw_data": {
                "type": work_type,
                "cited_by_count": cited_by,
                "openalex_id": work.get("id", ""),
            },
        }
        return normalize_paper(raw)


def _decode_inverted_index(inv_index: dict) -> str:
    """Decode OpenAlex inverted abstract index back to plain text.

    Input: ``{"word": [pos1, pos2, ...], ...}``
    Output: plain text string.
    """
    if not inv_index:
        return ""
    word_positions: list[tuple[int, str]] = []
    for word, positions in inv_index.items():
        for pos in positions:
            word_positions.append((pos, word))
    if not word_positions:
        return ""
    word_positions.sort(key=lambda x: x[0])
    return " ".join(w for _, w in word_positions)
