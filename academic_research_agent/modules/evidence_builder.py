"""Evidence pack builder — assigns paper IDs and extracts structured evidence.

Converts raw paper dicts into a compact, LLM-friendly evidence pack with:
- Short paper IDs (P1, P2, ...)
- Key findings extracted from abstracts
- Metric mentions extracted via regex
- Truncated abstracts (≤1200 chars)
"""

import re
from datetime import datetime


# ── Sentence splitter ────────────────────────────────────────────────────

def _split_sentences(text: str) -> list[str]:
    """Naive sentence splitter for academic English text."""
    if not text:
        return []
    # Split on period/exclamation/question followed by space or end
    raw = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in raw if s.strip()]


# ── Key-finding verbs ────────────────────────────────────────────────────

_FINDING_VERBS = [
    "propose", "present", "introduce", "show", "achieve", "outperform",
    "reduce", "improve", "demonstrate", "reveal", "establish",
    "obtain", "report", "exceed", "surpass", "advance",
]

# Must appear near the start of a sentence (first 15 words) to count as a claim
_FINDING_PATTERN = re.compile(
    r"(?:^|\.\s+)\s*"  # sentence start
    r"(?:\w+\s){0,15}"  # up to 15 words
    r"(" + "|".join(_FINDING_VERBS) + r")",
    re.IGNORECASE,
)


def _extract_key_findings(abstract: str, max_findings: int = 5) -> list[str]:
    """Extract sentences from abstract that express a finding/contribution."""
    if not abstract:
        return []
    sents = _split_sentences(abstract)
    findings = []
    for s in sents:
        sl = s.lower()
        if any(sl.startswith(v) or f" {v}" in sl[:80] for v in _FINDING_VERBS):
            clean = s.strip()[:300]
            findings.append(clean)
            if len(findings) >= max_findings:
                break
    return findings


# ── Metric mention extraction ────────────────────────────────────────────

_METRIC_PATTERNS = [
    # Percentages
    re.compile(r"\d{1,3}(?:\.\d{1,2})?\s*%"),
    # Accuracy / precision / recall / F1 with optional value
    re.compile(
        r"(?:accuracy|precision|recall|F1|F1-score|AUC|ROC-AUC|BLEU|ROUGE|METEOR|"
        r"RMSE|MAE|MSE|MAPE|perplexity|throughput|latency)"
        r"\s*(?:of|:|is|was|at|up to|≥|<=?|≥|≈)?\s*"
        r"\d{1,3}(?:\.\d{1,4})?\s*%?",
        re.IGNORECASE,
    ),
    # "X points" improvements
    re.compile(r"\d+(?:\.\d+)?\s*(?:percentage\s*)?points?\s*(?:improvement|gain|boost|increase|reduction|drop)", re.IGNORECASE),
    # "by X%" improvements
    re.compile(r"by\s+\d{1,3}(?:\.\d{1,2})?\s*%", re.IGNORECASE),
    # State-of-the-art / SOTA mentions
    re.compile(
        r"(?:state-of-the-art|SOTA|state of the art|new state|best|highest|record)(?:\s+\w+){0,10}",
        re.IGNORECASE,
    ),
    # Speed / memory claims
    re.compile(
        r"(?:\d+(?:\.\d+)?\s*[×xX]\s*(?:faster|speedup|speed-up)|"
        r"reduce[sd]?\s+(?:by|up to)\s+\d+(?:\.\d+)?\s*%|"
        r"\d+(?:\.\d+)?\s*%\s*(?:reduction|fewer|less|lower|smaller))",
        re.IGNORECASE,
    ),
]


def _extract_metric_mentions(abstract: str) -> list[str]:
    """Extract metric/numeric claims from the abstract."""
    if not abstract:
        return []
    seen: set[str] = set()
    mentions: list[str] = []
    for pat in _METRIC_PATTERNS:
        for m in pat.finditer(abstract):
            text = m.group(0).strip()
            # Normalise whitespace
            text = re.sub(r"\s+", " ", text)
            if text and text not in seen:
                seen.add(text)
                mentions.append(text)
                if len(mentions) >= 10:
                    return mentions
    return mentions


# ── Year extractor ───────────────────────────────────────────────────────

def _extract_year(paper: dict) -> str:
    pub = paper.get("published") or paper.get("published_date")
    if pub is None:
        return "未知"
    if isinstance(pub, datetime):
        return str(pub.year)
    if isinstance(pub, str):
        m = re.search(r"(\d{4})", pub)
        if m:
            return m.group(1)
    return "未知"


# ── Main builder ─────────────────────────────────────────────────────────

def build_evidence_pack(
    papers: list[dict],
    max_papers: int = 15,
) -> dict:
    """Build a compact evidence pack for LLM consumption.

    Parameters
    ----------
    papers : list[dict]
        Paper dicts (unified schema).  Should be top-ranked papers from the pipeline.
    max_papers : int
        Maximum number of papers to include (default 15).

    Returns
    -------
    dict with keys:
        paper_id_map : dict[str, dict]
            Mapping of "P1" → {title, authors, year, venue, sources, citation_count, url}
        papers : list[dict]
            Full evidence entries with paper_id, title, authors, year, venue,
            sources, citation_count, abstract, key_findings, metric_mentions, url
        total_papers : int
            Number of papers in the pack.
    """
    if not papers:
        return {"paper_id_map": {}, "papers": [], "total_papers": 0}

    trimmed = papers[:max_papers]
    pack_papers: list[dict] = []
    paper_id_map: dict[str, dict] = {}

    for idx, p in enumerate(trimmed):
        pid = f"P{idx + 1}"
        abstract_raw = (p.get("abstract") or p.get("summary") or "").strip()
        abstract = abstract_raw[:1200] if len(abstract_raw) > 1200 else abstract_raw

        authors_raw = p.get("authors", "")
        if isinstance(authors_raw, list):
            authors_str = ", ".join(authors_raw[:5])
        else:
            authors_str = str(authors_raw)

        year = _extract_year(p)
        venue = p.get("venue") or "未知"
        url = p.get("url") or p.get("link") or ""
        citation_count = int(p.get("citation_count", 0) or 0)

        sources = p.get("sources") or [p.get("source", "未知")]

        key_findings = _extract_key_findings(abstract_raw)
        metric_mentions = _extract_metric_mentions(abstract_raw)

        entry = {
            "paper_id": pid,
            "title": p.get("title", "Untitled").strip(),
            "authors": authors_str or "未知",
            "year": year,
            "venue": venue,
            "sources": sources if sources else ["未知"],
            "citation_count": citation_count,
            "abstract": abstract or "无摘要",
            "key_findings": key_findings,
            "metric_mentions": metric_mentions,
            "url": url,
        }
        pack_papers.append(entry)

        paper_id_map[pid] = {
            "title": entry["title"],
            "authors": entry["authors"],
            "year": entry["year"],
            "venue": entry["venue"],
            "citation_count": entry["citation_count"],
            "url": entry["url"],
        }

    return {
        "paper_id_map": paper_id_map,
        "papers": pack_papers,
        "total_papers": len(pack_papers),
    }


def format_evidence_pack_for_prompt(evidence_pack: dict) -> str:
    """Render the evidence pack as a text block for insertion into an LLM prompt.

    Returns a string suitable for inclusion in a user message.
    """
    papers = evidence_pack.get("papers", [])
    if not papers:
        return "（无可用的证据论文）"

    lines: list[str] = []
    lines.append(f"以下是可以引用的事实证据库（共 {len(papers)} 篇论文）：")
    lines.append("")

    for ep in papers:
        pid = ep["paper_id"]
        lines.append(f"─── {pid} ───")
        lines.append(f"标题: {ep['title']}")
        lines.append(f"作者: {ep['authors']}")
        lines.append(f"年份: {ep['year']}  |  期刊/会议: {ep['venue']}  |  引用数: {ep['citation_count']}")
        lines.append(f"来源: {', '.join(ep['sources'])}")
        if ep["url"]:
            lines.append(f"链接: {ep['url']}")
        lines.append("")
        lines.append(f"摘要:\n{ep['abstract']}")
        if ep["key_findings"]:
            lines.append("")
            lines.append("关键发现:")
            for kf in ep["key_findings"]:
                lines.append(f"  • {kf}")
        if ep["metric_mentions"]:
            lines.append("")
            lines.append("指标数据:")
            for mm in ep["metric_mentions"]:
                lines.append(f"  • {mm}")
        lines.append("")

    return "\n".join(lines)
