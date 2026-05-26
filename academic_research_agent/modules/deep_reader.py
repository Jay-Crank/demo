"""Deep reader — full-text PDF analysis for arXiv papers.

Downloads PDFs from arXiv, extracts full text, parses academic sections,
generates deep reading cards and suggested reading paths.

Pipeline:
  1. download_arxiv_pdf(arxiv_id) → PDF file
  2. extract_full_text(pdf_path) → raw text
  3. parse_paper_sections(full_text) → structured sections
  4. build_deep_reading_card(paper, sections) → structured card
  5. build_reading_path(papers, cards) → reading order suggestions
  6. build_fulltext_evidence_pack(papers, cards) → LLM-ready evidence
"""

import os
import re
import time
import hashlib
from pathlib import Path
from collections import defaultdict

import requests


#  Constants 

_DEFAULT_CACHE_DIR = Path(__file__).parent.parent / ".cache" / "pdfs"
_ARXIV_PDF_URL = "https://arxiv.org/pdf/{arxiv_id}.pdf"
_REQUEST_DELAY = 3.0  # seconds between arXiv requests (rate limit)
_REQUEST_TIMEOUT = 30  # seconds

# Regex to extract arXiv ID from URLs or strings
_ARXIV_ID_FROM_URL = re.compile(r"arxiv\.org/(?:abs|pdf)/([\w.]+)")
_ARXIV_ID_CLEAN = re.compile(r"^(\d{4}\.\d{4,5})(?:v\d+)?$")


def extract_arxiv_id(paper: dict) -> str:
    """Extract arXiv ID from a paper dict, trying multiple fields.

    Checks in order: arxiv_id, paper_id, source_id, link, url, entry_id.
    Strips version suffix and URL prefixes.
    """
    # Direct fields
    for key in ("arxiv_id", "paper_id", "source_id"):
        val = paper.get(key, "")
        if val:
            # Try to extract from URL first
            m = _ARXIV_ID_FROM_URL.search(str(val))
            if m:
                return m.group(1)
            # Check if it looks like a raw ID
            m = _ARXIV_ID_CLEAN.match(str(val))
            if m:
                return m.group(1)
            # If it's just a number-like string without version
            if re.match(r"^\d{4}\.\d{4,5}$", str(val)):
                return str(val)

    # URL fields
    for key in ("link", "url", "entry_id"):
        val = paper.get(key, "")
        m = _ARXIV_ID_FROM_URL.search(str(val))
        if m:
            return m.group(1)

    # Check externalIds for Semantic Scholar papers
    ext = paper.get("externalIds", {})
    if isinstance(ext, dict):
        arxiv_val = ext.get("ArXiv", "") or ext.get("arxiv", "")
        if arxiv_val:
            return str(arxiv_val)

    return ""

# Section boundary patterns.
# Match section headings: (blank-line or text-start) + optional numbering +
# keyword + rest of short line.  The heading line must be ≤ 100 chars to avoid
# matching random sentences that happen to contain a section keyword.
# Patterns use re.IGNORECASE so only lowercase alternatives are needed.
_SECTION_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("abstract", re.compile(
        r"(?:\n\s*\n|\A)\s*(?:abstract|a b s t r a c t)\b[^\n]{0,80}\s*\n", re.M | re.I,
    )),
    ("introduction", re.compile(
        r"(?:\n\s*\n|\A)\s*(?:\d+(?:\.\d+)*\.?\s+)?(?:introduction)\b[^\n]{0,80}\s*\n", re.M | re.I,
    )),
    ("related_work", re.compile(
        r"(?:\n\s*\n|\A)\s*(?:\d+(?:\.\d+)*\.?\s+)?(?:related\s*work|background|literature\s*review|prior\s*work|preliminaries)\b[^\n]{0,80}\s*\n", re.M | re.I,
    )),
    ("method", re.compile(
        r"(?:\n\s*\n|\A)\s*(?:\d+(?:\.\d+)*\.?\s+)?(?:method(?:ology|s)?|approach|proposed\s*(?:method|framework|model|architecture)?|our\s*(?:method|approach|model|framework)|model\s*(?:architecture|overview|description)?|architecture|framework|system\s*design|technical\s*approach|methodology)\b[^\n]{0,80}\s*\n", re.M | re.I,
    )),
    ("experiments", re.compile(
        r"(?:\n\s*\n|\A)\s*(?:\d+(?:\.\d+)*\.?\s+)?(?:experiments?|experimental\s*(?:setup|results?|evaluation|study)?|evaluation|results?(?:\s*(?:and|&)\s*(?:analysis|discussion))?|empirical\s*(?:study|evaluation)|benchmarks?|performance\s*(?:evaluation|comparison)|experiment\s*(?:setup|design|results?))\b[^\n]{0,80}\s*\n", re.M | re.I,
    )),
    ("discussion", re.compile(
        r"(?:\n\s*\n|\A)\s*(?:\d+(?:\.\d+)*\.?\s+)?(?:discussions?|analysis|open\s*problems?|challenges?(?:\s*(?:and|&))?|limitations?(?:\s*(?:and|&))?|case\s*stud(?:y|ies))\b[^\n]{0,80}\s*\n", re.M | re.I,
    )),
    ("conclusion", re.compile(
        r"(?:\n\s*\n|\A)\s*(?:\d+(?:\.\d+)*\.?\s+)?(?:conclusions?|summary|concluding\s*remarks?|future\s*work|final\s*remarks?|limitations?\s*(?:and|&)\s*future)\b[^\n]{0,80}\s*\n", re.M | re.I,
    )),
]

# Limitation / future work indicators in full text
_LIMITATION_PATTERNS = re.compile(
    r"(?:limitation|drawback|weakness|shortcoming|fail|not yet|"
    r"remains|future work|open problem|bottleneck|"
    r"still lacks|remains unclear|requires further|more research|"
    r"not well understood|poorly understood|"
    r"lack of|limited by|suffers from|"
    r"future research|further investigation|"
    r"we acknowledge|we note that|"
    r"one limitation|a limitation|the limitation|"
    r"not address|does not address|"
    r"beyond the scope|left for future)",
    re.IGNORECASE,
)

# Experiment table indicators
_EXPERIMENT_TABLE_KEYWORDS = [
    "accuracy", "precision", "recall", "f1", "bleu", "rouge", "perplexity",
    "baseline", "ours", "model", "method", "dataset", "result",
    "table", "performance", "comparison", "compared",
]


#  PDF download 

def _ensure_cache_dir(cache_dir: Path | None = None) -> Path:
    d = cache_dir or _DEFAULT_CACHE_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def _pdf_path_for(arxiv_id: str, cache_dir: Path) -> Path:
    """Deterministic filename from arxiv_id."""
    fid = arxiv_id.replace("/", "_")
    return cache_dir / f"{fid}.pdf"


def download_arxiv_pdf(
    arxiv_id: str,
    cache_dir: Path | None = None,
    force: bool = False,
) -> Path | None:
    """Download a PDF from arXiv, caching it locally.

    Parameters
    ----------
    arxiv_id : str
        arXiv identifier, e.g. "2301.12345" or "2301.12345v2".
    cache_dir : Path | None
        Directory for cached PDFs. Defaults to .cache/pdfs/.
    force : bool
        Re-download even if cached.

    Returns
    -------
    Path to the downloaded PDF, or None on failure.
    """
    if not arxiv_id or not arxiv_id.strip():
        return None

    cd = _ensure_cache_dir(cache_dir)
    dest = _pdf_path_for(arxiv_id, cd)

    if dest.exists() and not force:
        # Already cached — verify it's a valid PDF
        try:
            with open(dest, "rb") as f:
                header = f.read(5)
            if header == b"%PDF-":
                return dest
            # Corrupted, re-download
        except Exception:
            pass

    url = _ARXIV_PDF_URL.format(arxiv_id=arxiv_id)
    try:
        resp = requests.get(url, timeout=_REQUEST_TIMEOUT, headers={
            "User-Agent": "AcademicResearchAgent/1.0 (mailto:research@example.com)",
        })
        if resp.status_code == 200 and resp.content[:5] == b"%PDF-":
            with open(dest, "wb") as f:
                f.write(resp.content)
            return dest
        elif resp.status_code == 404:
            # Try without version suffix
            clean_id = re.sub(r"v\d+$", "", arxiv_id)
            if clean_id != arxiv_id:
                return download_arxiv_pdf(clean_id, cache_dir, force)
        return None
    except Exception:
        return None


def batch_download_pdfs(
    arxiv_ids: list[str],
    cache_dir: Path | None = None,
    max_workers: int = 3,
    progress_callback=None,
) -> dict[str, Path | None]:
    """Download multiple PDFs with rate limiting.

    Returns dict mapping arxiv_id → Path (or None if failed).
    """
    results: dict[str, Path | None] = {}
    for i, aid in enumerate(arxiv_ids):
        if not aid:
            results[aid] = None
            continue
        path = download_arxiv_pdf(aid, cache_dir)
        results[aid] = path
        if progress_callback:
            progress_callback(i + 1, len(arxiv_ids), aid, path is not None)
        if i < len(arxiv_ids) - 1 and path is not None:
            time.sleep(_REQUEST_DELAY)
    return results


#  PDF text extraction 

def extract_full_text(pdf_path: Path | str) -> tuple[str, str | None]:
    """Extract full text from a PDF using pymupdf.

    Handles multi-column academic papers by using pymupdf's
    built-in text extraction with block sorting.

    Returns (text, error_message). text is "" on failure, error_message
    explains why (or None on success).
    """
    # Try pymupdf first
    try:
        import fitz  # pymupdf
    except ImportError:
        try:
            fb = _fallback_extract(pdf_path)
            if fb:
                return fb, None
        except Exception:
            pass
        return "", "pymupdf 未安装，请运行: pip install pymupdf"

    try:
        doc = fitz.open(str(pdf_path))
    except Exception as e:
        return "", f"无法打开 PDF: {e}"

    # Method 1: get_text("text") with sort — works for most LaTeX PDFs
    pages_text: list[str] = []
    for page in doc:
        text = page.get_text("text", sort=True)
        if text and text.strip():
            pages_text.append(text)
    doc.close()
    full_text = "\n\n".join(pages_text)

    # Method 2: blocks mode (correctly uses b[4] for text, not b[6] which is block_type)
    if len(full_text) < 100:
        try:
            blocks_text: list[str] = []
            doc2 = fitz.open(str(pdf_path))
            for page in doc2:
                blocks = page.get_text("blocks")
                for b in sorted(blocks, key=lambda b: (b[1], b[0])):
                    if len(b) > 4 and isinstance(b[4], str) and b[4].strip():
                        blocks_text.append(b[4].strip())
            doc2.close()
            if blocks_text:
                full_text = "\n\n".join(blocks_text)
        except Exception:
            pass

    # Method 3: get_text("text") without sort (some PDFs work better this way)
    if len(full_text) < 100:
        try:
            doc3 = fitz.open(str(pdf_path))
            raw_pages: list[str] = []
            for page in doc3:
                text = page.get_text("text")
                if text and text.strip():
                    raw_pages.append(text)
            doc3.close()
            if raw_pages:
                raw_text = "\n\n".join(raw_pages)
                if len(raw_text) > len(full_text):
                    full_text = raw_text
        except Exception:
            pass

    if len(full_text) < 50:
        return "", "PDF 文本提取失败（可能为扫描版或图片格式）"

    # Clean: remove excessive whitespace, fix line breaks within paragraphs
    full_text = re.sub(r"-\n(\w)", r"\1", full_text)  # hyphenated words
    full_text = re.sub(r"(?<!\n)\n(?!\n)", " ", full_text)  # single newlines → space
    full_text = re.sub(r"\n{3,}", "\n\n", full_text)  # collapse multiple blank lines
    full_text = full_text.strip()

    return full_text, None


def _fallback_extract(pdf_path: Path | str) -> str:
    """Fallback PDF extraction without pymupdf."""
    try:
        import subprocess
        result = subprocess.run(
            ["pdftotext", "-layout", str(pdf_path), "-"],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return ""


#  Section parsing 

def parse_paper_sections(full_text: str) -> dict:
    """Parse academic paper into named sections.

    Returns dict with keys: abstract, introduction, related_work,
    method, experiments, discussion, conclusion, limitations,
    full_text, section_boundaries.

    If a section is not found, its value is an empty string.
    """
    if not full_text:
        return _empty_sections()

    # Find section boundaries
    matches: list[tuple[str, int, int]] = []
    for sec_name, pat in _SECTION_PATTERNS:
        m = pat.search(full_text)
        if m:
            matches.append((sec_name, m.start(), m.end()))

    # Sort by position in text
    matches.sort(key=lambda x: x[1])

    # Extract text for each section
    sections: dict[str, str] = {name: "" for name, _ in _SECTION_PATTERNS}
    sections["full_text"] = full_text
    sections["section_boundaries"] = {}  # type: ignore

    for i, (sec_name, start, _) in enumerate(matches):
        if i + 1 < len(matches):
            end = matches[i + 1][1]
        else:
            end = len(full_text)
        sections[sec_name] = full_text[start:end].strip()
        sections["section_boundaries"][sec_name] = {"start": start, "end": end}  # type: ignore

    # Use first 500 chars as abstract if not found
    if not sections.get("abstract"):
        sections["abstract"] = full_text[:500].strip()

    # Extract limitations
    sections["limitations"] = _extract_limitations(full_text)

    return sections


def _empty_sections() -> dict:
    d = {"full_text": "", "section_boundaries": {}, "limitations": ""}
    for sec_name, _ in _SECTION_PATTERNS:
        d[sec_name] = ""
    return d


def _extract_limitations(full_text: str) -> str:
    """Extract sentences containing limitation/future work signals."""
    if not full_text:
        return ""
    sents = re.split(r"(?<=[.!?])\s+", full_text)
    lim_sents = []
    for s in sents:
        if _LIMITATION_PATTERNS.search(s) and len(s) > 30:
            lim_sents.append(s.strip()[:500])
    return "\n".join(lim_sents[:10])


#  Experiment data extraction 

def _extract_experiment_highlights(sections: dict) -> str:
    """Extract key experimental findings from the experiments/discussion sections."""
    exp_text = sections.get("experiments", "") or sections.get("discussion", "")
    if not exp_text:
        return ""

    # Find lines with numbers + metric keywords
    metric_pat = re.compile(
        r"(\d{1,3}(?:\.\d{1,4})?\s*%?)\s*.*?"
        r"(accuracy|precision|recall|F1|BLEU|ROUGE|AUC|perplexity|RMSE|MAE)",
        re.IGNORECASE,
    )
    highlights = []
    for line in exp_text.split("\n"):
        if metric_pat.search(line) and len(line) > 20:
            highlights.append(line.strip()[:300])
    return "\n".join(highlights[:20])


def _extract_method_summary(sections: dict) -> str:
    """Extract a concise method description."""
    method_text = sections.get("method", "")
    if not method_text:
        # Try intro + conclusion
        method_text = (
            (sections.get("introduction", "") or "")[:1000] + " " +
            (sections.get("conclusion", "") or "")[:500]
        )

    # Extract sentences with method-indicating verbs
    method_verbs = [
        "propose", "present", "introduce", "design", "develop",
        "architecture", "framework", "model", "algorithm",
        "consists of", "composed of", "based on", "comprises",
    ]
    sents = re.split(r"(?<=[.!?])\s+", method_text[:3000])
    method_sents = []
    for s in sents:
        if any(v in s.lower() for v in method_verbs) and len(s) > 40:
            method_sents.append(s.strip()[:300])
    return " ".join(method_sents[:5]) if method_sents else method_text[:500]


#  Deep reading card 

def build_deep_reading_card(
    paper: dict,
    sections: dict | None = None,
    pdf_path: Path | None = None,
) -> dict:
    """Build a structured deep reading card for one paper.

    Parameters
    ----------
    paper : dict
        Paper metadata dict (must have title, authors, year, arxiv_id, etc.).
    sections : dict | None
        Parsed sections from parse_paper_sections(). If None, only metadata
        is included (no full-text analysis).
    pdf_path : Path | None
        Path to the downloaded PDF (for reference).

    Returns
    -------
    dict with keys:
        title, authors, year, venue, citation_count, url, arxiv_id,
        core_contribution, method_overview, key_results, limitations,
        novelty_assessment, target_audience, paper_type, pdf_available,
        has_full_text
    """
    has_ft = sections is not None and bool(sections.get("full_text", ""))
    pdf_ok = pdf_path is not None and pdf_path.exists() if pdf_path else False

    card: dict = {
        "title": paper.get("title", "Untitled"),
        "authors": paper.get("authors", "未知"),
        "year": paper.get("year", "未知"),
        "venue": paper.get("venue", "未知"),
        "citation_count": paper.get("citation_count", 0) or 0,
        "url": paper.get("url") or paper.get("link", ""),
        "arxiv_id": paper.get("arxiv_id", ""),
        "pdf_available": pdf_ok,
        "has_full_text": has_ft,
        "composite_score": paper.get("composite_score", 0),
    }

    if not has_ft:
        # Summary-only card (from abstract)
        abstract = paper.get("abstract") or paper.get("summary", "")
        card["core_contribution"] = _extract_core_contribution(abstract)
        card["method_overview"] = abstract[:400]
        card["key_results"] = "\n".join(paper.get("key_findings", []))[:500]
        card["limitations"] = "（未获取全文，无法提取局限性）"
        card["novelty_assessment"] = "（未获取全文，无法评估新颖性）"
        card["target_audience"] = "（未获取全文）"
        card["paper_type"] = _classify_paper_type(paper)
        return card

    # Full-text enriched card
    ft = sections.get("full_text", "")
    card["core_contribution"] = _extract_core_contribution(
        sections.get("abstract", "") or sections.get("introduction", "")[:800] or ft[:500]
    )
    card["method_overview"] = _extract_method_summary(sections)
    card["key_results"] = _extract_experiment_highlights(sections)
    card["limitations"] = sections.get("limitations", "") or "（论文中未明确提及）"
    card["novelty_assessment"] = _assess_novelty(sections, paper)
    card["target_audience"] = _suggest_audience(sections, paper)
    card["paper_type"] = _classify_paper_type(paper)
    card["section_summary"] = {
        k: sections.get(k, "")[:500]
        for k in ["abstract", "method", "experiments", "conclusion"]
    }

    return card


def _extract_core_contribution(text: str) -> str:
    """Extract 1-2 sentence core contribution from text."""
    if not text:
        return "（无法提取）"
    # Look for contribution-indicating sentences
    contrib_verbs = [
        "propose", "present", "introduce", "demonstrate", "achieve",
        "shows? that", "find that", "reveals?", "establish",
    ]
    sents = re.split(r"(?<=[.!?])\s+", text[:1500])
    for s in sents:
        sl = s.lower()
        if any(re.search(rf"\b{v}\b", sl) for v in contrib_verbs) and len(s) > 50:
            return s.strip()[:300]
    # Fallback: first long enough sentence
    for s in sents:
        if len(s) > 60:
            return s.strip()[:300]
    return text[:300]


def _assess_novelty(sections: dict, paper: dict) -> str:
    """Assess the novelty level based on paper content."""
    intro = sections.get("introduction", "") or ""
    conc = sections.get("conclusion", "") or ""
    combined = f"{intro} {conc}".lower()

    novelty_signals = {
        "突破性创新": ["first", "novel", "new paradigm", "breakthrough", "pioneering"],
        "渐进式改进": ["improves?", "better than", "outperforms", "extends", "advances"],
        "应用适配": ["applied to", "application", "domain adaptation", "transfer"],
        "评测基准": ["benchmark", "evaluation", "compare", "survey", "review"],
    }

    scores = {}
    for level, keywords in novelty_signals.items():
        scores[level] = sum(1 for kw in keywords if re.search(rf"\b{kw}\b", combined))

    if not any(scores.values()):
        return "未明确说明"

    best = max(scores, key=scores.get)
    return best if scores[best] >= 2 else "未明确说明"


def _suggest_audience(sections: dict, paper: dict) -> str:
    """Suggest who should read this paper."""
    text = (
        sections.get("abstract", "") + " " +
        sections.get("introduction", "")[:500]
    ).lower()

    audiences = []
    if any(w in text for w in ["transformer", "attention", "bert", "gpt", "llm", "language model"]):
        audiences.append("NLP / LLM 研究者")
    if any(w in text for w in ["image", "vision", "cnn", "convolutional", "segmentation"]):
        audiences.append("计算机视觉研究者")
    if any(w in text for w in ["graph", "gnn", "node", "message passing"]):
        audiences.append("图学习研究者")
    if any(w in text for w in ["reinforcement", "policy", "reward", "rl"]):
        audiences.append("强化学习研究者")
    if any(w in text for w in ["survey", "review", "taxonomy", "comprehensive"]):
        audiences.append("领域入门者 / 研究生")
    if any(w in text for w in ["benchmark", "evaluation", "dataset"]):
        audiences.append("评测基准研究者")
    if any(w in text for w in ["application", "system", "deploy", "production"]):
        audiences.append("工业界实践者")

    return "、".join(audiences) if audiences else "该领域研究者"


def _classify_paper_type(paper: dict) -> str:
    """Classify paper type from title and abstract."""
    text = f"{paper.get('title', '')} {paper.get('abstract', '')} {paper.get('summary', '')}".lower()
    if any(w in text for w in ["survey", "review", "overview", "taxonomy"]):
        return "综述"
    if any(w in text for w in ["benchmark", "evaluation", "dataset"]):
        return "评测基准"
    if any(w in text for w in ["novel", "new", "propose", "introduce", "framework"]):
        return "方法创新"
    if any(w in text for w in ["application", "system", "tool", "platform"]):
        return "应用系统"
    return "其他"


#  Reading path builder 

def build_reading_path(
    papers: list[dict],
    deep_cards: list[dict],
) -> dict:
    """Build suggested reading paths based on paper relationships.

    Returns dict with:
        paths: list of {label, papers (ordered), rationale}
        dependency_graph: simple dependency hints
    """
    if len(papers) < 2:
        return {
            "paths": [{
                "label": "仅有 1 篇论文",
                "papers": [papers[0]] if papers else [],
                "rationale": "仅此一篇，无需排序。",
            }],
            "dependency_graph": {},
        }

    cards = list(deep_cards)
    paths: list[dict] = []

    #  Path 1: Quick overview (survey first, then latest SOTA) 
    overview_papers = []
    for i, c in enumerate(cards):
        if c.get("paper_type") == "综述":
            overview_papers.append(i)
    if not overview_papers:
        # Use highest composite score
        overview_papers = [max(range(len(cards)), key=lambda i: cards[i].get("composite_score", 0))]
    # Add freshest
    fresh_idx = max(range(len(cards)), key=lambda i: papers[i].get("freshness_score", 0))
    if fresh_idx not in overview_papers:
        overview_papers.append(fresh_idx)

    paths.append({
        "label": "快速了解领域",
        "papers": [papers[i] for i in overview_papers],
        "rationale": (
            f"[{cards[overview_papers[0]]['paper_type']}] 先读综述或最高分论文建立全局认知，"
            f"再读最新进展了解当前 SOTA。"
        ),
    })

    #  Path 2: Method lineage (same method family, chronological) 
    method_groups = _group_by_method_family(papers, cards)
    if method_groups:
        best_group = max(method_groups.values(), key=len)
        if len(best_group) >= 2:
            best_group.sort(key=lambda i: str(papers[i].get("year", "0")))
            paths.append({
                "label": f"方法深入：{cards[best_group[0]].get('paper_type', '方法')} 演进",
                "papers": [papers[i] for i in best_group],
                "rationale": "按时间顺序阅读同一方法族的论文，追踪技术演进脉络。",
            })

    #  Path 3: Application-oriented 
    app_indices = [
        i for i, c in enumerate(cards)
        if c.get("paper_type") in ("应用系统", "评测基准")
    ]
    if len(app_indices) >= 2:
        paths.append({
            "label": "应用与落地路径",
            "papers": [papers[i] for i in app_indices],
            "rationale": "关注应用系统和评测基准，适合需要工程落地的读者。",
        })

    return {
        "paths": paths,
        "dependency_graph": _build_simple_deps(papers, cards),
    }


def _group_by_method_family(papers: list[dict], cards: list[dict]) -> dict[str, list[int]]:
    """Group paper indices by method family."""
    method_kws = {
        "Transformer": ["transformer", "attention"],
        "Diffusion": ["diffusion", "denoising", "ddpm"],
        "GNN": ["graph neural", "gnn", "graph convolution"],
        "Contrastive": ["contrastive", "simclr"],
        "Reinforcement": ["reinforcement", "policy gradient"],
        "LLM / Fine-tuning": ["large language", "fine-tun", "lora", "prompt"],
    }
    groups: dict[str, list[int]] = defaultdict(list)
    for i, p in enumerate(papers):
        title_abs = f"{p.get('title', '')} {p.get('abstract', '')} {p.get('summary', '')}".lower()
        for fam, kws in method_kws.items():
            if any(kw in title_abs for kw in kws):
                groups[fam].append(i)
    return dict(groups)


def _build_simple_deps(papers: list[dict], cards: list[dict]) -> dict:
    """Simple heuristic dependency hints."""
    deps: dict[str, list[str]] = {}
    for i, p in enumerate(papers):
        deps[p.get("title", "?")[:60]] = []
        # Papers with earlier years might be foundational
        for j, p2 in enumerate(papers):
            if i == j:
                continue
            yi = str(p.get("year", "")).strip()
            yj = str(p2.get("year", "")).strip()
            if yi.isdigit() and yj.isdigit() and int(yi) < int(yj):
                if p2.get("citation_count", 0) > p.get("citation_count", 0) * 2:
                    deps[p.get("title", "?")[:60]].append(p2.get("title", "?")[:60])
    return deps


#  Full-text evidence pack 

def build_fulltext_evidence_pack(
    papers: list[dict],
    deep_cards: list[dict],
    sections_list: list[dict | None],
    max_papers: int = 10,
) -> dict:
    """Build an extended evidence pack with full-text enriched content.

    Returns same structure as evidence_builder but with:
    - full_text_sections (method, experiments, limitations) per paper
    - deep_card summaries
    """
    from modules.evidence_builder import build_evidence_pack, _extract_key_findings, _extract_metric_mentions

    base_pack = build_evidence_pack(papers, max_papers=max_papers)

    for i, ep in enumerate(base_pack.get("papers", [])):
        if i < len(deep_cards):
            card = deep_cards[i]
            ep["core_contribution"] = card.get("core_contribution", "")
            ep["method_overview"] = card.get("method_overview", "")
            ep["key_results_deep"] = card.get("key_results", "")[:1000]
            ep["limitations"] = card.get("limitations", "")
            ep["paper_type"] = card.get("paper_type", "")
            ep["has_full_text"] = card.get("has_full_text", False)

        if sections_list and i < len(sections_list) and sections_list[i]:
            sec = sections_list[i]
            ep["full_method"] = sec.get("method", "")[:2000]
            ep["full_experiments"] = sec.get("experiments", "")[:2000]
            ep["full_conclusion"] = sec.get("conclusion", "")[:1000]

    # Also enrich paper_id_map
    for i, ep in enumerate(base_pack.get("papers", [])):
        pid = ep.get("paper_id", "")
        if pid in base_pack.get("paper_id_map", {}):
            base_pack["paper_id_map"][pid]["paper_type"] = ep.get("paper_type", "")
            base_pack["paper_id_map"][pid]["has_full_text"] = ep.get("has_full_text", False)

    return base_pack


#  Formatting for UI 

def format_deep_card_markdown(card: dict) -> str:
    """Render a deep reading card as Markdown for Streamlit."""
    title = card.get("title", "Untitled")
    paper_type = card.get("paper_type", "")
    has_ft = card.get("has_full_text", False)
    pdf_ok = card.get("pdf_available", False)

    ft_badge = "[全文]" if has_ft else "[摘要]"
    pdf_badge = "[PDF]" if pdf_ok else ""

    lines = [
        f"## {ft_badge} {title}",
        "",
        f"**作者**：{card.get('authors', '未知')}　|　"
        f"**年份**：{card.get('year', '未知')}　|　"
        f"**类型**：{paper_type}　|　"
        f"**引用**：{card.get('citation_count', 0)}　{pdf_badge}",
    ]

    if card.get("url"):
        lines.append(f"**链接**：[{card['url'][:60]}]({card['url']})")

    lines.extend([
        "",
        "### 核心贡献",
        "",
        card.get("core_contribution", "（未提取）"),
        "",
        "### 方法概述",
        "",
        card.get("method_overview", "（未提取）")[:800],
    ])

    results = card.get("key_results", "")
    if results:
        lines.extend([
            "",
            "### 关键实验发现",
            "",
            results[:1000],
        ])

    limitations = card.get("limitations", "")
    if limitations and limitations != "（论文中未明确提及）":
        lines.extend([
            "",
            "### 局限性",
            "",
            limitations[:600],
        ])

    lines.extend([
        "",
        "### 新颖性评估",
        "",
        card.get("novelty_assessment", "未评估"),
        "",
        "### 适合读者",
        "",
        card.get("target_audience", "该领域研究者"),
    ])

    return "\n".join(lines)


def format_reading_path_markdown(reading_path: dict) -> str:
    """Render reading paths as Markdown."""
    paths = reading_path.get("paths", [])
    if not paths:
        return "（未生成阅读路径）"

    lines = ["## 推荐阅读路径", ""]

    for i, path in enumerate(paths, 1):
        lines.append(f"### 路径 {i}：{path['label']}")
        lines.append("")
        lines.append(f"**理由**：{path['rationale']}")
        lines.append("")
        lines.append("**阅读顺序**：")
        papers = path.get("papers", [])
        for j, p in enumerate(papers, 1):
            arrow = "+--" if j == len(papers) else "|--"
            title = p.get("title", "Untitled")[:80]
            year = p.get("year", "")
            authors = str(p.get("authors", ""))[:40]
            lines.append(f"  {arrow} **{j}. [{title}]({p.get('url', '#')})**")
            lines.append(f"     {authors} ({year})")
        lines.append("")

    return "\n".join(lines)
