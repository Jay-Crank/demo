"""Report generator — template-based answer and survey report generation.

Each generator returns a dict:
  { "report": <markdown-string>, "sections": {<name>: <content>, ...} }
The sections dict is consumed by verifier.assess_evidence for credibility checking.
"""

from datetime import datetime

from modules.utils import format_date


# ── Section helper ────────────────────────────────────────────────────

class _SectionBuilder:
    """Collects markdown lines and tracks named section boundaries."""

    def __init__(self):
        self._lines: list[str] = []
        self._sections: dict[str, str] = {}
        self._section_order: list[str] = []
        self._cursor: tuple[int, str | None] = (0, None)  # (start_line, name)

    def heading(self, name: str) -> None:
        self._flush_cursor()
        self._cursor = (len(self._lines), name)
        self._lines.append(f"## {name}")

    def add(self, *lines: str) -> None:
        self._lines.extend(lines)

    def blank(self) -> None:
        self._lines.append("")

    def _flush_cursor(self) -> None:
        start, name = self._cursor
        if name is not None and start < len(self._lines):
            body = "\n".join(self._lines[start:]).strip()
            self._sections[name] = body
            self._section_order.append(name)

    def build(self) -> dict:
        self._flush_cursor()
        return {
            "report": "\n".join(self._lines),
            "sections": dict(self._sections),
        }


# ── Search answer ─────────────────────────────────────────────────────

def generate_search_answer(
    query: str,
    keywords: list[str],
    papers: list[dict],
    top_n: int = 10,
) -> dict:
    """Generate a comprehensive answer based on top papers.

    Returns {"report": str, "sections": {name: content, ...}}.
    """
    top_papers = papers[:top_n]
    sb = _SectionBuilder()

    # Title
    sb._lines.append("## 综合回答")
    sb._lines.append("")

    if not top_papers:
        sb._lines.append(f"No papers found for query: *{query}*")
        return sb.build()

    best_paper = top_papers[0]

    # Count keyword hits in titles
    kw_counts = {kw: 0 for kw in keywords}
    for p in top_papers:
        title_lower = p["title"].lower()
        for kw in keywords:
            if kw.lower() in title_lower:
                kw_counts[kw] += 1

    # Intro paragraph
    sb._lines.append(
        f'针对您提出的问题 **"{query}"**，系统通过检索 arXiv 学术论文库，'
    )
    sb._lines.append(
        f"使用 {len(keywords)} 个扩展关键词检索并分析了 {len(papers)} 篇相关论文，"
    )
    sb._lines.append("经过去重和相关性排序后，得出以下综合结论：")
    sb._lines.append("")

    # ── Section 1: 研究现状概述 ──
    sb.heading("1. 研究现状概述")
    sb.blank()

    methods = set()
    for p in top_papers[:5]:
        summary_lower = p["summary"].lower()
        for term in [
            "transformer", "neural network", "deep learning", "reinforcement",
            "attention", "fine-tuning", "pre-training", "graph neural",
            "diffusion", "generative", "contrastive", "self-supervised",
            "multi-modal", "large language", "prompt", "retrieval-augmented",
            "chain-of-thought", "optimization", "federated", "knowledge graph",
        ]:
            if term in summary_lower:
                methods.add(term)

    if methods:
        sb.add(f"当前研究主要涉及以下技术方向：**{', '.join(sorted(methods)[:8])}**。")
    else:
        sb.add(f"从检索到的 {len(papers)} 篇论文来看，该领域的研究正在积极展开。")
    sb.blank()

    # ── Section 2: 关键发现 ──
    sb.heading("2. 关键发现")
    sb.blank()

    for i, p in enumerate(top_papers[:8], 1):
        s = (p.get("summary") or "No abstract available.")[:300]
        summary_short = s.rsplit(".", 1)[0].strip() + "."
        rel = p.get("composite_score", 0)
        sb.add(f"{i}. **{p.get('title', 'Untitled')}**（相关性: {rel:.2%}）")
        sb.add(f"   - {summary_short}")
        sb.blank()

    # ── Section 3: 代表性论文 ──
    sb.heading("3. 代表性论文")
    sb.blank()

    first_author = best_paper['authors'].split(',')[0] if ',' in best_paper['authors'] else best_paper['authors']
    sb.add(
        f"其中最具代表性的工作是 **{best_paper['title']}**"
        f"（{first_author} 等，{format_date(best_paper.get('published'))}），"
    )
    sb.add("该研究的相关度得分最高，可作为进一步阅读的起点。")
    sb.blank()

    # ── Section 4: 趋势总结 ──
    sb.heading("4. 趋势总结")
    sb.blank()

    recent_count = sum(
        1 for p in top_papers
        if p.get("published") and (datetime.now().date() - p["published"].date()).days <= 180
    )
    sb.add(f"在 Top {top_n} 论文中，有 {recent_count} 篇发表于近半年内。")
    sb.add(f"表明该方向研究活跃度{'较高' if recent_count >= 4 else '一般'}。")
    sb.blank()

    sb.add("---")
    sb.add(f"*本回答由系统基于 {len(papers)} 篇 arXiv 论文自动生成，未接入大语言模型。*")

    return sb.build()


# ── Survey report ─────────────────────────────────────────────────────

def generate_survey_report(
    research_area: str,
    sub_directions: list[str],
    all_papers: dict[str, list[dict]],
    clustering_result: dict | None = None,
) -> dict:
    """Generate a structured survey report.

    Parameters
    ----------
    clustering_result : dict | None
        Output from topic_clusterer.cluster_papers(). If provided, a
        "研究方向图谱" section is inserted after 主要技术路线.

    Returns {"report": str, "sections": {name: content, ...}}.
    """
    now = datetime.now()
    total_papers = sum(len(v) for v in all_papers.values())

    sb = _SectionBuilder()

    # Title
    sb._lines.append(f"# 学术调研报告：{research_area}")
    sb._lines.append("")
    sb._lines.append(f"*生成日期：{format_date(now)}*")
    sb._lines.append("")
    sb._lines.append("---")
    sb._lines.append("")

    # ── Section 1 ──
    sb.heading("1. 研究背景")
    sb.blank()
    sb.add(f"{research_area}是当前学术界和工业界广泛关注的研究领域。")
    sb.add(f"本报告通过系统检索 arXiv 论文库，将该领域拆分为 {len(sub_directions)} 个子方向，")
    sb.add(f"共检索并分析 {total_papers} 篇相关论文，整理出以下结构化调研报告。")
    sb.blank()

    # ── Section 2 ──
    sb.heading("2. 核心问题")
    sb.blank()

    problem_idx = 1
    for sub_dir, papers in all_papers.items():
        if not papers:
            continue
        top = papers[0]
        s = (top.get("summary") or "No abstract available.")[:200]
        sb.add(f"{problem_idx}. **{sub_dir}**")
        sb.add(f"   - {s}...")
        sb.blank()
        problem_idx += 1

    # ── Section 3 ──
    sb.heading("3. 主要技术路线")
    sb.blank()

    tech_methods = {
        "Deep Learning / Neural Networks": ["neural network", "deep learning", "transformer"],
        "Reinforcement Learning": ["reinforcement", "policy", "reward"],
        "Self-Supervised Learning": ["self-supervised", "contrastive", "pretext"],
        "Graph-Based Methods": ["graph", "GNN", "node"],
        "Generative Models": ["generative", "diffusion", "GAN"],
        "Optimization Methods": ["optimization", "gradient", "SGD", "Adam"],
    }

    method_hits = {}
    for papers in all_papers.values():
        for p in papers:
            text = (p["title"] + " " + p["summary"]).lower()
            for method, terms in tech_methods.items():
                if any(t.lower() in text for t in terms):
                    method_hits[method] = method_hits.get(method, 0) + 1

    sorted_methods = sorted(method_hits.items(), key=lambda x: x[1], reverse=True)
    for method, count in sorted_methods[:8]:
        pct = count / max(total_papers, 1) * 100
        sb.add(f"- **{method}**：相关论文 {count} 篇（{pct:.0f}%）")
    sb.blank()

    # ── Section 4: clustering (only when provided) ──
    sec_idx = 4
    if clustering_result and clustering_result.get("clusters"):
        sb.heading(f"{sec_idx}. 研究方向图谱（主题聚类）")
        sb.blank()
        clusters = clustering_result["clusters"]
        sb.add(
            f"基于 TF-IDF + KMeans 聚类算法，将 {clustering_result['total_papers']} 篇论文"
            f"自动划分为 {len(clusters)} 个主题簇"
            f"（轮廓系数: {clustering_result.get('silhouette_score', 0):.3f}）。"
        )
        sb.blank()

        for c in clusters:
            sb.add(f"### 主题 {c['id']+1}：{c['topic_name']}")
            sb.blank()
            sb.add(f"- **关键词**：{', '.join(c['keywords'])}")
            sb.add(f"- **论文数量**：{c['paper_count']} 篇")
            rep = c.get("representative", {})
            if rep:
                sb.add(f"- **代表论文**：[{rep['title']}]({rep.get('link', '#')})")
            # List top papers in cluster
            sb.add(f"- **包含论文**：")
            for p in c["papers"][:5]:
                sb.add(f"  - {p['title']}")
            if c["paper_count"] > 5:
                sb.add(f"  - ...及其他 {c['paper_count'] - 5} 篇")
            sb.blank()
        sec_idx += 1

    # ── Section: 代表性论文 ──
    sb.heading(f"{sec_idx}. 代表性论文")
    sb.blank()
    sec_idx += 1

    for sub_dir, papers in all_papers.items():
        if not papers:
            continue
        sb.add(f"### {sub_dir}")
        sb.blank()
        for i, p in enumerate(papers[:3], 1):
            authors_short = ", ".join(p["authors"].split(",")[:3])
            sb.add(
                f"| {i} | **{p['title']}** | {authors_short} | "
                f"{format_date(p.get('published'))} | "
                f"[arXiv]({p['link']}) |"
            )
        sb.blank()

    # ── Section: 当前研究趋势 ──
    sb.heading(f"{sec_idx}. 当前研究趋势")
    sb.blank()
    sec_idx += 1

    recent_180 = 0
    for papers in all_papers.values():
        for p in papers:
            pub = p.get("published")
            if pub and (now.date() - pub.date()).days <= 180:
                recent_180 += 1

    sb.add(f"- 近半年内发表论文 {recent_180} 篇，占总数 {total_papers} 篇的 {recent_180/max(total_papers,1)*100:.0f}%")
    sb.add(f"- 共覆盖 {len(sub_directions)} 个子方向，各方向论文分布如下：")
    for sub_dir, papers in all_papers.items():
        sb.add(f"  - {sub_dir}：{len(papers)} 篇")
    sb.blank()

    # ── Section: 存在问题 ──
    sb.heading(f"{sec_idx}. 存在问题")
    sb.blank()
    sec_idx += 1
    sb.add("1. **数据与基准**：各子方向使用的评测基准差异较大，缺乏统一标准。")
    sb.add("2. **可复现性**：部分论文未开源代码和模型，影响结果复现。")
    sb.add("3. **实际落地**：从学术方法到工业应用仍存在工程化和规模化挑战。")
    sb.add("4. **跨方向融合**：不同子方向的方法融合研究较少，存在割裂现象。")
    sb.blank()

    # ── Section: 未来方向 ──
    sb.heading(f"{sec_idx}. 未来方向")
    sb.blank()
    sec_idx += 1
    sb.add("1. **多模态融合**：将文本、图像、结构化数据等多模态信息融合建模。")
    sb.add("2. **效率与可扩展性**：面向大规模数据的高效训练和推理方法。")
    sb.add("3. **可解释性与可信赖性**：提升模型决策过程的透明度和可靠性。")
    sb.add("4. **跨领域迁移**：增强方法在不同领域间的零样本或少样本迁移能力。")
    sb.add("5. **安全与伦理**：建立负责任的研究框架，关注数据隐私和算法公平。")
    sb.blank()

    # ── Section: 参考文献 ──
    sb.heading(f"{sec_idx}. 参考文献")
    sb.blank()

    ref_id = 1
    for sub_dir, papers in all_papers.items():
        for p in papers[:5]:
            sb.add(
                f"[{ref_id}] {p['authors'].split(',')[0]} et al., "
                f"\"{p['title']}\", arXiv:{p['arxiv_id']}, {format_date(p.get('published'))}."
            )
            ref_id += 1

    sb.blank()
    sb.add("---")
    sb.add(f"*本报告由系统自动生成，共引用了 {ref_id - 1} 篇 arXiv 论文。未接入大语言模型，内容基于模板化组织。*")

    return sb.build()
