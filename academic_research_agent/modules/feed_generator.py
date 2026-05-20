"""Feed generator — daily digest and deep paper interpretation."""

from datetime import datetime

from modules.utils import format_date


def generate_daily_digest(
    interest_directions: list[str],
    all_papers: dict[str, list[dict]],
    top_papers: list[dict],
    top_n: int = 10,
) -> str:
    """
    Generate a daily academic progress report.

    all_papers: {direction: [papers], ...} per direction results.
    top_papers: globally ranked top papers.
    """
    now = datetime.now()
    total = sum(len(v) for v in all_papers.values())

    lines = [
        f"# 今日学术进展日报",
        "",
        f"*生成时间：{format_date(now)}*",
        "",
        "---",
        "",
        "## 关注方向",
        "",
    ]
    for i, d in enumerate(interest_directions, 1):
        lines.append(f"{i}. {d}")
    lines.append("")

    # Per-direction papers
    lines.append("## 各方向最新论文")
    lines.append("")
    for direction, papers in all_papers.items():
        lines.append(f"### {direction}（{len(papers)} 篇）")
        lines.append("")
        if not papers:
            lines.append("*该方向今日未检索到新论文。*")
            lines.append("")
            continue
        for p in papers[:5]:
            authors_short = ", ".join(p["authors"].split(",")[:2])
            lines.append(
                f"- **{p['title']}** — {authors_short} et al. "
                f"({format_date(p.get('published'))}), "
                f"[arXiv]({p['link']})"
            )
        if len(papers) > 5:
            lines.append(f"  *...及其他 {len(papers) - 5} 篇*")
        lines.append("")

    # Top 10
    lines.append("---")
    lines.append("")
    lines.append("## 今日推荐 Top 10")
    lines.append("")
    lines.append("| 排名 | 论文标题 | 综合得分 | 发表日期 |")
    lines.append("|------|---------|---------|---------|")
    for i, p in enumerate(top_papers[:top_n], 1):
        title_short = p["title"][:80] + ("..." if len(p["title"]) > 80 else "")
        lines.append(
            f"| {i} | [{title_short}]({p['link']}) | "
            f"{p.get('composite_score', 0):.3f} | "
            f"{format_date(p.get('published'))} |"
        )
    lines.append("")

    # Trend summary
    lines.append("---")
    lines.append("")
    lines.append("## 今日学术趋势综述")
    lines.append("")

    recent_7 = sum(
        1 for p in top_papers
        if p.get("published") and (now.date() - p["published"].date()).days <= 7
    )
    recent_30 = sum(
        1 for p in top_papers
        if p.get("published") and (now.date() - p["published"].date()).days <= 30
    )

    lines.append(f"本次检索共覆盖 {len(interest_directions)} 个研究方向，")
    lines.append(f"经去重后共获得 {total} 篇论文。")
    lines.append(f"近 7 天内发表 {recent_7} 篇，近 30 天内发表 {recent_30} 篇。")

    # Identify hot topics
    all_titles = " ".join(p["title"] for p in top_papers)
    hot_terms = []
    for term in [
        "LLM", "large language", "diffusion", "transformer", "graph neural",
        "reinforcement learning", "contrastive", "multi-modal", "federated",
        "prompt", "fine-tuning", "retrieval-augmented", "chain-of-thought",
    ]:
        if term.lower() in all_titles.lower():
            hot_terms.append(term)

    if hot_terms:
        lines.append("")
        lines.append(f"热门技术词：**{', '.join(hot_terms[:6])}**。")

    # Cross-direction observation
    if len(interest_directions) >= 2:
        lines.append("")
        lines.append(
            f"您关注的 {len(interest_directions)} 个方向之间存在潜在交叉，"
            f"建议关注跨方向融合研究的进展。"
        )

    lines.append("")
    lines.append("---")
    lines.append(f"*本日报由系统基于 {total} 篇 arXiv 论文自动生成，未接入大语言模型。*")

    return "\n".join(lines)


def generate_deep_interpretation(
    paper: dict,
    rank: int,
) -> str:
    """
    Generate a deep interpretation for a single paper.
    Template-based, no LLM required.
    """
    lines = [
        f"### Top {rank} 深度解读",
        "",
        f"**标题**：{paper['title']}",
        "",
        f"**作者**：{paper['authors']}",
        "",
        f"**发表时间**：{format_date(paper.get('published'))}",
        "",
        f"**arXiv**：[{paper['arxiv_id']}]({paper['link']})",
        "",
        "---",
        "",
        "**研究问题**",
        "",
    ]

    summary = paper.get("summary") or "No abstract available."
    summary_lower = summary.lower()
    # Extract first sentence as research problem
    first_sent = summary.split(".")[0] + "."
    lines.append(f"该论文研究 {first_sent[:200]}")

    lines.append("")
    lines.append("**方法概述**")

    # Try to identify method keywords
    method_kw = []
    for kw in [
        "transformer", "neural network", "attention", "diffusion", "reinforcement",
        "fine-tun", "pre-train", "self-supervised", "contrastive", "graph",
        "encoder", "decoder", "optimization", "gradient", "loss function",
        "propose", "introduce", "novel", "framework", "architecture",
    ]:
        if kw.lower() in summary_lower:
            method_kw.append(kw)

    if method_kw:
        lines.append(f"论文涉及的关键技术方法包括：{', '.join(method_kw)}。")
    lines.append(f"详细摘要：{summary[:500]}...")

    lines.append("")
    lines.append("**创新点分析**")

    if "novel" in summary_lower or "propose" in summary_lower:
        lines.append("- 论文提出了新的方法或框架，具有一定的创新性。")
    if "state-of-the-art" in summary_lower or "outperform" in summary_lower:
        lines.append("- 在实验中取得了领先性能，具有较强竞争力。")
    if "benchmark" in summary_lower:
        lines.append("- 在标准基准数据集上进行了验证。")
    if not any(kw in summary_lower for kw in ["novel", "propose", "state-of-the-art", "outperform", "benchmark"]):
        lines.append("- 论文在所研究问题上做出了有价值的贡献。")

    lines.append("")
    lines.append("**相关性分析**")

    composite = paper.get("composite_score", 0)
    relevance = paper.get("relevance_score", 0)
    freshness = paper.get("freshness_score", 0)
    quality = paper.get("quality_score", 0)

    lines.append(f"- 综合得分：{composite:.3f}（相关度 {relevance:.3f} + 新鲜度 {freshness:.3f} + 质量分 {quality:.3f}）")

    if relevance >= 0.3:
        lines.append("- 该论文与您的研究方向高度相关，建议优先阅读。")
    elif relevance >= 0.1:
        lines.append("- 该论文与您的研究方向有一定相关性，可作为参考阅读。")
    else:
        lines.append("- 该论文提供了不同的研究视角，扩展阅读可能有所启发。")

    lines.append("")
    lines.append("**阅读建议**")
    lines.append("")
    lines.append(f"1. 首先阅读摘要和引言，了解研究动机和主要贡献。")
    lines.append(f"2. 关注方法部分的核心算法和实验设计。")
    lines.append(f"3. 对比作者提出的方法与已有方法的差异。")
    lines.append(f"4. 阅读论文链接：[{paper['arxiv_id']}]({paper['link']})")

    return "\n".join(lines)
