"""Research gap analyzer — rule-based gap detection and topic suggestion.

Analyzes papers, clustering results, and method distributions to identify:
- Research hotspots
- Method concentration areas
- Recurring problems/limitations
- Potential research gaps
- Feasible topic suggestions
"""

import re
from collections import Counter
from datetime import datetime


# ── Keyword banks ──────────────────────────────────────────────────────

_LIMITATION_KEYWORDS = [
    "limitation", "challenge", "future work", "open problem",
    "remain", "drawback", "weakness", "fail", "not yet",
    "further investigation", "need to be", "unexplored",
    "under-explored", "under explored", "bottleneck",
    "still lacks", "remains unclear", "unclear",
    "requires further", "more research", "needed",
    "remains an open", "remains a challenge",
    "not well understood", "poorly understood",
    "lack of", "limited by", "suffers from",
]

_GAP_INDICATORS = [
    "future work", "open problem", "remains to be", "not yet been",
    "little attention", "few studies", "limited research",
    "underexplored", "under-explored", "has not been",
    "needs further", "warrants further", "remains unclear",
    "not well understood", "lacks", "absence of",
]

_HOTSPOT_KEYWORDS = [
    "state-of-the-art", "novel", "transformer", "large language",
    "diffusion", "reinforcement learning", "graph neural",
    "self-supervised", "contrastive", "multi-modal",
    "federated", "prompt", "fine-tuning", "chain-of-thought",
    "retrieval-augmented", "generative", "adversarial",
]

_METHOD_KEYWORDS = [
    "propose", "framework", "architecture", "algorithm",
    "model", "approach", "method", "technique",
    "design", "introduce", "novel", "new method",
]

_EVALUATION_KEYWORDS = [
    "benchmark", "evaluate", "evaluation", "dataset",
    "experiment", "performance", "accuracy", "result",
    "compare", "outperform", "metric", "baseline",
    "test on", "validate", "ablation",
]

_APPLICATION_KEYWORDS = [
    "application", "deploy", "system", "real-world",
    "industry", "production", "applied to", "demo",
    "prototype", "practical", "tool", "platform",
    "open-source", "release", "available",
]

_CROSS_DIRECTION_KEYWORDS = [
    "combine", "integrate", "fusion", "multi-modal",
    "cross-modal", "cross-domain", "hybrid", "joint",
    "unified", "bridge", "synergy",
]


# ── Helpers ────────────────────────────────────────────────────────────

def _extract_sentences(text: str, keywords: list[str]) -> list[str]:
    """Return sentences containing any of the given keywords."""
    if not text:
        return []
    sents = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sents if any(kw.lower() in s.lower() for kw in keywords)]


def _categorize_paper(paper: dict) -> str:
    """Classify a paper as method / evaluation / application based on text."""
    text = f"{paper.get('title', '')} {paper.get('summary', '')}".lower()
    scores = {
        "method": sum(1 for kw in _METHOD_KEYWORDS if kw.lower() in text),
        "evaluation": sum(1 for kw in _EVALUATION_KEYWORDS if kw.lower() in text),
        "application": sum(1 for kw in _APPLICATION_KEYWORDS if kw.lower() in text),
    }
    norm = {
        k: scores[k] / len(kws) for k, kws in [
            ("method", _METHOD_KEYWORDS),
            ("evaluation", _EVALUATION_KEYWORDS),
            ("application", _APPLICATION_KEYWORDS),
        ]
    }
    best = max(norm, key=norm.get)
    return best if norm[best] > 0 else "other"


def _count_category(papers: list[dict]) -> dict[str, int]:
    counts = {"method": 0, "evaluation": 0, "application": 0, "other": 0}
    for p in papers:
        cat = _categorize_paper(p)
        counts[cat] += 1
    return counts


# ── Main analyzer ──────────────────────────────────────────────────────

def analyze_research_gaps(
    research_area: str,
    papers: list[dict],
    clustering_result: dict | None = None,
    sub_directions: list[str] | None = None,
    all_papers: dict[str, list[dict]] | None = None,
) -> dict:
    """Analyze research gaps and generate topic suggestions.

    Parameters
    ----------
    research_area : str
        The research area being studied.
    papers : list[dict]
        All deduplicated papers across all sub-directions.
    clustering_result : dict | None
        Output from topic_clusterer.cluster_papers().
    sub_directions : list[str] | None
        Sub-direction names.
    all_papers : dict[str, list[dict]] | None
        Per-sub-direction paper lists.

    Returns
    -------
    dict with keys:
      hotspots, method_distribution, recurring_problems,
      potential_gaps, topic_suggestions, gap_report (markdown).
    """
    n = len(papers)
    if n == 0:
        return _empty_result(research_area)

    clusters = clustering_result.get("clusters", []) if clustering_result else []
    sub_dirs = sub_directions or []
    dir_papers = all_papers or {}

    # ── 1. Research hotspots ──
    hotspots = _detect_hotspots(papers, clusters)

    # ── 2. Method distribution ──
    method_dist = _analyze_method_distribution(papers)

    # ── 3. Recurring problems ──
    recurring_problems = _extract_recurring_problems(papers)

    # ── 4. Potential research gaps ──
    potential_gaps = _detect_gaps(papers, clusters, method_dist, sub_dirs, dir_papers)

    # ── 5. Topic suggestions ──
    topic_suggestions = _generate_suggestions(
        research_area, hotspots, method_dist, recurring_problems, potential_gaps, clusters
    )

    # ── 6. Build markdown report ──
    gap_report = _build_gap_report(
        research_area, hotspots, method_dist, recurring_problems,
        potential_gaps, topic_suggestions, n
    )

    return {
        "hotspots": hotspots,
        "method_distribution": method_dist,
        "recurring_problems": recurring_problems,
        "potential_gaps": potential_gaps,
        "topic_suggestions": topic_suggestions,
        "gap_report": gap_report,
    }


# ── Sub-analyzers ──────────────────────────────────────────────────────

def _detect_hotspots(papers: list[dict], clusters: list[dict]) -> dict:
    """Identify current research hotspots."""
    all_text = " ".join(p.get("title", "") + " " + p.get("summary", "") for p in papers)

    # Count hotspot keyword hits
    kw_hits = Counter()
    for kw in _HOTSPOT_KEYWORDS:
        count = all_text.lower().count(kw.lower())
        if count > 0:
            kw_hits[kw] = count

    # Top clusters by paper count → hotspot themes
    hot_clusters = []
    for c in sorted(clusters, key=lambda x: x.get("paper_count", 0), reverse=True)[:3]:
        if c.get("paper_count", 0) > 0:
            hot_clusters.append({
                "theme": c.get("topic_name", "Unknown"),
                "paper_count": c.get("paper_count", 0),
                "keywords": c.get("keywords", []),
            })

    # Recent papers (last 6 months)
    now = datetime.now()
    recent_titles = []
    for p in papers:
        pub = p.get("published")
        days = 999
        if pub and hasattr(pub, 'date'):
            days = (now.date() - pub.date()).days
        elif isinstance(pub, str) and len(pub) >= 10:
            try:
                pub_date = datetime.strptime(pub[:10], "%Y-%m-%d").date()
                days = (now.date() - pub_date).days
            except Exception:
                pass

        if days <= 180:
            recent_titles.append(p.get("title", ""))

    return {
        "top_keywords": [kw for kw, _ in kw_hits.most_common(8)],
        "hot_themes": hot_clusters,
        "recent_paper_count": len(recent_titles),
        "recent_sample": recent_titles[:3],
    }


def _analyze_method_distribution(papers: list[dict]) -> dict:
    """Analyze the distribution of paper types and method families."""
    cat_counts = _count_category(papers)
    total = max(sum(cat_counts.values()), 1)

    method_families = {
        "Transformer / Attention": ["transformer", "attention", "self-attention"],
        "Diffusion Models": ["diffusion", "denoising", "ddpm", "score-based"],
        "Reinforcement Learning": ["reinforcement", "policy gradient", "q-learning", "actor-critic"],
        "Graph Neural Networks": ["graph neural", "gnn", "gcn", "message passing"],
        "Contrastive Learning": ["contrastive", "simclr", "moco"],
        "Self-Supervised Learning": ["self-supervised", "pretext", "masked"],
        "LLM / Fine-tuning": ["large language", "llm", "fine-tun", "lora", "prompt"],
        "Retrieval-Augmented": ["retrieval-augment", "rag", "retriever"],
    }

    all_text = " ".join(p.get("title", "") + " " + p.get("summary", "") for p in papers).lower()
    family_counts = {}
    for family, kws in method_families.items():
        count = sum(all_text.count(kw.lower()) for kw in kws)
        if count > 0:
            family_counts[family] = count

    return {
        "category_distribution": {
            "method": {"count": cat_counts["method"], "pct": round(cat_counts["method"] / total * 100, 1)},
            "evaluation": {"count": cat_counts["evaluation"], "pct": round(cat_counts["evaluation"] / total * 100, 1)},
            "application": {"count": cat_counts["application"], "pct": round(cat_counts["application"] / total * 100, 1)},
            "other": {"count": cat_counts["other"], "pct": round(cat_counts["other"] / total * 100, 1)},
        },
        "method_families": dict(sorted(family_counts.items(), key=lambda x: x[1], reverse=True)[:6]),
        "total_papers": total,
    }


def _extract_recurring_problems(papers: list[dict]) -> dict:
    """Extract recurring problems and limitations from paper abstracts."""
    all_problems = []
    for p in papers:
        summary = p.get("summary", "") or ""
        sents = _extract_sentences(summary, _LIMITATION_KEYWORDS)
        for s in sents:
            all_problems.append({
                "text": s[:200],
                "title": p.get("title", ""),
                "link": p.get("link", ""),
            })

    # Group similar problems by keyword
    problem_groups: dict[str, list[dict]] = {}
    for prob in all_problems:
        text_lower = prob["text"].lower()
        matched_kws = [kw for kw in _LIMITATION_KEYWORDS if kw.lower() in text_lower]
        key = matched_kws[0] if matched_kws else "other"
        if key not in problem_groups:
            problem_groups[key] = []
        problem_groups[key].append(prob)

    sorted_groups = sorted(problem_groups.items(), key=lambda x: len(x[1]), reverse=True)

    return {
        "total_mentions": len(all_problems),
        "problem_categories": [
            {
                "category": kw,
                "count": len(items),
                "samples": items[:3],
            }
            for kw, items in sorted_groups[:8]
        ],
    }


def _detect_gaps(
    papers: list[dict],
    clusters: list[dict],
    method_dist: dict,
    sub_dirs: list[str],
    dir_papers: dict[str, list[dict]],
) -> list[dict]:
    """Detect potential research gaps based on multiple heuristic rules."""
    gaps: list[dict] = []
    n = len(papers)

    # ── Rule 1: Small clusters → underexplored niches ──
    if clusters and n >= 10:
        avg_size = n / max(len(clusters), 1)
        small_threshold = max(1, int(avg_size * 0.5))
        for c in clusters:
            cnt = c.get("paper_count", 0)
            if 0 < cnt <= small_threshold:
                gaps.append({
                    "type": "underexplored_niche",
                    "theme": f"未充分探索方向: {c.get('topic_name', 'Unknown')}",
                    "description": (
                        f"主题「{c['topic_name']}」仅有 {cnt} 篇论文，"
                        f"远低于平均簇大小 {avg_size:.0f} 篇，"
                        f"可能是一个尚未被充分探索的研究方向。"
                    ),
                    "keywords": c.get("keywords", []),
                    "paper_count": cnt,
                    "confidence": "中",
                })

    # ── Rule 2: Category imbalance ──
    cats = method_dist.get("category_distribution", {})
    method_pct = cats.get("method", {}).get("pct", 0)
    eval_pct = cats.get("evaluation", {}).get("pct", 0)
    app_pct = cats.get("application", {}).get("pct", 0)

    if eval_pct > 40 and method_pct < 25:
        gaps.append({
            "type": "method_innovation_gap",
            "theme": "方法创新空间充足",
            "description": (
                f"评测基准类论文占比 {eval_pct}%，而方法创新类论文仅占 {method_pct}%。"
                f"大量工作集中在评测现有方法上，该领域存在明确的方法创新空白。"
            ),
            "keywords": ["method innovation", "novel architecture", "algorithm design"],
            "confidence": "高",
        })

    if method_pct > 35 and app_pct < 15:
        gaps.append({
            "type": "application_gap",
            "theme": "方法到应用转化鸿沟",
            "description": (
                f"方法创新类论文占比 {method_pct}%，但应用系统类仅占 {app_pct}%。"
                f"理论方法与实际应用之间存在明显鸿沟，落地转化需求突出。"
            ),
            "keywords": ["application", "deployment", "system", "real-world"],
            "confidence": "高",
        })

    if eval_pct > 35 and app_pct < 15:
        if not any(g.get("type") == "application_gap" for g in gaps):
            gaps.append({
                "type": "benchmark_application_gap",
                "theme": "基准测试到实际应用转化不足",
                "description": (
                    f"评测基准类论文占比 {eval_pct}%，但实际应用类仅占 {app_pct}%。"
                    f"大量基准测试结果未转化为实际系统，存在从实验室到工业场景的转化空白。"
                ),
                "keywords": ["benchmark to application", "real-world deployment", "industry"],
                "confidence": "中",
            })

    # ── Rule 3: Cross-direction gaps ──
    if sub_dirs and len(sub_dirs) >= 2 and dir_papers:
        cross_count = 0
        for p in papers:
            text = f"{p.get('title', '')} {p.get('summary', '')}".lower()
            if any(kw.lower() in text for kw in _CROSS_DIRECTION_KEYWORDS):
                cross_count += 1

        cross_pct = cross_count / max(n, 1) * 100
        if cross_pct < 25:
            dir_names = "、".join(sub_dirs[:3])
            gaps.append({
                "type": "cross_direction_gap",
                "theme": "跨方向融合研究不足",
                "description": (
                    f"仅 {cross_pct:.0f}% 的论文涉及跨方向融合研究。"
                    f"「{dir_names}」等子方向之间存在潜在交叉点，"
                    f"跨方向融合可能产生创新性成果。"
                ),
                "keywords": ["cross-direction", "multi-modal", "hybrid", "integration"],
                "confidence": "中",
            })

    # ── Rule 4: Keyword-based gap signals in literature ──
    gap_signals = _scan_for_gap_signals(papers)
    for gs in gap_signals[:3]:
        gaps.append({
            "type": "literature_gap",
            "theme": f"文献指出的空白: {gs['phrase'][:40]}...",
            "description": (
                f"多篇论文摘要中提到类似的研究空白："
                f"「{gs['phrase'][:150]}」。"
                f"该方向被 {gs['count']} 篇论文提及但未深入探索。"
            ),
            "keywords": gs.get("keywords", []),
            "confidence": "低" if gs["count"] < 3 else "中",
        })

    # Fallback
    if not gaps:
        gaps.append({
            "type": "general_opportunity",
            "theme": "综合研究机会",
            "description": (
                f"在 {n} 篇论文的系统分析中，各方向分布较为均衡。"
                f"建议从方法创新、跨方向融合和实际应用落地三个维度寻找切入点。"
            ),
            "keywords": [],
            "confidence": "低",
        })

    return gaps


def _scan_for_gap_signals(papers: list[dict]) -> list[dict]:
    """Scan paper abstracts for explicit gap/opportunity signals."""
    phrase_counter: Counter = Counter()
    phrase_examples: dict[str, list[str]] = {}

    for p in papers:
        summary = p.get("summary", "") or ""
        sents = _extract_sentences(summary, _GAP_INDICATORS)
        for s in sents:
            s_lower = s.lower()
            for indicator in _GAP_INDICATORS:
                if indicator.lower() in s_lower:
                    idx = s_lower.find(indicator.lower())
                    start = max(0, idx - 50)
                    end = min(len(s), idx + len(indicator) + 100)
                    snippet = s[start:end].strip()
                    phrase_counter[indicator] += 1
                    if indicator not in phrase_examples:
                        phrase_examples[indicator] = []
                    phrase_examples[indicator].append(snippet)

    results = []
    for phrase, count in phrase_counter.most_common(8):
        if count >= 2:
            examples = phrase_examples.get(phrase, [phrase])
            results.append({
                "phrase": examples[0],
                "count": count,
                "keywords": [phrase],
            })
    return results


def _generate_suggestions(
    research_area: str,
    hotspots: dict,
    method_dist: dict,
    recurring_problems: dict,
    gaps: list[dict],
    clusters: list[dict],
) -> list[dict]:
    """Generate feasible topic suggestions from all analysis signals."""
    suggestions: list[dict] = []
    sid = 1

    # ── From hot themes → ride the wave ──
    hot_themes = hotspots.get("hot_themes", [])
    if hot_themes:
        top_theme = hot_themes[0]
        suggestions.append({
            "id": sid,
            "direction": f"{top_theme['theme']} 方向的深入优化",
            "rationale": (
                f"「{top_theme['theme']}」是当前最活跃的研究主题"
                f"（{top_theme['paper_count']} 篇论文），"
                f"但仍有优化空间，可从效率、精度或可解释性角度切入。"
            ),
            "feasibility": "高",
            "innovation_level": "渐进式创新",
            "estimated_effort": "中",
        })
        sid += 1

    # ── From gaps → targeted suggestions ──
    for gap in gaps:
        if gap.get("type") == "method_innovation_gap":
            suggestions.append({
                "id": sid,
                "direction": f"面向{research_area}的新方法/新架构设计",
                "rationale": (
                    "当前领域评测工作占主导但方法创新不足。"
                    "可借鉴 NLP、CV 等领域的最新架构，结合领域特点设计新型方法。"
                ),
                "feasibility": "中",
                "innovation_level": "突破性创新",
                "estimated_effort": "高",
            })
            sid += 1

        elif gap.get("type") == "application_gap":
            suggestions.append({
                "id": sid,
                "direction": f"{research_area}方法在具体场景中的落地应用",
                "rationale": (
                    "现有方法缺乏真实场景验证。可选取一个具体应用场景，"
                    "将前沿方法适配落地，输出可复现的系统和最佳实践。"
                ),
                "feasibility": "高",
                "innovation_level": "应用创新",
                "estimated_effort": "中",
            })
            sid += 1

        elif gap.get("type") == "cross_direction_gap":
            suggestions.append({
                "id": sid,
                "direction": "跨子方向融合的混合方法研究",
                "rationale": (
                    "各子方向相对独立，跨方向融合可能产生新突破。"
                    "建议选择 2-3 个互补子方向，设计统一框架或混合模型。"
                ),
                "feasibility": "中",
                "innovation_level": "突破性创新",
                "estimated_effort": "高",
            })
            sid += 1

        elif gap.get("type") == "underexplored_niche":
            theme = gap.get("theme", "Unknown").replace("未充分探索方向: ", "")
            suggestions.append({
                "id": sid,
                "direction": f"探索性研究：{theme}",
                "rationale": gap.get("description", ""),
                "feasibility": "低",
                "innovation_level": "突破性创新",
                "estimated_effort": "高",
            })
            sid += 1

    # ── From recurring problems ──
    problems = recurring_problems.get("problem_categories", [])
    if problems:
        top = problems[0]
        suggestions.append({
            "id": sid,
            "direction": f"针对「{top['category']}」问题的系统性解决方案",
            "rationale": (
                f"「{top['category']}」是论文中反复提及的核心挑战"
                f"（{top['count']} 次提及），"
                f"提出系统性解决方案具有明确的学术价值。"
            ),
            "feasibility": "中",
            "innovation_level": "渐进式创新",
            "estimated_effort": "中",
        })
        sid += 1

    # ── From method family combinations ──
    families = method_dist.get("method_families", {})
    family_names = list(families.keys())
    if len(family_names) >= 2:
        suggestions.append({
            "id": sid,
            "direction": f"{family_names[0]} 与 {family_names[1]} 的融合方法",
            "rationale": (
                f"「{family_names[0]}」和「{family_names[1]}」是排名前二的技术路线，"
                "将两者结合可能产生互补优势，值得探索。"
            ),
            "feasibility": "中",
            "innovation_level": "突破性创新",
            "estimated_effort": "高",
        })
        sid += 1

    # ── Fill generic suggestions if too few ──
    generic_pool = [
        {
            "direction": f"构建面向{research_area}的统一评测基准",
            "rationale": "缺乏统一评测标准是该领域常见问题，建立权威基准具有长期价值。",
            "feasibility": "高",
            "innovation_level": "基础设施",
            "estimated_effort": "中",
        },
        {
            "direction": f"{research_area}的可解释性与可信赖性研究",
            "rationale": "随着方法复杂度提升，可解释性成为学术界和工业界的共同需求。",
            "feasibility": "中",
            "innovation_level": "渐进式创新",
            "estimated_effort": "中",
        },
        {
            "direction": f"{research_area}的高效训练与推理方法",
            "rationale": "计算效率是大规模应用的核心瓶颈，优化训练/推理成本有实际价值。",
            "feasibility": "中",
            "innovation_level": "渐进式创新",
            "estimated_effort": "中",
        },
    ]

    existing_dirs = {s["direction"] for s in suggestions}
    for gs in generic_pool:
        if len(suggestions) >= 5:
            break
        if gs["direction"] not in existing_dirs:
            gs["id"] = sid
            suggestions.append(gs)
            sid += 1

    return suggestions[:5]


# ── Report builder ─────────────────────────────────────────────────────

def _build_gap_report(
    research_area: str,
    hotspots: dict,
    method_dist: dict,
    recurring_problems: dict,
    gaps: list[dict],
    suggestions: list[dict],
    total_papers: int,
) -> str:
    """Build a markdown report section for research gaps and suggestions."""
    lines: list[str] = []
    confidence_emoji = {"高": "🟢", "中": "🟡", "低": "🔴"}

    lines.append("---")
    lines.append("")
    lines.append("## 研究空白与选题建议")
    lines.append("")
    lines.append(f"*基于 {total_papers} 篇论文的系统分析*")
    lines.append("")

    # ── 1. Hotspots ──
    lines.append("### 🔥 当前研究热点")
    lines.append("")

    hot_kw = hotspots.get("top_keywords", [])
    if hot_kw:
        lines.append(f"高频技术关键词：**{'、'.join(hot_kw)}**")
        lines.append("")

    hot_themes = hotspots.get("hot_themes", [])
    if hot_themes:
        lines.append("热门主题簇：")
        lines.append("")
        for t in hot_themes:
            kws = t.get("keywords", [])
            lines.append(
                f"- **{t['theme']}**（{t['paper_count']} 篇）"
                f"{' — ' + '、'.join(kws[:5]) if kws else ''}"
            )
        lines.append("")

    recent_count = hotspots.get("recent_paper_count", 0)
    recent_pct = recent_count / max(total_papers, 1) * 100
    activity = "较高" if recent_pct >= 30 else "一般"
    lines.append(f"近半年内发表 **{recent_count}** 篇（{recent_pct:.0f}%），研究活跃度**{activity}**。")
    lines.append("")

    # ── 2. Method distribution ──
    lines.append("### 📊 方法分布与集中度分析")
    lines.append("")

    cats = method_dist.get("category_distribution", {})
    lines.append("| 类别 | 论文数 | 占比 |")
    lines.append("|------|--------|------|")
    cat_labels = {"method": "方法创新", "evaluation": "评测基准", "application": "应用系统", "other": "其他"}
    for cat_name in ["method", "evaluation", "application", "other"]:
        if cat_name in cats:
            cd = cats[cat_name]
            lines.append(f"| {cat_labels.get(cat_name, cat_name)} | {cd['count']} | {cd['pct']}% |")
    lines.append("")

    families = method_dist.get("method_families", {})
    if families:
        lines.append("主要技术路线：")
        lines.append("")
        for family, count in families.items():
            lines.append(f"- **{family}**：{count} 次提及")
        lines.append("")

    # Imbalance insights
    method_pct = cats.get("method", {}).get("pct", 0)
    eval_pct = cats.get("evaluation", {}).get("pct", 0)
    app_pct = cats.get("application", {}).get("pct", 0)

    if eval_pct > method_pct * 1.5:
        lines.append(
            f"> ⚡ 评测基准论文（{eval_pct}%）远超方法创新（{method_pct}%），"
            f"该领域**方法创新空间充足**。"
        )
        lines.append("")
    if method_pct > app_pct * 2:
        lines.append(
            f"> ⚡ 方法类论文（{method_pct}%）远超应用系统（{app_pct}%），"
            f"存在明显的**落地转化鸿沟**。"
        )
        lines.append("")

    # ── 3. Recurring problems ──
    lines.append("### ⚠️ 反复出现的问题与挑战")
    lines.append("")

    problems = recurring_problems.get("problem_categories", [])
    if problems:
        total_mentions = recurring_problems.get("total_mentions", 0)
        lines.append(f"从 {total_mentions} 处问题/局限描述中，归纳出以下高频挑战：")
        lines.append("")
        for i, prob in enumerate(problems[:6], 1):
            lines.append(f"**{i}. {prob['category']}**（{prob['count']} 次提及）")
            for sample in prob.get("samples", [])[:1]:
                snippet = sample['text'][:150]
                title = sample['title'][:80]
                link = sample.get('link', '#')
                lines.append(f"   > *\"{snippet}...\"*")
                lines.append(f"   > — [{title}]({link})")
            lines.append("")
    else:
        lines.append("未从摘要中提取到明确的共性问题，可能需要全文分析。")
        lines.append("")

    # ── 4. Potential gaps ──
    lines.append("### 🕳️ 潜在研究空白")
    lines.append("")

    for i, gap in enumerate(gaps, 1):
        conf = gap.get("confidence", "低")
        emoji = confidence_emoji.get(conf, "⚪")
        lines.append(f"**{i}. {gap['theme']}**  {emoji} 可信度：{conf}")
        lines.append("")
        lines.append(f"   {gap['description']}")
        kws = gap.get("keywords", [])
        if kws:
            lines.append(f"   - 关联词：{'、'.join(kws[:5])}")
        lines.append("")

    # ── 5. Topic suggestions ──
    lines.append("### 💡 可行选题建议")
    lines.append("")

    lines.append("| # | 选题方向 | 可行性 | 创新级别 | 预估投入 |")
    lines.append("|---|---------|--------|---------|---------|")
    for s in suggestions:
        lines.append(
            f"| {s['id']} | {s['direction']} | "
            f"{s['feasibility']} | {s['innovation_level']} | "
            f"{s['estimated_effort']} |"
        )
    lines.append("")

    for s in suggestions:
        lines.append(f"**选题 {s['id']}：{s['direction']}**")
        lines.append("")
        lines.append(f"- **理由**：{s['rationale']}")
        lines.append(
            f"- **可行性**：{s['feasibility']}　|　"
            f"**创新级别**：{s['innovation_level']}　|　"
            f"**预估投入**：{s['estimated_effort']}"
        )
        lines.append("")

    lines.append("---")
    lines.append("*本分析基于规则与模板生成，未接入大语言模型，仅供参考。请结合领域知识和导师意见进行判断。*")

    return "\n".join(lines)


def _empty_result(research_area: str) -> dict:
    return {
        "hotspots": {"top_keywords": [], "hot_themes": [], "recent_paper_count": 0, "recent_sample": []},
        "method_distribution": {
            "category_distribution": {}, "method_families": {}, "total_papers": 0,
        },
        "recurring_problems": {"total_mentions": 0, "problem_categories": []},
        "potential_gaps": [],
        "topic_suggestions": [],
        "gap_report": (
            f"## 研究空白与选题建议\n\n"
            f"未能检索到 {research_area} 相关论文，无法进行分析。\n"
        ),
    }
