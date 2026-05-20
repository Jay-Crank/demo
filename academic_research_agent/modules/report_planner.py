"""Report planner — builds data-driven section plans.

Selects papers for each of the 10 standard sections based on:
- Citation scores (研究背景)
- Limitation/challenge keywords (核心问题)
- Method-type classification (主要技术路线, 方法对比矩阵)
- Topic clusters (研究方向图谱)
- Composite scores (代表性工作)
- Freshness scores (最新进展)
- Benchmark/metric overlap analysis (矛盾与争议)
- Gap signals (研究空白)
- Topic suggestions (选题建议)
"""

import re
from collections import Counter, defaultdict
from datetime import datetime

from modules.section_templates import (
    ALL_SECTION_TEMPLATES,
    SEC_BACKGROUND,
    SEC_CORE_PROBLEMS,
    SEC_TECHNICAL_ROUTES,
    SEC_TOPIC_MAP,
    SEC_REPRESENTATIVE,
    SEC_LATEST,
    SEC_COMPARISON_MATRIX,
    SEC_CONTROVERSIES,
    SEC_RESEARCH_GAPS,
    SEC_TOPIC_SUGGESTIONS,
)
from modules.evidence_builder import build_evidence_pack, format_evidence_pack_for_prompt


# ── Paper scoring helpers ────────────────────────────────────────────────

def _ensure_scores(papers: list[dict]) -> list[dict]:
    """Make sure papers have score fields; compute defaults if missing."""
    for p in papers:
        if "composite_score" not in p:
            p["composite_score"] = p.get("relevance_score", 0.5)
        if "citation_score" not in p:
            p["citation_score"] = p.get("citation_count", 0) / max(
                max((pp.get("citation_count", 0) or 0) for pp in papers), 1
            )
        if "freshness_score" not in p:
            p["freshness_score"] = 0.5
    return papers


def _sort_by(papers: list[dict], key: str, reverse: bool = True) -> list[dict]:
    return sorted(papers, key=lambda p: p.get(key, 0) or 0, reverse=reverse)


# ── Method classification ────────────────────────────────────────────────

_METHOD_FAMILIES = {
    "Transformer / Attention": ["transformer", "attention", "self-attention", "multi-head"],
    "Diffusion Models": ["diffusion", "denoising", "ddpm", "score-based", "ddim"],
    "Reinforcement Learning": ["reinforcement", "policy gradient", "q-learning", "actor-critic", "ppo"],
    "Graph Neural Networks": ["graph neural", "gnn", "gcn", "message passing", "graph attention"],
    "Contrastive Learning": ["contrastive", "simclr", "moco", "simsiam"],
    "Self-Supervised Learning": ["self-supervised", "pretext", "masked", "mae"],
    "LLM / Fine-tuning": ["large language", "llm", "fine-tun", "lora", "prompt", "instruction"],
    "Retrieval-Augmented": ["retrieval-augment", "rag", "retriever", "dense retrieval"],
    "Generative Models": ["generative", "gan", "vae", "generation", "llm"],
    "Optimization Methods": ["optimization", "gradient", "sgd", "adam", "learning rate"],
    "Multi-Modal": ["multi-modal", "cross-modal", "vision-language", "text-image"],
    "Federated Learning": ["federated", "decentralized", "privacy-preserving"],
    "Neural Architecture Search": ["nas", "architecture search", "automl"],
    "Knowledge Distillation": ["distillation", "teacher-student", "knowledge transfer"],
    "Chain-of-Thought": ["chain-of-thought", "reasoning", "cot"],
}


def _classify_paper_to_methods(paper: dict) -> list[str]:
    """Return method family names that this paper belongs to."""
    text = f"{paper.get('title', '')} {paper.get('abstract', '')} {paper.get('summary', '')}".lower()
    matched = []
    for family, keywords in _METHOD_FAMILIES.items():
        if any(kw.lower() in text for kw in keywords):
            matched.append(family)
    return matched


def build_comparison_matrix(papers: list[dict]) -> dict:
    """Build method comparison matrix from papers.

    Returns dict with:
        method_types: list of {name, keywords, papers, paper_count, aggregated_metrics}
        total_papers: int
    """
    papers = _ensure_scores(papers)
    method_map: dict[str, list[dict]] = defaultdict(list)

    for p in papers:
        families = _classify_paper_to_methods(p)
        for fam in families:
            method_map[fam].append(p)

    method_types = []
    for name, keywords in _METHOD_FAMILIES.items():
        members = method_map.get(name, [])
        if not members:
            continue
        # Sort by composite score
        members = _sort_by(members, "composite_score")
        # Aggregate metrics from metric_mentions or abstract
        agg_metrics: list[str] = []
        for p in members:
            mm = p.get("metric_mentions", [])
            if not mm:
                abstract = p.get("abstract", "") or p.get("summary", "")
                mm = _extract_metric_mentions(abstract)
            agg_metrics.extend(mm[:3])
        method_types.append({
            "name": name,
            "keywords": keywords,
            "papers": members,
            "paper_count": len(members),
            "aggregated_metrics": list(set(agg_metrics))[:10],
        })

    # Sort by paper count
    method_types.sort(key=lambda m: m["paper_count"], reverse=True)

    return {
        "method_types": method_types,
        "total_papers": len(papers),
    }


def _extract_metric_mentions(text: str) -> list[str]:
    """Quick metric extraction for comparison matrix building."""
    if not text:
        return []
    pat = re.compile(
        r"\d{1,3}(?:\.\d{1,2})?\s*%|"
        r"(?:accuracy|precision|recall|F1|AUC|BLEU|ROUGE|RMSE|MAE|perplexity)"
        r"\s*(?:of|:|is|was|at)?\s*\d{1,3}(?:\.\d{1,4})?",
        re.IGNORECASE,
    )
    return [m.group().strip() for m in pat.finditer(text)][:10]


# ── Contradiction signal detector ────────────────────────────────────────

# Common benchmark/dataset names
_BENCHMARK_NAMES = [
    "ImageNet", "CIFAR-10", "CIFAR-100", "MNIST", "COCO", "Pascal VOC",
    "GLUE", "SuperGLUE", "SQuAD", "MMLU", "HumanEval", "GSM8K",
    "WikiText", "PTB", "WMT", "BoolQ", "HellaSwag", "ARC",
    "TruthfulQA", "MT-Bench", "AlpacaEval", "Chatbot Arena",
    "PubMed", "OGB", "Freebase", "DBpedia", "YAGO",
    "Cityscapes", "nuScenes", "Waymo", "KITTI",
]


def _detect_contradiction_signals(
    papers: list[dict],
    comparison_matrix: dict,
) -> dict:
    """Detect potential contradiction signals across papers.

    Rules:
    1. Same benchmark/task with different metric values (>5% gap)
    2. Opposite conclusions on same research question

    Returns dict with:
        signals: list of {benchmark, papers, values, gap_description}
        has_signals: bool
    """
    signals: list[dict] = []
    all_text = " ".join(
        p.get("title", "") + " " + p.get("abstract", "") + " " + p.get("summary", "")
        for p in papers
    ).lower()

    # Find benchmarks mentioned by multiple papers
    bench_papers: dict[str, list[dict]] = defaultdict(list)
    for p in papers:
        text = f"{p.get('title', '')} {p.get('abstract', '')} {p.get('summary', '')}".lower()
        for bench in _BENCHMARK_NAMES:
            if bench.lower() in text:
                bench_papers[bench].append(p)

    # For each benchmark, try to find metric values
    for bench, bp_list in bench_papers.items():
        if len(bp_list) < 2:
            continue
        # Extract metric values near the benchmark mention
        values: list[tuple[float, dict]] = []
        for p in bp_list:
            text = f"{p.get('abstract', '')} {p.get('summary', '')}"
            # Find numbers near benchmark
            idx = text.lower().find(bench.lower())
            if idx >= 0:
                window = text[max(0, idx - 100):idx + 200]
                nums = re.findall(r"(\d{1,2}(?:\.\d{1,2})?)\s*%", window)
                for n in nums:
                    values.append((float(n), p))

        if len(values) >= 2:
            vals_only = [v[0] for v in values]
            max_v, min_v = max(vals_only), min(vals_only)
            if max_v - min_v > 5:  # >5 percentage points difference
                signals.append({
                    "benchmark": bench,
                    "paper_count": len(bp_list),
                    "value_range": f"{min_v:.1f}% – {max_v:.1f}%",
                    "gap": round(max_v - min_v, 1),
                    "papers_involved": [p.get("paper_id", "?") for _, p in values],
                    "description": (
                        f"在 {bench} 基准上，不同论文报告的性能从 {min_v:.1f}% 到 {max_v:.1f}%，"
                        f"差距达 {max_v - min_v:.1f} 个百分点，可能存在实验设置差异或结论矛盾。"
                    ),
                })

    return {
        "signals": signals[:5],
        "has_signals": len(signals) > 0,
    }


# ── Paper selectors per section ──────────────────────────────────────────

def _select_background_papers(papers: list[dict]) -> list[dict]:
    papers = _ensure_scores(papers)
    return _sort_by(papers, "citation_score")[:3]


def _select_core_problem_papers(papers: list[dict]) -> list[dict]:
    problem_kw = [
        "challenge", "limitation", "future work", "open problem",
        "drawback", "remain", "not yet", "bottleneck",
    ]
    selected = []
    for p in papers:
        text = f"{p.get('abstract', '')} {p.get('summary', '')}".lower()
        if any(kw in text for kw in problem_kw):
            selected.append(p)
    papers = _ensure_scores(selected)
    return _sort_by(papers, "composite_score")[:5]


def _select_technical_route_papers(papers: list[dict], comparison_matrix: dict) -> list[dict]:
    papers = _ensure_scores(papers)
    selected = []
    for mt in comparison_matrix.get("method_types", [])[:5]:
        members = _sort_by(mt["papers"], "composite_score")[:2]
        for p in members:
            if p not in selected:
                selected.append(p)
    return selected[:10]


def _select_topic_map_papers(papers: list[dict], topic_clusters: dict) -> list[dict]:
    papers = _ensure_scores(papers)
    clusters = topic_clusters.get("clusters", [])
    if not clusters:
        return _sort_by(papers, "composite_score")[:10]
    selected = []
    for c in clusters:
        cluster_papers = c.get("papers", [])
        for p in _sort_by(cluster_papers, "composite_score")[:3]:
            if p not in selected:
                selected.append(p)
    return selected[:15] if selected else _sort_by(papers, "composite_score")[:10]


def _select_representative_papers(papers: list[dict]) -> list[dict]:
    papers = _ensure_scores(papers)
    return _sort_by(papers, "composite_score")[:10]


def _select_latest_papers(papers: list[dict]) -> list[dict]:
    papers = _ensure_scores(papers)
    return _sort_by(papers, "freshness_score")[:10]


def _select_comparison_matrix_papers(papers: list[dict], comparison_matrix: dict) -> list[dict]:
    papers = _ensure_scores(papers)
    selected = []
    for mt in comparison_matrix.get("method_types", []):
        members = _sort_by(mt["papers"], "composite_score")[:5]
        for p in members:
            if p not in selected:
                selected.append(p)
    return selected[:15]


def _select_controversy_papers(papers: list[dict], contradiction_signals: dict) -> list[dict]:
    papers = _ensure_scores(papers)
    if not contradiction_signals.get("has_signals"):
        return _sort_by(papers, "composite_score")[:5]
    selected = []
    for sig in contradiction_signals.get("signals", []):
        for pid_placeholder in sig.get("papers_involved", []):
            for p in papers:
                if p.get("paper_id") == pid_placeholder:
                    selected.append(p)
    if not selected:
        return _sort_by(papers, "composite_score")[:5]
    return selected[:8]


def _select_gap_papers(papers: list[dict], research_gaps: dict | None) -> list[dict]:
    papers = _ensure_scores(papers)
    if not research_gaps:
        return _sort_by(papers, "composite_score")[:5]
    gap_kw = [
        "future work", "open problem", "little attention", "few studies",
        "underexplored", "has not been", "remains unclear",
    ]
    selected = []
    for p in papers:
        text = f"{p.get('abstract', '')} {p.get('summary', '')}".lower()
        if any(kw in text for kw in gap_kw):
            selected.append(p)
    return _sort_by(selected, "composite_score")[:8]


def _select_suggestion_papers(
    papers: list[dict],
    topic_suggestions: list[dict] | None,
) -> list[dict]:
    papers = _ensure_scores(papers)
    if not topic_suggestions:
        return _sort_by(papers, "composite_score")[:5]
    selected = []
    for s in topic_suggestions:
        direction = s.get("direction", "")
        keywords = re.findall(r"[一-鿿\w]+", direction)
        for p in papers:
            text = f"{p.get('title', '')} {p.get('abstract', '')} {p.get('summary', '')}".lower()
            if any(kw.lower() in text for kw in keywords if len(kw) >= 3):
                if p not in selected:
                    selected.append(p)
    return _sort_by(selected, "composite_score")[:8]


# ── Section data builders ────────────────────────────────────────────────

def _build_section_evidence(papers: list[dict], max_papers: int) -> str:
    """Build evidence text for a set of selected papers."""
    if not papers:
        return "（无可用证据论文）"
    pack = build_evidence_pack(papers, max_papers=max_papers)
    return format_evidence_pack_for_prompt(pack)


# ── Main planner ─────────────────────────────────────────────────────────

def build_data_driven_report_plan(
    user_query: str,
    papers: list[dict],
    topic_clusters: dict | None = None,
    comparison_matrix: dict | None = None,
    research_gaps: dict | None = None,
    ranking_results: dict | None = None,
) -> dict:
    """Build a data-driven report plan with paper assignments per section.

    Parameters
    ----------
    user_query : str
        The research area or question.
    papers : list[dict]
        All deduplicated papers (should have scores attached).
    topic_clusters : dict | None
        Output from topic_clusterer.cluster_papers().
    comparison_matrix : dict | None
        Output from build_comparison_matrix(). Built automatically if None.
    research_gaps : dict | None
        Output from analyze_research_gaps().
    ranking_results : dict | None
        Reserved for future use. Papers with computed scores.

    Returns
    -------
    dict with keys:
        sections : list[dict]
            Each: {template_title, section_data, selected_papers, evidence_text}
        comparison_matrix : dict
        contradiction_signals : dict
        total_papers : int
    """
    papers = _ensure_scores(papers)
    n = len(papers)

    # Build comparison matrix if not provided
    if comparison_matrix is None:
        comparison_matrix = build_comparison_matrix(papers)

    # Detect contradiction signals
    contradiction_signals = _detect_contradiction_signals(papers, comparison_matrix)

    # Default empty structures
    if topic_clusters is None:
        topic_clusters = {"clusters": [], "n_clusters": 0, "total_papers": n, "silhouette_score": 0}
    if research_gaps is None:
        research_gaps = {}
    if ranking_results is None:
        ranking_results = {}

    # ── Build section plans ──
    sections: list[dict] = []

    # Sub-directions string for background
    sub_dirs = research_gaps.get("sub_directions", []) if research_gaps else []
    if not sub_dirs:
        sub_dirs = [user_query]

    # 1. 研究背景
    bg_papers = _select_background_papers(papers)
    sections.append({
        "template_title": "研究背景",
        "template": SEC_BACKGROUND,
        "section_data": {
            "user_query": user_query,
            "sub_directions": "\n".join(f"- {sd}" for sd in sub_dirs),
            "total_papers": str(n),
            "n_directions": str(len(sub_dirs)),
        },
        "selected_papers": bg_papers,
        "evidence_text": _build_section_evidence(bg_papers, 3),
    })

    # 2. 核心问题
    cp_papers = _select_core_problem_papers(papers)
    rp = research_gaps.get("recurring_problems", {})
    rp_text = _format_recurring_problems(rp)
    sections.append({
        "template_title": "核心问题",
        "template": SEC_CORE_PROBLEMS,
        "section_data": {
            "user_query": user_query,
            "recurring_problems": rp_text,
        },
        "selected_papers": cp_papers,
        "evidence_text": _build_section_evidence(cp_papers, 5),
    })

    # 3. 主要技术路线
    tr_papers = _select_technical_route_papers(papers, comparison_matrix)
    md = research_gaps.get("method_distribution", {})
    md_text = _format_method_distribution(md)
    sections.append({
        "template_title": "主要技术路线",
        "template": SEC_TECHNICAL_ROUTES,
        "section_data": {
            "user_query": user_query,
            "method_distribution": md_text,
        },
        "selected_papers": tr_papers,
        "evidence_text": _build_section_evidence(tr_papers, 10),
    })

    # 4. 研究方向图谱
    tm_papers = _select_topic_map_papers(papers, topic_clusters)
    cs_text = _format_cluster_summary(topic_clusters)
    sections.append({
        "template_title": "研究方向图谱",
        "template": SEC_TOPIC_MAP,
        "section_data": {
            "user_query": user_query,
            "cluster_summary": cs_text,
        },
        "selected_papers": tm_papers,
        "evidence_text": _build_section_evidence(tm_papers, 15),
    })

    # 5. 代表性工作
    rep_papers = _select_representative_papers(papers)
    sections.append({
        "template_title": "代表性工作",
        "template": SEC_REPRESENTATIVE,
        "section_data": {"user_query": user_query},
        "selected_papers": rep_papers,
        "evidence_text": _build_section_evidence(rep_papers, 10),
    })

    # 6. 最新进展
    latest_papers = _select_latest_papers(papers)
    sections.append({
        "template_title": "最新进展",
        "template": SEC_LATEST,
        "section_data": {"user_query": user_query},
        "selected_papers": latest_papers,
        "evidence_text": _build_section_evidence(latest_papers, 10),
    })

    # 7. 方法对比矩阵
    cm_papers = _select_comparison_matrix_papers(papers, comparison_matrix)
    ma_text = _format_method_assignment(comparison_matrix, cm_papers)
    sections.append({
        "template_title": "方法对比矩阵",
        "template": SEC_COMPARISON_MATRIX,
        "section_data": {
            "user_query": user_query,
            "method_assignment": ma_text,
        },
        "selected_papers": cm_papers,
        "evidence_text": _build_section_evidence(cm_papers, 15),
    })

    # 8. 矛盾与争议
    cv_papers = _select_controversy_papers(papers, contradiction_signals)
    cv_text = _format_contradiction_signals(contradiction_signals)
    sections.append({
        "template_title": "矛盾与争议",
        "template": SEC_CONTROVERSIES,
        "section_data": {
            "user_query": user_query,
            "contradiction_signals": cv_text,
        },
        "selected_papers": cv_papers,
        "evidence_text": _build_section_evidence(cv_papers, 8),
    })

    # 9. 研究空白
    gap_papers = _select_gap_papers(papers, research_gaps)
    gs_text = _format_gaps_summary(research_gaps)
    sections.append({
        "template_title": "研究空白",
        "template": SEC_RESEARCH_GAPS,
        "section_data": {
            "user_query": user_query,
            "research_gaps_summary": gs_text,
        },
        "selected_papers": gap_papers,
        "evidence_text": _build_section_evidence(gap_papers, 8),
    })

    # 10. 选题建议
    ts_list = research_gaps.get("topic_suggestions", [])
    sug_papers = _select_suggestion_papers(papers, ts_list)
    ts_text = _format_topic_suggestions(ts_list)
    sections.append({
        "template_title": "选题建议",
        "template": SEC_TOPIC_SUGGESTIONS,
        "section_data": {
            "user_query": user_query,
            "topic_suggestions": ts_text,
        },
        "selected_papers": sug_papers,
        "evidence_text": _build_section_evidence(sug_papers, 8),
    })

    return {
        "sections": sections,
        "comparison_matrix": comparison_matrix,
        "contradiction_signals": contradiction_signals,
        "total_papers": n,
    }


# ── Formatting helpers ───────────────────────────────────────────────────

def _format_recurring_problems(rp: dict) -> str:
    if not rp:
        return "（未提取到明确的共性问题）"
    lines = [f"共 {rp.get('total_mentions', 0)} 处问题/局限描述："]
    for cat in rp.get("problem_categories", [])[:5]:
        lines.append(f"- {cat['category']}（{cat['count']} 次）")
    return "\n".join(lines)


def _format_method_distribution(md: dict) -> str:
    if not md:
        return "（无法获取方法分布数据）"
    lines = []
    cats = md.get("category_distribution", {})
    for cat_name, cd in cats.items():
        lines.append(f"- {cat_name}: {cd['count']} 篇 ({cd['pct']}%)")
    families = md.get("method_families", {})
    if families:
        lines.append("\n主要技术路线：")
        for name, count in families.items():
            lines.append(f"- {name}: {count} 次提及")
    return "\n".join(lines)


def _format_cluster_summary(tc: dict) -> str:
    if not tc:
        return "（未进行主题聚类）"
    lines = [
        f"共 {tc.get('n_clusters', 0)} 个主题簇，"
        f"轮廓系数: {tc.get('silhouette_score', 0):.3f}",
        "",
    ]
    for c in tc.get("clusters", []):
        kws = c.get("keywords", [])
        lines.append(
            f"- 主题 {c['id']+1}: {c['topic_name']} "
            f"（{c['paper_count']} 篇）关键词: {', '.join(kws[:5])}"
        )
    return "\n".join(lines)


def _format_method_assignment(cm: dict, selected_papers: list[dict]) -> str:
    """Format method assignment for the comparison matrix section."""
    lines = []
    for mt in cm.get("method_types", [])[:6]:
        papers_in_type = mt["papers"][:5]
        pids = [p.get("paper_id", "?") for p in papers_in_type]
        lines.append(f"- **{mt['name']}**（{mt['paper_count']} 篇）: {', '.join(pids)}")
    return "\n".join(lines)


def _format_contradiction_signals(cs: dict) -> str:
    if not cs.get("has_signals"):
        return "（未检测到明显矛盾信号）"
    lines = [f"检测到 {len(cs.get('signals', []))} 个潜在矛盾信号："]
    for sig in cs.get("signals", []):
        lines.append(f"- {sig['description']}")
    return "\n".join(lines)


def _format_gaps_summary(rg: dict) -> str:
    if not rg:
        return "（未进行分析）"
    lines = []
    gaps = rg.get("potential_gaps", [])
    if not gaps:
        return "（未检测到明显研究空白）"
    for g in gaps[:5]:
        lines.append(f"- [{g.get('confidence', '?')}] {g.get('theme', '')}: {g.get('description', '')[:150]}")
    return "\n".join(lines)


def _format_topic_suggestions(ts_list: list[dict]) -> str:
    if not ts_list:
        return "（未生成选题建议）"
    lines = []
    for s in ts_list[:5]:
        lines.append(
            f"- **选题 {s['id']}**: {s['direction']} "
            f"（可行性: {s.get('feasibility', '?')}, 创新: {s.get('innovation_level', '?')}）"
        )
        lines.append(f"  理由: {s.get('rationale', '')[:200]}")
    return "\n".join(lines)
