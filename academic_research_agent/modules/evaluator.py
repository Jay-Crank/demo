"""Experiment evaluator — four experiments comparing retrieval / ranking strategies.

Each function accepts inputs and returns a dict of metrics + display data
consumed by the "实验评估" Streamlit page.
"""

import numpy as np

from modules.arxiv_client import search_papers, search_multi_keywords
from modules.deduplicator import deduplicate
from modules.query_planner import generate_keywords
from modules.reranker import (
    compute_composite,
    compute_relevance,
    compute_freshness,
    compute_quality,
    compute_citation_score,
    compute_source_score,
    compute_preference_score,
)
from modules.topic_clusterer import cluster_papers
from modules.report_generator import generate_search_answer
from modules.verifier import assess_evidence


# ═══════════════════════════════════════════════════════════════════════
# Experiment 1 — Query expansion effectiveness
# ═══════════════════════════════════════════════════════════════════════

def evaluate_query_expansion(
    query: str,
    papers_per_kw: int = 15,
) -> dict:
    """Compare raw query vs. expanded-keyword retrieval.

    Gracefully handles empty results: if both sides return 0 papers,
    metrics default to 0 and the UI should show a friendly message.
    """
    """
    Compare raw query vs. expanded-keyword retrieval.

    Returns dict with keys:
      raw_metrics, expanded_metrics, comparison_table, chart_data.
    """
    # ── Raw query ──
    raw_papers = search_papers(query, max_results=20)
    raw_deduped = deduplicate(raw_papers)
    raw_ranked = compute_composite(raw_deduped, query)

    # ── Expanded query ──
    keywords = generate_keywords(query, n=5)
    exp_raw = search_multi_keywords(keywords, papers_per_keyword=papers_per_kw)
    exp_deduped = deduplicate(exp_raw)
    exp_ranked = compute_composite(exp_deduped, query)

    # ── Topic cluster counts ──
    raw_clusters = len(cluster_papers(raw_ranked).get("clusters", [])) if len(raw_ranked) >= 6 else 0
    exp_clusters = len(cluster_papers(exp_ranked).get("clusters", [])) if len(exp_ranked) >= 6 else 0

    # ── High-credibility conclusion count ──
    def _count_high(ranked_papers, q):
        if len(ranked_papers) < 3:
            return 0
        try:
            result = generate_search_answer(q, [q], ranked_papers, top_n=min(10, len(ranked_papers)))
            sections = result.get("sections", {})
            if not sections:
                return 0
            ver = assess_evidence(q, ranked_papers, sections)
            return ver["overall"].get("high_count", 0)
        except Exception:
            return 0

    raw_high = _count_high(raw_ranked, query)
    exp_high = _count_high(exp_ranked, query)

    # ── Build comparison ──
    def _top_avg(papers, field, n=10):
        vals = [p.get(field, 0) for p in papers[:n]]
        return round(float(np.mean(vals)), 4) if vals else 0.0

    comparison = {
        "指标": [
            "检索论文总数", "去重后论文数", "Top10 平均相关度",
            "Top10 平均新鲜度", "聚类主题数", "高可信度结论数",
        ],
        "原始查询": [
            len(raw_papers), len(raw_deduped),
            _top_avg(raw_ranked, "relevance_score"),
            _top_avg(raw_ranked, "freshness_score"),
            raw_clusters, raw_high,
        ],
        "查询扩展": [
            len(exp_raw), len(exp_deduped),
            _top_avg(exp_ranked, "relevance_score"),
            _top_avg(exp_ranked, "freshness_score"),
            exp_clusters, exp_high,
        ],
    }

    chart_data = {
        "指标": ["检索论文总数", "去重后论文数", "Top10相关度", "Top10新鲜度", "聚类主题数", "高可信度结论数"],
        "原始查询": [
            len(raw_papers), len(raw_deduped),
            _top_avg(raw_ranked, "relevance_score"),
            _top_avg(raw_ranked, "freshness_score"),
            raw_clusters, raw_high,
        ],
        "查询扩展": [
            len(exp_raw), len(exp_deduped),
            _top_avg(exp_ranked, "relevance_score"),
            _top_avg(exp_ranked, "freshness_score"),
            exp_clusters, exp_high,
        ],
    }

    return {
        "keywords": keywords,
        "comparison": comparison,
        "chart_data": chart_data,
        "raw_ranked": raw_ranked,
        "exp_ranked": exp_ranked,
    }


# ═══════════════════════════════════════════════════════════════════════
# Experiment 2 — Ranking method comparison
# ═══════════════════════════════════════════════════════════════════════

def compare_ranking_methods(
    papers: list[dict],
    query: str,
) -> dict:
    """
    Compare three ranking strategies on the same paper set.

    Returns dict with keys:
      date_ranked, relevance_ranked, composite_ranked,
      comparison_table, chart_data.
    """
    if not papers:
        return _empty_ranking_result()

    # ── Sort by publish date (newest first) ──
    date_ranked = sorted(papers, key=lambda p: p.get("published") or _sentinel_date, reverse=True)
    # Attach relevance / freshness / quality scores for display
    _attach_scores(date_ranked, query)

    # ── Sort by relevance only ──
    rel_scores = compute_relevance(papers, query)
    for i, p in enumerate(papers):
        p = dict(p)
        p["relevance_score"] = round(float(rel_scores[i]), 4)
    relevance_ranked = sorted(
        [dict(p, relevance_score=round(float(rel_scores[i]), 4)) for i, p in enumerate(papers)],
        key=lambda x: x["relevance_score"], reverse=True,
    )
    _attach_scores(relevance_ranked, query, skip_rel=True)

    # ── Sort by composite (default weights) ──
    composite_ranked = compute_composite(papers, query)

    # ── Compute metrics ──
    def _top_avg(papers_list, field, n=5):
        vals = [p.get(field, 0) for p in papers_list[:n]]
        return round(float(np.mean(vals)), 4) if vals else 0.0

    def _count_pref(papers_list, pref_key, n=5):
        scores = compute_preference_score(papers_list[:n], [pref_key])
        return int((scores > 0).sum())

    metrics = {}
    metric_keys = [
        "Top5 平均相关度", "Top5 平均新鲜度", "Top5 平均引用分",
        "Top5 平均来源分", "Top5 平均质量分",
        "Top5 综述论文数", "Top5 方法论文数",
    ]
    for label, ranked in [("按时间排序", date_ranked), ("按相关度排序", relevance_ranked), ("综合排序", composite_ranked)]:
        metrics[label] = {
            "Top5 平均相关度": _top_avg(ranked, "relevance_score"),
            "Top5 平均新鲜度": _top_avg(ranked, "freshness_score"),
            "Top5 平均引用分": _top_avg(ranked, "citation_score"),
            "Top5 平均来源分": _top_avg(ranked, "source_score"),
            "Top5 平均质量分": _top_avg(ranked, "quality_score"),
            "Top5 综述论文数": _count_pref(ranked, "survey"),
            "Top5 方法论文数": _count_pref(ranked, "method"),
        }

    comparison = {
        "指标": metric_keys,
        "按时间排序": [metrics["按时间排序"][k] for k in metric_keys],
        "按相关度排序": [metrics["按相关度排序"][k] for k in metric_keys],
        "综合排序": [metrics["综合排序"][k] for k in metric_keys],
    }

    return {
        "date_ranked": date_ranked,
        "relevance_ranked": relevance_ranked,
        "composite_ranked": composite_ranked,
        "comparison": comparison,
    }


_sentinel_date = __import__("datetime").datetime(2000, 1, 1, tzinfo=__import__("datetime").timezone.utc)


def _attach_scores(papers, query, skip_rel=False):
    if not skip_rel:
        rel = compute_relevance(papers, query)
    fresh = compute_freshness(papers)
    qual = compute_quality(papers)
    cite = compute_citation_score(papers)
    src = compute_source_score(papers)
    for i, p in enumerate(papers):
        if not skip_rel:
            p["relevance_score"] = round(float(rel[i]), 4)
        if "freshness_score" not in p:
            p["freshness_score"] = round(float(fresh[i]), 4)
        if "quality_score" not in p:
            p["quality_score"] = round(float(qual[i]), 4)
        if "citation_score" not in p:
            p["citation_score"] = round(float(cite[i]), 4)
        if "source_score" not in p:
            p["source_score"] = round(float(src[i]), 4)


# ═══════════════════════════════════════════════════════════════════════
# Experiment 3 — Personalized weight comparison
# ═══════════════════════════════════════════════════════════════════════

def evaluate_personalized_ranking(
    papers: list[dict],
    query: str,
) -> dict:
    """
    Compare three user profiles (A/B/C) on the same paper set.

    Profile A: 关注最新进展 → w_fresh dominant
    Profile B: 关注综述论文 → preference survey dominant
    Profile C: 关注方法创新 → preference method dominant

    Returns dict with keys:
      profile_a/b/c (ranked lists),
      overlap, comparison_table, chart_data.
    """
    if not papers:
        return _empty_personalized_result()

    profiles = {
        "A-关注最新进展": {
            "w_rel": 0.20, "w_fresh": 0.50, "w_cite": 0.10,
            "w_src": 0.10, "w_qual": 0.10,
        },
        "B-关注高质量来源": {
            "w_rel": 0.25, "w_fresh": 0.10, "w_cite": 0.30,
            "w_src": 0.20, "w_qual": 0.15,
        },
        "C-关注方法创新": {
            "w_rel": 0.55, "w_fresh": 0.10, "w_cite": 0.10,
            "w_src": 0.10, "w_qual": 0.15,
        },
    }

    results = {}
    for label, cfg in profiles.items():
        ranked = compute_composite(
            papers, query,
            w_rel=cfg["w_rel"], w_fresh=cfg["w_fresh"],
            w_cite=cfg["w_cite"], w_src=cfg["w_src"], w_qual=cfg["w_qual"],
        )
        results[label] = ranked

    # ── Overlap ──
    def _top_ids(ranked, n=5):
        return {p.get("arxiv_id", str(i)) for i, p in enumerate(ranked[:n])}

    ids_a = _top_ids(results["A-关注最新进展"])
    ids_b = _top_ids(results["B-关注高质量来源"])
    ids_c = _top_ids(results["C-关注方法创新"])

    overlap = {
        "A ∩ B": len(ids_a & ids_b),
        "A ∩ C": len(ids_a & ids_c),
        "B ∩ C": len(ids_b & ids_c),
        "A ∩ B ∩ C": len(ids_a & ids_b & ids_c),
    }

    # ── Metrics ──
    def _top_avg(papers_list, field, n=5):
        vals = [p.get(field, 0) for p in papers_list[:n]]
        return round(float(np.mean(vals)), 4) if vals else 0.0

    metrics = {}
    for label, ranked in results.items():
        metrics[label] = {
            "Top5 平均相关度": _top_avg(ranked, "relevance_score"),
            "Top5 平均新鲜度": _top_avg(ranked, "freshness_score"),
            "Top5 平均引用分": _top_avg(ranked, "citation_score"),
            "Top5 平均来源分": _top_avg(ranked, "source_score"),
        }

    comparison = {
        "指标": ["Top5 平均相关度", "Top5 平均新鲜度", "Top5 平均引用分", "Top5 平均来源分"],
        "A-关注最新进展": [metrics["A-关注最新进展"]["Top5 平均相关度"], metrics["A-关注最新进展"]["Top5 平均新鲜度"], metrics["A-关注最新进展"]["Top5 平均引用分"], metrics["A-关注最新进展"]["Top5 平均来源分"]],
        "B-关注高质量来源": [metrics["B-关注高质量来源"]["Top5 平均相关度"], metrics["B-关注高质量来源"]["Top5 平均新鲜度"], metrics["B-关注高质量来源"]["Top5 平均引用分"], metrics["B-关注高质量来源"]["Top5 平均来源分"]],
        "C-关注方法创新": [metrics["C-关注方法创新"]["Top5 平均相关度"], metrics["C-关注方法创新"]["Top5 平均新鲜度"], metrics["C-关注方法创新"]["Top5 平均引用分"], metrics["C-关注方法创新"]["Top5 平均来源分"]],
    }

    return {
        "profile_a": results["A-关注最新进展"],
        "profile_b": results["B-关注高质量来源"],
        "profile_c": results["C-关注方法创新"],
        "overlap": overlap,
        "comparison": comparison,
    }


# ═══════════════════════════════════════════════════════════════════════
# Experiment 4 — Topic clustering analysis
# ═══════════════════════════════════════════════════════════════════════

def evaluate_topic_clustering(
    papers: list[dict],
) -> dict:
    """
    Run topic clustering and return displayable results.

    Returns dict with keys:
      n_clusters, silhouette, clusters, chart_data, summary.
    """
    if len(papers) < 6:
        return {
            "n_clusters": 0,
            "silhouette": 0.0,
            "clusters": [],
            "summary": "论文数量不足（需 >= 6 篇），无法进行聚类分析。",
        }

    result = cluster_papers(papers)
    clusters = result.get("clusters", [])
    n_clusters = result.get("n_clusters", 0)
    sil = result.get("silhouette_score", 0.0)

    # Build chart data
    chart_data = {
        "主题簇": [f"主题{c['id']+1}: {c['topic_name']}" for c in clusters],
        "论文数量": [c["paper_count"] for c in clusters],
    }

    # Auto-generate summary
    if n_clusters >= 3:
        top_cluster = clusters[0]
        summary = (
            f"共识别出 {n_clusters} 个研究方向，其中 **{top_cluster['topic_name']}** "
            f"是论文数量最多的主题（{top_cluster['paper_count']} 篇）。"
        )
        if sil >= 0.5:
            summary += " 聚类轮廓系数较高，说明主题之间边界清晰。"
        elif sil >= 0.3:
            summary += " 聚类轮廓系数中等，各主题之间存在一定交叉。"
        else:
            summary += " 聚类轮廓系数较低，论文内容交叉较多。"
    elif n_clusters >= 1:
        summary = f"共识别出 {n_clusters} 个研究方向，论文分布相对集中。"
    else:
        summary = "未能识别出明显的研究方向聚类。"

    return {
        "n_clusters": n_clusters,
        "silhouette": sil,
        "clusters": clusters,
        "chart_data": chart_data,
        "summary": summary,
    }


# ── Empty-result helpers ──────────────────────────────────────────────

def _empty_ranking_result() -> dict:
    return {
        "date_ranked": [], "relevance_ranked": [], "composite_ranked": [],
        "comparison": {"指标": [], "按时间排序": [], "按相关度排序": [], "综合排序": []},
    }


def _empty_personalized_result() -> dict:
    return {
        "profile_a": [], "profile_b": [], "profile_c": [],
        "overlap": {"A ∩ B": 0, "A ∩ C": 0, "B ∩ C": 0, "A ∩ B ∩ C": 0},
        "comparison": {
            "指标": [], "A-关注最新进展": [], "B-关注高质量来源": [], "C-关注方法创新": [],
        },
    }


# ═══════════════════════════════════════════════════════════════════════
# Experiment 5 — Multi-source comparison
# ═══════════════════════════════════════════════════════════════════════

def evaluate_multi_source(
    query: str,
    max_results: int = 20,
) -> dict:
    """Compare arXiv, OpenAlex, and Semantic Scholar on the same query.

    Returns dict with keys:
      source_stats, merge_stats, comparison_table, chart_data, errors.
    """
    from datetime import datetime
    from modules.sources.arxiv_client import ArxivSource
    from modules.sources.openalex_client import OpenAlexSource
    from modules.sources.semantic_scholar_client import SemanticScholarSource
    from modules.deduplicator import deduplicate

    sources = {
        "arxiv": ArxivSource(),
        "openalex": OpenAlexSource(),
        "semantic_scholar": SemanticScholarSource(),
    }

    # ── Query each source independently ──
    raw_by_source: dict[str, list[dict]] = {}
    source_errors: dict[str, str | None] = {}

    for name, src in sources.items():
        try:
            papers = src.search(query, max_results=max_results)
            raw_by_source[name] = papers or []
            source_errors[name] = src.last_error if src.had_error else None
        except Exception as e:
            raw_by_source[name] = []
            source_errors[name] = str(e)

    # ── Compute per-source stats ──
    source_stats: dict[str, dict] = {}
    now = datetime.now()

    for name, papers in raw_by_source.items():
        n = len(papers)
        years = []
        citations = []
        no_abstract = 0
        no_doi = 0

        for p in papers:
            y = p.get("year")
            if y and isinstance(y, (int, float)) and 1900 <= y <= now.year + 2:
                years.append(int(y))
            cit = p.get("citation_count", 0)
            citations.append(int(cit) if cit else 0)
            if not p.get("abstract") or not p.get("abstract", "").strip():
                no_abstract += 1
            if not p.get("doi") or not p.get("doi", "").strip():
                no_doi += 1

        avg_year = round(sum(years) / len(years), 1) if years else None
        avg_citations = round(sum(citations) / len(citations), 1) if citations else 0.0
        abstract_miss_rate = round(no_abstract / n * 100, 1) if n > 0 else 0.0
        doi_miss_rate = round(no_doi / n * 100, 1) if n > 0 else 0.0

        source_stats[name] = {
            "count": n,
            "avg_year": avg_year,
            "avg_citations": avg_citations,
            "abstract_miss_rate": abstract_miss_rate,
            "doi_miss_rate": doi_miss_rate,
            "error": source_errors.get(name),
        }

    # ── Merge & dedup ──
    all_raw: list[dict] = []
    for papers in raw_by_source.values():
        all_raw.extend(papers)
    total_raw = len(all_raw)

    if all_raw:
        deduped = deduplicate(all_raw, threshold=0.92)
    else:
        deduped = []
    total_deduped = len(deduped)
    total_duplicates = total_raw - total_deduped

    merge_stats = {
        "total_raw": total_raw,
        "total_deduped": total_deduped,
        "total_duplicates": total_duplicates,
        "dedup_rate": round(total_duplicates / total_raw * 100, 1) if total_raw > 0 else 0.0,
    }

    # ── Overlap matrix ──
    def _ids(papers_list: list[dict], key: str) -> set[str]:
        s = set()
        for p in papers_list:
            v = p.get(key, "")
            if v:
                s.add(str(v).lower())
        return s

    arxiv_doi = _ids(raw_by_source.get("arxiv", []), "doi")
    oa_doi = _ids(raw_by_source.get("openalex", []), "doi")
    s2_doi = _ids(raw_by_source.get("semantic_scholar", []), "doi")

    arxiv_aid = _ids(raw_by_source.get("arxiv", []), "arxiv_id")
    oa_aid = _ids(raw_by_source.get("openalex", []), "arxiv_id")
    s2_aid = _ids(raw_by_source.get("semantic_scholar", []), "arxiv_id")

    # Combine DOI + arXiv ID for overlap
    arxiv_ids = arxiv_doi | arxiv_aid
    oa_ids = oa_doi | oa_aid
    s2_ids = s2_doi | s2_aid

    overlap = {
        "arxiv ∩ openalex": len(arxiv_ids & oa_ids),
        "arxiv ∩ semantic_scholar": len(arxiv_ids & s2_ids),
        "openalex ∩ semantic_scholar": len(oa_ids & s2_ids),
        "三源重合": len(arxiv_ids & oa_ids & s2_ids),
    }

    # ── Comparison table ──
    source_display_names = {
        "arxiv": "arXiv",
        "openalex": "OpenAlex",
        "semantic_scholar": "Semantic Scholar",
    }

    comparison = {
        "指标": [
            "返回论文数", "平均年份", "平均引用数",
            "摘要缺失率(%)", "DOI缺失率(%)", "调用状态",
        ],
    }
    chart_data = {
        "指标": ["返回论文数", "平均引用数", "摘要缺失率(%)", "DOI缺失率(%)"],
    }

    for key, display in source_display_names.items():
        s = source_stats.get(key, {})
        err = s.get("error")
        status = "成功" if not err else f"失败: {err[:50]}"
        comparison[display] = [
            s.get("count", 0),
            f"{s['avg_year']}" if s.get("avg_year") else "N/A",
            f"{s['avg_citations']}" if s.get("avg_citations") is not None else "N/A",
            s.get("abstract_miss_rate", 0),
            s.get("doi_miss_rate", 0),
            status,
        ]
        # Chart data — numeric only
        avg_cit = s.get("avg_citations") or 0
        chart_data[display] = [
            s.get("count", 0),
            avg_cit,
            s.get("abstract_miss_rate", 0),
            s.get("doi_miss_rate", 0),
        ]

    return {
        "source_stats": source_stats,
        "merge_stats": merge_stats,
        "overlap": overlap,
        "comparison": comparison,
        "chart_data": chart_data,
        "source_errors": source_errors,
    }
