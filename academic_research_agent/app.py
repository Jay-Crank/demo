"""Streamlit main app — Academic Research Agent with 3 pages."""

import os
import sys
from pathlib import Path

import streamlit as st

# Ensure modules can be imported
sys.path.insert(0, str(Path(__file__).parent))

from modules.arxiv_client import search_multi_keywords, search_papers, get_last_error, had_api_error
from modules.deduplicator import deduplicate
from modules.feed_generator import generate_daily_digest, generate_deep_interpretation
from modules.sources.arxiv_client import ArxivSource
from modules.sources.openalex_client import OpenAlexSource
from modules.sources.semantic_scholar_client import SemanticScholarSource
from modules.source_aggregator import aggregate_search
from modules.evaluator import (
    evaluate_query_expansion,
    compare_ranking_methods,
    evaluate_personalized_ranking,
    evaluate_topic_clustering,
    evaluate_multi_source,
)
from modules.query_planner import generate_keywords, generate_sub_directions
from modules.reranker import (
    compute_composite,
    get_default_weights,
    get_preference_keys,
    normalize_weights,
)
from modules.paper_card_generator import generate_paper_card, format_card_markdown
from modules.report_generator import generate_search_answer, generate_survey_report
from modules.research_gap_analyzer import analyze_research_gaps
from modules.topic_clusterer import cluster_papers
from modules.verifier import assess_evidence, format_credibility_badge
from modules.evidence_builder import build_evidence_pack
from modules.llm_client import try_configure_llm
from modules.llm_report_generator import (
    generate_evidence_anchored_report,
    generate_template_report_fallback,
    generate_data_driven_report,
)
from modules.report_validator import validate_report, format_validation_summary
from modules.report_planner import build_data_driven_report_plan, build_comparison_matrix
from modules.deep_reader import (
    download_arxiv_pdf, extract_arxiv_id, extract_full_text, parse_paper_sections,
    build_deep_reading_card, build_reading_path, build_fulltext_evidence_pack,
    format_deep_card_markdown, format_reading_path_markdown,
)


st.set_page_config(
    page_title="学术调研 Agent",
    page_icon=" ",
    layout="wide",
)

#  Sidebar 
st.sidebar.title("学术调研 Agent")
st.sidebar.markdown("基于多轮检索与个性化排序")
page = st.sidebar.radio(
    "选择功能",
    ["深度搜索", "深度调研", "个性化推送", "实验评估"],
)

st.sidebar.markdown("---")
st.sidebar.caption("数据来源：arXiv API | 算法：TF-IDF + 综合排序")
st.sidebar.caption("MVP v2.0 — 支持 AI 增强报告（证据锚定）")


#  Weight settings widget (shared) 
def show_weight_settings(defaults: dict | None = None, show_preferences: bool = False) -> dict:
    """Render collapsible weight settings (5 components).

    Returns dict with keys: w_rel, w_fresh, w_cite, w_src, w_qual, selected_preferences.
    """
    if defaults is None:
        defaults = get_default_weights()

    with st.expander("排序权重设置", expanded=False):
        st.caption(
            "Score = w_r·R + w_t·T + w_c·C + w_s·S + w_q·Q  "
            "— 系统会自动归一化使总和为 1。"
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            w_rel = st.slider(
                "相关度 w_r", 0.0, 1.0, defaults["w_rel"], 0.05,
                help="TF-IDF 余弦相似度 — 论文与查询的语义匹配程度",
            )
        with col2:
            w_fresh = st.slider(
                "新鲜度 w_t", 0.0, 1.0, defaults["w_fresh"], 0.05,
                help="exp(-Δdays/30) — 越新越接近 1",
            )
        with col3:
            w_cite = st.slider(
                "引用分 w_c", 0.0, 1.0, defaults["w_cite"], 0.05,
                help="log(1+citations) 归一化 — 反映论文学术影响力",
            )

        col4, col5, _ = st.columns(3)
        with col4:
            w_src = st.slider(
                "来源分 w_s", 0.0, 1.0, defaults["w_src"], 0.05,
                help="多源收录的论文得分更高（max 3 源）",
            )
        with col5:
            w_qual = st.slider(
                "质量分 w_q", 0.0, 1.0, defaults["w_qual"], 0.05,
                help="基于摘要/DOI/期刊/URL 的元数据完整度",
            )

        n_rel, n_fresh, n_cite, n_src, n_qual = normalize_weights(
            w_rel, w_fresh, w_cite, w_src, w_qual
        )
        st.caption(
            f"归一化 → w_r={n_rel:.2f}  w_t={n_fresh:.2f}  w_c={n_cite:.2f}  "
            f"w_s={n_src:.2f}  w_q={n_qual:.2f}"
        )

        selected_preferences = []
        if show_preferences:
            st.markdown("---")
            st.caption(" **主题偏好**（选中后自动调整权重偏向）")
            cols = st.columns(3)
            with cols[0]:
                if st.checkbox("关注综述论文", value=False, key="pref_survey"):
                    selected_preferences.append("关注综述论文")
            with cols[1]:
                if st.checkbox("关注方法创新", value=False, key="pref_method"):
                    selected_preferences.append("关注方法创新")
            with cols[2]:
                if st.checkbox("关注应用系统", value=False, key="pref_app"):
                    selected_preferences.append("关注应用系统")

        return {
            "w_rel": n_rel,
            "w_fresh": n_fresh,
            "w_cite": n_cite,
            "w_src": n_src,
            "w_qual": n_qual,
            "selected_preferences": selected_preferences,
        }


def show_source_selector(default_all: bool = True) -> list[str]:
    """Render source selection checkboxes, return list of selected source names."""
    st.caption(" **数据源选择**")
    cols = st.columns(3)
    selected: list[str] = []
    with cols[0]:
        if st.checkbox("arXiv", value=default_all, key="src_arxiv"):
            selected.append("arxiv")
    with cols[1]:
        if st.checkbox("OpenAlex", value=default_all, key="src_openalex"):
            selected.append("openalex")
    with cols[2]:
        if st.checkbox("Semantic Scholar", value=default_all, key="src_s2"):
            selected.append("semantic_scholar")
    return selected


def _init_sources(source_names: list[str]) -> list:
    """Create source instances from selected names."""
    sources = []
    if "arxiv" in source_names:
        sources.append(ArxivSource())
    if "openalex" in source_names:
        sources.append(OpenAlexSource())
    if "semantic_scholar" in source_names:
        sources.append(SemanticScholarSource())
    return sources


#  Helpers 

def _format_pub_date(pub) -> str:
    """Safely format a publication date to YYYY-MM-DD."""
    if pub is None:
        return "N/A"
    try:
        from datetime import datetime as _dt
        if isinstance(pub, _dt):
            return pub.strftime("%Y-%m-%d")
        return str(pub)[:10]
    except Exception:
        return "N/A"


def run_search_pipeline(
    query: str,
    papers_per_kw: int = 15,
    weights: dict | None = None,
    selected_preferences: list[str] | None = None,
    sources: list | None = None,
):
    """Common pipeline: expand keywords → multi-source search → dedup → rank."""
    if weights is None:
        weights = get_default_weights()
    if selected_preferences is None:
        selected_preferences = []
    if sources is None:
        sources = [ArxivSource()]

    source_names = [s.name for s in sources]

    with st.status("处理中...", expanded=True) as status:
        st.write("生成检索关键词...")
        keywords = generate_keywords(query, n=5)

        st.write(f"关键词: {', '.join(keywords)}")
        st.write(f"检索数据源: {', '.join(source_names)}")

        # Multi-keyword × multi-source search
        all_raw: list[dict] = []
        source_errors: list[str] = []
        for kw in keywords:
            result = aggregate_search(
                sources, kw, max_results=papers_per_kw, parallel=True
            )
            all_raw.extend(result["papers"])
            # Collect per-source errors
            for sname, stats in result.get("source_stats", {}).items():
                if stats.get("error"):
                    source_errors.append(f"{sname}: {stats['error']}")

        st.write(f"各源检索共获得 {len(all_raw)} 篇论文（未去重）")
        if source_errors:
            for err in source_errors[:3]:
                st.warning(f" {err}")

        st.write("去重处理 (DOI / arXiv ID / 标题相似度)...")
        deduped = deduplicate(all_raw, threshold=0.92)
        st.write(f"去重后剩余 {len(deduped)} 篇论文")

        if not deduped:
            status.update(label="未检索到论文", state="error")
            return keywords, []

        st.write("计算综合排序...")
        ranked = compute_composite(
            deduped,
            query,
            w_rel=weights["w_rel"],
            w_fresh=weights["w_fresh"],
            w_cite=weights["w_cite"],
            w_src=weights["w_src"],
            w_qual=weights["w_qual"],
        )

        status.update(label="处理完成！", state="complete")
    return keywords, ranked


def show_papers_table(papers: list[dict], n: int = 10, show_extra: bool = False):
    """Display top N papers as a table with all sub-scores."""
    if not papers:
        st.warning("未检索到相关论文。")
        return

    top = papers[:n]
    rows = []
    for i, p in enumerate(top, 1):
        title = p["title"][:80] + "..." if len(p["title"]) > 80 else p["title"]
        authors = p["authors"][:60] + "..." if len(p["authors"]) > 60 else p["authors"]
        sources_display = ", ".join(p.get("sources", [p.get("source", "")]) or [])
        row = {
            "排名": i,
            "论文标题": title,
            "作者": authors,
            "发表时间": _format_pub_date(p.get("published")),
            "数据源": sources_display,
            "最终得分": f"{p.get('composite_score', 0):.4f}",
            "相关度": f"{p.get('relevance_score', 0):.3f}",
            "新鲜度": f"{p.get('freshness_score', 0):.3f}",
            "引用分": f"{p.get('citation_score', 0):.3f}",
            "来源分": f"{p.get('source_score', 0):.3f}",
            "质量分": f"{p.get('quality_score', 0):.3f}",
        }
        if show_extra:
            row["DOI"] = p.get("doi", "") or "N/A"
            row["期刊/会议"] = p.get("venue", "") or "N/A"
            row["引用数"] = str(p.get("citation_count", 0))
            row["开放获取"] = "" if p.get("is_open_access") else "—"
        row["链接"] = f"[arXiv]({p['link']})"
        rows.append(row)
    st.dataframe(rows, use_container_width=True, hide_index=True)


def show_audit_view(
    paper_id_map: dict,
    validation: dict,
    report_text: str = "",
    heading: str = "引用审计视图",
) -> None:
    """Display full citation audit view for an evidence-anchored report.

    Shows:
    - Risk level with color-coded badge
    - Paper ID → info mapping table
    - Citation usage statistics
    - Invalid refs highlighted in red
    - Uncited claims highlighted in yellow
    - Per-paper expandable details
    """
    risk = validation.get("risk_level", "低风险")
    stats = validation.get("stats", {})
    cited = validation.get("cited_papers", [])
    invalid = validation.get("invalid_refs", [])
    uncited = validation.get("uncited_claims", [])
    suggestions = validation.get("suggestions", [])

    #  Section heading 
    st.markdown(f"## {heading}")

    #  Risk level badge 
    if risk == "高风险":
        st.error(f"###  风险等级：{risk}")
    elif risk == "中风险":
        st.warning(f"###  风险等级：{risk}")
    else:
        st.success(f"###  风险等级：{risk}")

    #  Summary metrics 
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("引用标记总数", stats.get("total_citation_marks", 0))
    with col2:
        st.metric("独立引用论文", stats.get("unique_papers_cited", 0))
    with col3:
        st.metric("证据覆盖率", f"{stats.get('paper_coverage', 0):.0%}")
    with col4:
        st.metric("无效引用", stats.get("invalid_ref_count", 0),
                  delta=f"-{stats.get('invalid_ref_count', 0)}" if stats.get("invalid_ref_count", 0) > 0 else None)
    with col5:
        st.metric("无据陈述", stats.get("uncited_claim_count", 0),
                  delta=f"-{stats.get('uncited_claim_count', 0)}" if stats.get("uncited_claim_count", 0) > 0 else None)

    #  Invalid refs (red) 
    if invalid:
        st.markdown("---")
        st.error(f"###  无效引用（{len(invalid)} 个）")
        st.markdown(
            "以下论文编号在报告中被引用，但**不存在于证据库中**，属于严重幻觉："
        )
        for ref in invalid:
            st.markdown(f"-  **`[{ref}]`** — 证据库中不存在此编号")
    else:
        st.markdown("---")
        st.success("###  无无效引用")

    #  Uncited claims (yellow) 
    if uncited:
        st.markdown("---")
        st.warning(f"###  无引用陈述（{len(uncited)} 条）")
        st.markdown("以下句子包含事实性信号（数字/比较/指标），但**未引用任何论文编号**：")
        for i, uc in enumerate(uncited, 1):
            sent = uc.get("sentence", "")
            line_num = uc.get("line", "?")
            reason = uc.get("reason", "疑似事实性陈述")
            with st.expander(f" #{i} 行 {line_num}: {sent[:80]}...", expanded=False):
                st.markdown(f"**完整句子：** {sent}")
                st.markdown(f"**标记原因：** {reason}")
                st.caption(f"所在行：{line_num}")
                # Show surrounding context from report
                if report_text:
                    lines = report_text.split("\n")
                    ln = int(line_num) if isinstance(line_num, int) or line_num.isdigit() else 0
                    if 0 < ln <= len(lines):
                        st.markdown("**上下文：**")
                        ctx_start = max(0, ln - 2)
                        ctx_end = min(len(lines), ln + 2)
                        for ctx_ln in range(ctx_start, ctx_end):
                            prefix = "→" if ctx_ln + 1 == ln else " "
                            st.code(f"{prefix} L{ctx_ln + 1}: {lines[ctx_ln][:200]}", language=None)
    else:
        st.markdown("---")
        st.success("###  无未引用的事实性陈述")

    #  Suggestions 
    if suggestions:
        st.markdown("---")
        st.info("###  建议")
        for s in suggestions:
            st.markdown(f"- {s}")

    #  Paper ID mapping table 
    st.markdown("---")
    st.subheader("论文编号映射表")
    st.caption("报告中的每个 [P#] 编号对应以下论文：")

    if paper_id_map:
        for pid, meta in paper_id_map.items():
            is_cited = pid in cited
            is_invalid = pid in invalid
            status_icon = "" if is_cited else ("" if is_invalid else "⬜")

            with st.expander(
                f"{status_icon} **{pid}** — {meta.get('title', '未知')[:70]}"
                f"{'...' if len(meta.get('title', '')) > 70 else ''}",
                expanded=is_invalid,  # auto-expand invalid refs
            ):
                # Use a compact metrics row
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.caption(f"**作者**：{meta.get('authors', '未知')}")
                    st.caption(f"**年份**：{meta.get('year', '未知')}")
                with c2:
                    st.caption(f"**期刊/会议**：{meta.get('venue', '未知')}")
                    st.caption(f"**引用数**：{meta.get('citation_count', 0)}")
                with c3:
                    url = meta.get("url", "")
                    st.caption(f"**来源**：{', '.join(meta.get('sources', ['未知'])) if isinstance(meta.get('sources'), list) else meta.get('sources', '未知')}")
                    if url:
                        st.caption(f"**链接**：[{url[:50]}...]({url})" if len(url) > 50 else f"**链接**：[{url}]({url})")

                if is_invalid:
                    st.error(" 此编号在报告中作为引用出现，但不存在于证据库中！")
                elif is_cited:
                    st.success(" 此论文在报告中被正确引用。")
                else:
                    st.info("⬜ 此论文存在于证据库中，但在报告中未被引用。")
    else:
        st.info("（无可用的论文编号映射数据）")

    #  Citation usage summary 
    if cited:
        st.markdown("---")
        st.subheader("引用使用概览")
        cited_set = set(cited)
        available_set = set(paper_id_map.keys())
        unused = available_set - cited_set
        st.markdown(
            f"- 证据库共 **{len(available_set)}** 篇论文\n"
            f"- 报告中引用了 **{len(cited_set)}** 篇\n"
            f"- 未被引用 **{len(unused)}** 篇：{', '.join(sorted(unused)) if unused else '无'}"
        )


#  Per-section audit view for data-driven reports 

def show_section_audit_rows(section_results: list[dict]) -> None:
    """Display per-section audit expanders for a data-driven report."""
    if not section_results:
        return

    st.markdown("---")
    st.subheader("逐节引用审计详情")

    for sr in section_results:
        title = sr.get("section_title", "?")
        success = sr.get("success", False)
        v = sr.get("validation", {})
        risk = v.get("risk_level", "?")
        stats = v.get("stats", {})
        emoji = {"高风险": "", "中风险": "", "低风险": ""}.get(risk, "")

        invalid_count = stats.get("invalid_ref_count", 0)
        uncited_count = stats.get("uncited_claim_count", 0)
        cited_count = stats.get("unique_papers_cited", 0)

        with st.expander(
            f"{emoji} **{title}** — "
            f"引用 {cited_count} 篇 | 无据 {uncited_count} 条 | 无效 {invalid_count} 个",
            expanded=(invalid_count > 0 or uncited_count > 3),
        ):
            if not success:
                st.error(sr.get("error", "未知错误"))
                continue

            # Per-section metrics
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("引用标记", stats.get("total_citation_marks", 0))
            with c2:
                st.metric("引用论文", cited_count)
            with c3:
                st.metric("无效引用", invalid_count)
            with c4:
                st.metric("无据陈述", uncited_count)

            # Section invalid refs
            invalid_refs = v.get("invalid_refs", [])
            if invalid_refs:
                st.error(f"无效引用：{', '.join(f'[{r}]' for r in invalid_refs)}")

            # Section uncited claims
            uncited = v.get("uncited_claims", [])
            if uncited:
                st.warning(f"**无引用陈述（{len(uncited)} 条）：**")
                for uc in uncited:
                    st.markdown(
                        f"- 行 {uc.get('line', '?')}: "
                        f"{uc.get('sentence', '')[:120]}..."
                    )
                    st.caption(f"原因：{uc.get('reason', '?')}")

            # Show section text in collapsed sub-expander
            with st.expander("查看本节原文", expanded=False):
                st.markdown(sr.get("section_text", ""))


def show_credibility_panel(verification: dict) -> None:
    """Display credibility overview + per-section evidence."""
    overall = verification.get("overall", {})
    sections = verification.get("sections", [])
    if not sections:
        return

    #  Overall badge row 
    level = overall.get("level", "低")
    score = overall.get("score", 0)
    emoji = {"高": "", "中": "", "低": ""}.get(level, "")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("整体可信度", f"{emoji} {level}")
    with col2:
        st.metric("可信度评分", f"{score:.2f}")
    with col3:
        st.metric(
            "高/中/低章节",
            f"{overall.get('high_count', 0)} / {overall.get('mid_count', 0)} / {overall.get('low_count', 0)}"
        )
    with col4:
        st.metric("论文总量", f"{overall.get('total_papers', 0)} 篇")

    st.markdown("---")

    #  Per-section expanders 
    st.subheader("逐段引用溯源")

    for sec in sections:
        name = sec["name"]
        credibility = sec["credibility"]
        paper_count = sec["paper_count"]
        avg_rel = sec.get("avg_relevance", 0)
        badge = format_credibility_badge(credibility)
        evidence = sec.get("evidence_papers", [])

        with st.expander(
            f"{badge} **{name}** — 证据论文 {paper_count} 篇，平均相关度 {avg_rel:.1%}"
        ):
            if not evidence:
                st.caption(" 未找到相关证据论文，该段可信度较低。")
            else:
                for ep in evidence:
                    link = ep.get("link", "#")
                    st.markdown(
                        f"- [{ep['title']}]({link}) "
                        f"— {ep.get('authors', 'N/A')} "
                        f"`相关度 {ep['relevance_score']:.3f}`"
                    )


# 
# PAGE 1: 深度搜索
# 
if page == "深度搜索":
    st.title("深度搜索")
    st.markdown("输入学术问题，系统自动扩展关键词、检索论文并生成综合回答。")

    # Initialize session state for this page
    if "ds_query" not in st.session_state:
        st.session_state.ds_query = ""
    if "ds_keywords" not in st.session_state:
        st.session_state.ds_keywords = []
    if "ds_ranked" not in st.session_state:
        st.session_state.ds_ranked = []
    if "ds_cards" not in st.session_state:
        st.session_state.ds_cards = []
    if "ds_weights" not in st.session_state:
        st.session_state.ds_weights = {}
    if "ds_top_n" not in st.session_state:
        st.session_state.ds_top_n = 10

    query = st.text_input(
        "输入学术问题",
        placeholder="例如：What are the latest advances in large language model reasoning?",
        key="ds_query_input",
    )

    col1, col2 = st.columns(2)
    with col1:
        papers_per_kw = st.slider("每关键词检索论文数", 5, 30, 15)
    with col2:
        top_n = st.slider("展示 Top N 论文", 5, 20, 10, key="ds_top_n_slider")

    # Weight settings
    weights = show_weight_settings(show_preferences=True)

    # Source selection
    selected_sources = show_source_selector(default_all=True)
    sources = _init_sources(selected_sources)

    #  Search button 
    if st.button("开始深度搜索", type="primary", disabled=not query.strip() or not selected_sources):
        st.session_state.ds_query = query.strip()
        st.session_state.ds_top_n = top_n
        st.session_state.ds_weights = weights
        st.session_state.ds_cards = []  # reset cards on new search
        st.session_state.pop("ds_report", None)  # reset report cache
        st.session_state.pop("ds_verification", None)  # reset verification cache

        keywords, ranked = run_search_pipeline(
            query.strip(),
            papers_per_kw,
            weights=weights,
            selected_preferences=weights.get("selected_preferences", []),
            sources=sources,
        )
        st.session_state.ds_keywords = keywords
        st.session_state.ds_ranked = ranked

    #  Display results if available 
    ranked = st.session_state.ds_ranked
    keywords = st.session_state.ds_keywords
    saved_weights = st.session_state.ds_weights
    saved_top_n = st.session_state.ds_top_n
    saved_query = st.session_state.ds_query

    if keywords or ranked:
        st.markdown("---")
        st.subheader("查询扩展结果")
        st.markdown("、".join([f"`{kw}`" for kw in keywords]))

        if not ranked:
            st.warning("未检索到相关论文。请检查网络连接，或尝试更换查询关键词。")
        else:
            st.markdown("---")
            st.subheader(f" Top {saved_top_n} 相关论文")

            if saved_weights:
                w = saved_weights
                st.caption(
                    f"排序公式: {w.get('w_rel', 0):.2f}×R + {w.get('w_fresh', 0):.2f}×T + "
                    f"{w.get('w_cite', 0):.2f}×C + {w.get('w_src', 0):.2f}×S + "
                    f"{w.get('w_qual', 0):.2f}×Q"
                )
            show_papers_table(ranked, saved_top_n, show_extra=True)

            # Expandable abstracts
            st.markdown("### 论文摘要")
            for i, p in enumerate(ranked[:min(saved_top_n, 15)], 1):
                pub_date = _format_pub_date(p.get("published"))
                with st.expander(f"#{i} {p.get('title', 'Untitled')}"):
                    st.caption(f"**作者**: {p.get('authors', 'N/A')}")
                    st.caption(
                        f"**发表**: {pub_date} "
                        f"| **最终得分**: {p.get('composite_score', 0):.4f}"
                    )
                    st.caption(
                        f"相关度 {p.get('relevance_score', 0):.3f} | "
                        f"新鲜度 {p.get('freshness_score', 0):.3f} | "
                        f"引用分 {p.get('citation_score', 0):.3f} | "
                        f"来源分 {p.get('source_score', 0):.3f} | "
                        f"质量分 {p.get('quality_score', 0):.3f}"
                    )
                    st.write(p.get("summary", "No abstract available."))
                    st.markdown(f"[查看论文]({p.get('link', '#')})")

            #  Paper reading cards 
            st.markdown("---")
            st.subheader("论文精读卡片")

            n_cards = min(saved_top_n, len(ranked))
            if st.button(f" 为 Top {n_cards} 篇论文生成精读卡片", type="secondary", key="btn_gen_cards"):
                with st.spinner(f"正在为 {n_cards} 篇论文生成结构化精读卡片..."):
                    cards = []
                    for p in ranked[:n_cards]:
                        card = generate_paper_card(p)
                        cards.append(card)
                    st.session_state.ds_cards = cards

            # Display cached cards
            if st.session_state.ds_cards:
                for i, card in enumerate(st.session_state.ds_cards, 1):
                    with st.expander(f" Top {i}: {card['title'][:80]}..."):
                        st.markdown(format_card_markdown(card))

            #  Report + credibility 
            if "ds_report" not in st.session_state or not st.session_state.get("ds_report"):
                # Generate once, cache
                result = generate_search_answer(saved_query, keywords, ranked, saved_top_n)
                st.session_state.ds_report = result["report"]
                st.session_state.ds_answer_sections = result["sections"]

                with st.spinner("正在进行引用溯源与可信度检查..."):
                    verification = assess_evidence(
                        user_query=saved_query,
                        top_papers=ranked,
                        generated_sections=result["sections"],
                    )
                st.session_state.ds_verification = verification

            st.markdown("---")
            show_credibility_panel(st.session_state.get("ds_verification", {}))
            st.markdown("---")
            st.subheader("综合回答")
            st.markdown(st.session_state.get("ds_report", ""))


# 
# PAGE 2: 深度调研
# 
elif page == "深度调研":
    st.title("深度调研")
    st.markdown("输入研究领域，系统自动拆分子方向并生成结构化调研报告。")

    area = st.text_input(
        "输入研究领域",
        placeholder="例如：Large Language Model / Graph Neural Network / Diffusion Model",
    )

    col1, col2 = st.columns(2)
    with col1:
        n_directions = st.slider("拆分子方向数", 3, 6, 5)
    with col2:
        papers_per_dir = st.slider("每子方向检索论文数", 5, 20, 10)

    # Weight settings
    weights = show_weight_settings(show_preferences=True)

    # Source selection
    selected_sources = show_source_selector(default_all=True)
    sources = _init_sources(selected_sources)

    #  AI enhanced report toggle 
    st.markdown("---")
    st.subheader("AI 增强报告")

    report_mode = st.radio(
        "报告生成模式",
        options=["模板报告", "单次 LLM 报告（证据锚定）", "数据驱动分章节 LLM 报告"],
        index=0,
        help=(
            "**模板报告**：基于规则模板的结构化报告，无需 LLM。\n\n"
            "**单次 LLM 报告**：LLM 一次性生成完整报告，所有事实性陈述锚定到 [P#] 论文编号。\n\n"
            "**数据驱动分章节 LLM 报告**：系统先规划章节结构并为每节选择论文，再逐节调用 LLM 生成，"
            "每节独立进行引用校验，最终拼接为完整报告。"
        ),
    )
    use_ai = report_mode != "模板报告"
    use_data_driven = report_mode == "数据驱动分章节 LLM 报告"

    llm_api_key = ""
    llm_model = "deepseek-chat"
    llm_base_url = "https://api.deepseek.com"

    if use_ai:
        llm_api_key = os.environ.get("LLM_API_KEY", "") or os.environ.get("OPENAI_API_KEY", "")
        if not llm_api_key:
            llm_api_key = st.text_input(
                "LLM API Key",
                type="password",
                placeholder="sk-... (支持 DeepSeek / OpenAI / Ollama 等兼容 API)",
                help="不会存储，仅本次会话使用。也可设置环境变量 LLM_API_KEY 或 OPENAI_API_KEY。",
            )
        with st.expander("LLM 高级设置", expanded=False):
            llm_model = st.text_input("模型名称", value="deepseek-chat", placeholder="deepseek-chat")
            llm_base_url = st.text_input("API Base URL", value="https://api.deepseek.com")

    #  Audit view toggle 
    col_toggle1, col_toggle2 = st.columns(2)
    with col_toggle1:
        show_audit = st.toggle(
            "开启审计视图",
            value=False,
            help="展示每篇论文编号的详细信息、引用使用情况、无效引用和无据陈述。",
        )
    with col_toggle2:
        deep_read = st.toggle(
            "精读模式（全文分析）",
            value=False,
            help="下载 Top N 论文的 arXiv PDF 全文，提取方法、实验等章节，生成精读卡片和阅读路径。需联网下载。",
        )

    if st.button("开始深度调研", type="primary", disabled=not area.strip() or not selected_sources):
        sub_dirs = generate_sub_directions(area.strip(), n=n_directions)

        st.markdown("---")
        st.subheader("子方向拆分")
        for i, sd in enumerate(sub_dirs, 1):
            st.markdown(f"{i}. {sd}")

        st.markdown("---")
        st.subheader("各子方向检索与排序")

        all_dir_papers = {}
        all_ranked = []

        progress = st.progress(0)
        for idx, sd in enumerate(sub_dirs):
            st.markdown(f"###  {sd}")
            _, ranked = run_search_pipeline(
                sd,
                papers_per_dir,
                weights=weights,
                selected_preferences=weights.get("selected_preferences", []),
                sources=sources,
            )
            all_dir_papers[sd] = ranked
            all_ranked.extend(ranked)
            progress.progress((idx + 1) / len(sub_dirs))

        # Dedup across all directions
        from modules.deduplicator import deduplicate as dedup_fn

        st.markdown("---")
        st.subheader("汇总结果（跨方向去重）")
        all_deduped = dedup_fn(all_ranked)
        st.write(f"总计 {len(all_ranked)} 篇（去重后 {len(all_deduped)} 篇）")

        #  Topic clustering 
        st.markdown("---")
        st.subheader("研究方向聚类分析")

        if len(all_deduped) >= 6:
            with st.spinner("正在进行 TF-IDF + KMeans 主题聚类..."):
                clustering_result = cluster_papers(all_deduped)

            clusters = clustering_result.get("clusters", [])
            n_clusters = clustering_result.get("n_clusters", 0)
            sil = clustering_result.get("silhouette_score", 0)

            # Cluster bar chart
            st.markdown(f"**自动识别 {n_clusters} 个主题簇**（轮廓系数: {sil:.3f}）")
            import pandas as pd
            chart_data = pd.DataFrame([
                {"主题簇": f"{c['id']+1}. {c['topic_name']}", "论文数量": c["paper_count"]}
                for c in clusters
            ])
            st.bar_chart(chart_data.set_index("主题簇"), use_container_width=True)

            # Per-cluster cards
            st.markdown("### 各主题详情")
            cols_per_row = 2
            for i in range(0, len(clusters), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, col in enumerate(cols):
                    idx = i + j
                    if idx >= len(clusters):
                        break
                    c = clusters[idx]
                    with col:
                        with st.container(border=True):
                            st.markdown(f"**主题 {c['id']+1}: {c['topic_name']}**")
                            st.caption(f"论文数: {c['paper_count']} 篇")
                            st.caption(f"关键词: {', '.join(c['keywords'])}")
                            rep = c.get("representative", {})
                            if rep:
                                st.caption(f"代表: [{rep['title'][:60]}...]({rep.get('link', '#')})")
        else:
            st.info(f"论文数量不足（当前 {len(all_deduped)} 篇，需 >= 6），跳过聚类分析。")
            clustering_result = None

        #  Research gap analysis (shared across all report modes) 
        with st.spinner("正在进行研究空白分析与选题建议..."):
            gap_result = analyze_research_gaps(
                research_area=area.strip(),
                papers=all_deduped,
                clustering_result=clustering_result,
                sub_directions=sub_dirs,
                all_papers=all_dir_papers,
            )

        # 
        # Deep reading mode (full-text PDF analysis)
        # 
        deep_cards: list[dict] = []
        deep_sections: list[dict | None] = []
        deep_paper_id_map: dict[str, dict] = {}
        deep_enriched_pack: dict | None = None

        if deep_read and all_deduped:
            st.markdown("---")
            st.subheader("精读模式 — 全文深度分析")

            top_n_deep = min(10, len(all_deduped))
            top_papers = sorted(
                all_deduped, key=lambda p: p.get("composite_score", 0) or 0, reverse=True
            )[:top_n_deep]

            # --- Download PDFs ---
            st.markdown("** 下载论文 PDF...**")
            download_status = st.empty()

            pdf_map: dict[str, Path | None] = {}  # keyed by arxiv_id
            # Also map paper index → pdf_path for second pass
            paper_pdf_map: dict[int, Path | None] = {}

            for i, p in enumerate(top_papers):
                aid = extract_arxiv_id(p)
                download_status.text(f"下载中 ({i+1}/{top_n_deep}): {p.get('title', '')[:60]}...")
                if aid:
                    pdf_map[aid] = download_arxiv_pdf(aid)
                    paper_pdf_map[i] = pdf_map[aid]
                else:
                    paper_pdf_map[i] = None

            success_count = sum(1 for v in paper_pdf_map.values() if v is not None)
            arxiv_avail = sum(1 for p in top_papers if extract_arxiv_id(p))
            download_status.text(
                f" 下载完成：{success_count} 篇成功 / {arxiv_avail} 篇有 arXiv ID / {top_n_deep} 篇候选"
            )

            # --- Extract text + parse sections + build cards ---
            st.markdown("** 解析全文并构建精读卡片...**")

            deep_papers: list[dict] = []
            extract_summary_lines: list[str] = []

            for i, p in enumerate(top_papers):
                pdf_path = paper_pdf_map.get(i)
                sections = None
                card: dict = {}
                extract_status = ""

                if pdf_path:
                    full_text, ft_error = extract_full_text(pdf_path)
                    if full_text and len(full_text) >= 100:
                        sections = parse_paper_sections(full_text)
                        card = build_deep_reading_card(p, sections, pdf_path)
                        extract_status = f" 全文 ({len(full_text)} 字符)"
                    else:
                        card = build_deep_reading_card(p, None, pdf_path)
                        extract_status = f" 提取失败: {ft_error or '文本过短'}"
                else:
                    aid = extract_arxiv_id(p)
                    if aid:
                        card = build_deep_reading_card(p, None, None)
                        extract_status = " PDF 下载失败"
                    else:
                        card = build_deep_reading_card(p, None, None)
                        extract_status = "⬜ 非 arXiv 论文"

                deep_cards.append(card)
                deep_sections.append(sections)
                deep_papers.append(p)
                extract_summary_lines.append(
                    f"- {extract_status} — **{p.get('title', '?')[:60]}**"
                )

                if sections and card.get("has_full_text"):
                    deep_paper_id_map[extract_arxiv_id(p) or str(i)] = {
                        "title": p.get("title", "未知"),
                        "authors": p.get("authors", "未知"),
                        "year": p.get("year", "未知"),
                        "venue": p.get("venue", "未知"),
                        "citation_count": p.get("citation_count", 0),
                        "url": p.get("url", ""),
                        "has_full_text": True,
                    }

            st.markdown("\n".join(extract_summary_lines))

            # --- Build enriched evidence pack ---
            if success_count > 0:
                deep_enriched_pack = build_fulltext_evidence_pack(
                    deep_papers, deep_cards, deep_sections, max_papers=top_n_deep,
                )

            # --- Show reading path ---
            if len(deep_cards) >= 2:
                reading_path = build_reading_path(deep_papers, deep_cards)
                st.markdown(format_reading_path_markdown(reading_path))

            # --- Show deep reading cards ---
            st.markdown("---")
            st.subheader("精读卡片")

            for i, card in enumerate(deep_cards):
                paper_type = card.get("paper_type", "")
                has_ft = card.get("has_full_text", False)
                ft_badge = "[全文]" if has_ft else "[摘要]"

                with st.expander(
                    f"{ft_badge} **{card['title'][:80]}**"
                    f"{'...' if len(card['title']) > 80 else ''}"
                    f" — {card.get('authors', '未知')[:40]} ({card.get('year', '未知')})",
                ):
                    st.markdown(format_deep_card_markdown(card))

        if use_data_driven and llm_api_key:
            # 
            # Data-driven section-by-section LLM report path
            # 
            st.markdown("---")
            st.subheader("数据驱动分章节报告")

            # Configure LLM
            llm_client, llm_error = try_configure_llm(
                api_key=llm_api_key, base_url=llm_base_url, model=llm_model
            )
            if llm_error:
                st.error(f"LLM 配置失败：{llm_error}")
            else:
                #  1. Build comparison matrix 
                with st.spinner("正在构建方法对比矩阵..."):
                    comparison_matrix = build_comparison_matrix(all_deduped)

                #  2. Build report plan 
                with st.spinner("正在生成数据驱动报告计划..."):
                    report_plan = build_data_driven_report_plan(
                        user_query=area.strip(),
                        papers=all_deduped,
                        topic_clusters=clustering_result,
                        comparison_matrix=comparison_matrix,
                        research_gaps=gap_result,
                    )

                # Show report plan
                with st.expander("报告计划", expanded=True):
                    for sp in report_plan.get("sections", []):
                        title = sp.get("template_title", "?")
                        n_papers = len(sp.get("selected_papers", []))
                        pids = [p.get("paper_id", "?") if isinstance(p, dict) else "?"
                                for p in sp.get("selected_papers", [])]
                        st.markdown(f"- **{title}**：{n_papers} 篇论文 ({', '.join(pids[:8])})")

                #  3. Section-by-section generation 
                with st.spinner("正在逐节生成报告（共 10 节）..."):
                    dd_result = generate_data_driven_report(
                        report_plan=report_plan,
                        llm_client=llm_client,
                    )

                report_text = dd_result["report_text"]
                section_results = dd_result.get("sections", [])

                #  Append gap report 
                if gap_result.get("gap_report"):
                    report_text += "\n\n" + gap_result["gap_report"]

                #  Show per-section results 
                st.markdown("---")
                st.subheader("逐节生成结果与引用审计")

                for sr in section_results:
                    title = sr.get("section_title", "?")
                    success = sr.get("success", False)
                    v = sr.get("validation", {})
                    risk = v.get("risk_level", "?")
                    stats = v.get("stats", {})
                    emoji = {"高风险": "", "中风险": "", "低风险": ""}.get(risk, "")

                    with st.expander(
                        f"{emoji} **{title}** — "
                        f"引用 {stats.get('unique_papers_cited', 0)} 篇 "
                        f"| 无据陈述 {stats.get('uncited_claim_count', 0)} 条 "
                        f"| 无效引用 {stats.get('invalid_ref_count', 0)} 条"
                    ):
                        if not success:
                            st.error(sr.get("error", "未知错误"))
                        else:
                            st.markdown(sr["section_text"])
                            uncited = v.get("uncited_claims", [])
                            if uncited:
                                st.markdown("** 无引用陈述：**")
                                for uc in uncited:
                                    st.caption(f"行 {uc['line']}: {uc['sentence'][:100]}")

                # Show overall validation
                st.markdown("---")
                st.markdown(f"### 整体引用审计 — 风险等级：{dd_result.get('risk_level', '?')}")
                vs = dd_result.get("validation_summary", {})
                st.markdown(
                    f"- 累计引用论文数：{vs.get('total_cited_papers', 0)}\n"
                    f"- 累计无据陈述：{vs.get('total_uncited_claims', 0)}\n"
                    f"- 累计无效引用：{vs.get('total_invalid_refs', 0)}"
                )

                #  Audit view (data-driven) 
                if show_audit:
                    st.markdown("---")
                    show_section_audit_rows(section_results)

                    # Build aggregate paper_id_map from all sections
                    all_section_papers: dict[str, dict] = {}
                    for sp in report_plan.get("sections", []):
                        for p in sp.get("selected_papers", []):
                            pid = p.get("paper_id", "")
                            if pid and pid not in all_section_papers:
                                all_section_papers[pid] = {
                                    "title": p.get("title", "未知"),
                                    "authors": p.get("authors", "未知"),
                                    "year": p.get("year", "未知"),
                                    "venue": p.get("venue", "未知"),
                                    "sources": p.get("sources", ["未知"]),
                                    "citation_count": p.get("citation_count", 0),
                                    "url": p.get("url", ""),
                                }

                    # Run overall validation on concatenated report
                    overall_valid_ids = set(all_section_papers.keys())
                    overall_validation = validate_report(report_text, overall_valid_ids)
                    show_audit_view(
                        paper_id_map=all_section_papers,
                        validation=overall_validation,
                        report_text=report_text,
                        heading="数据驱动报告 — 整体引用审计",
                    )

                # Show full report
                st.markdown("---")
                st.subheader("完整调研报告")
                st.markdown(report_text)

                # Download button
                st.download_button(
                    "下载调研报告 (Markdown)",
                    data=report_text,
                    file_name=f"survey_data_driven_{area.strip().replace(' ', '_')}.md",
                    mime="text/markdown",
                )

        elif use_ai and llm_api_key:
            # 
            # AI-enhanced evidence-anchored report path
            # 
            st.markdown("---")
            st.subheader("AI 增强报告")

            # Configure LLM
            llm_client, llm_error = try_configure_llm(
                api_key=llm_api_key, base_url=llm_base_url, model=llm_model
            )
            if llm_error:
                st.error(f"LLM 配置失败：{llm_error}")
            else:
                # Use enriched evidence pack if deep reading is active
                if deep_read and deep_enriched_pack:
                    st.info(" 正在使用全文精读证据包生成深度分析...")
                    from modules.prompt_templates import (
                        EVIDENCE_ANCHORED_SYSTEM_PROMPT,
                        build_evidence_anchored_user_prompt,
                    )
                    from modules.evidence_builder import format_evidence_pack_for_prompt
                    enriched_text = format_evidence_pack_for_prompt(deep_enriched_pack)
                    messages = [
                        {"role": "system", "content": EVIDENCE_ANCHORED_SYSTEM_PROMPT},
                        {"role": "user", "content": build_evidence_anchored_user_prompt(
                            user_query=area.strip(),
                            evidence_pack_text=enriched_text,
                        )},
                    ]
                    with st.spinner("正在基于全文精读生成深度调研报告..."):
                        report_text = llm_client.chat(messages, temperature=0.3, max_tokens=4096)
                    evidence_pack = deep_enriched_pack
                    paper_id_map = deep_enriched_pack.get("paper_id_map", {})
                else:
                    with st.spinner("正在生成证据锚定的 AI 调研报告..."):
                        ai_result = generate_evidence_anchored_report(
                            user_query=area.strip(),
                            papers=all_deduped,
                            llm_client=llm_client,
                            max_papers=15,
                        )
                    report_text = ai_result["report_text"]
                    evidence_pack = ai_result["evidence_pack"]
                    paper_id_map = ai_result["paper_id_map"]

                #  Append gap report 
                gap_report = gap_result.get("gap_report", "")
                if gap_report:
                    report_text += "\n\n" + gap_report

                #  Citation validation 
                with st.spinner("正在进行引用审计..."):
                    valid_ids = set(evidence_pack.get("paper_id_map", {}).keys())
                    validation = validate_report(report_text, valid_ids)

                # Show validation summary
                st.markdown("---")
                st.markdown(format_validation_summary(validation))

                # Show evidence pack
                with st.expander("证据包（论文编号映射）", expanded=False):
                    for pid, meta in paper_id_map.items():
                        st.markdown(
                            f"- **{pid}**: [{meta.get('title', '未知')}]({meta.get('url', '#')}) "
                            f"— {meta.get('authors', '未知')} ({meta.get('year', '未知')}) "
                            f"| 引用 {meta.get('citation_count', 0)}"
                        )

                #  Audit view (single-pass AI) 
                if show_audit:
                    st.markdown("---")
                    show_audit_view(
                        paper_id_map=paper_id_map,
                        validation=validation,
                        report_text=report_text,
                        heading="单次 LLM 报告 — 引用审计",
                    )

                # Show uncited claims if any (only when audit is off, to avoid duplication)
                if not show_audit:
                    uncited = validation.get("uncited_claims", [])
                    if uncited:
                        with st.expander(f"无引用陈述详情（{len(uncited)} 条）", expanded=False):
                            for uc in uncited:
                                st.markdown(f"- **行 {uc['line']}**: {uc['sentence']}")
                                st.caption(f"原因：{uc['reason']}")

                # Show report
                st.markdown("---")
                st.subheader("调研报告")
                st.markdown(report_text)

                # Download button
                st.download_button(
                    "下载调研报告 (Markdown)",
                    data=report_text,
                    file_name=f"survey_ai_{area.strip().replace(' ', '_')}.md",
                    mime="text/markdown",
                )

        else:
            # 
            # Template-based report path (unchanged)
            # 
            if use_ai and not llm_api_key:
                st.warning(" 未配置 LLM API Key，已回退为模板报告。请设置环境变量 LLM_API_KEY 或在页面中输入 API Key。")

            #  Generate report 
            st.markdown("---")
            result = generate_survey_report(
                area.strip(), sub_dirs, all_dir_papers,
                clustering_result=clustering_result,
            )
            report = result["report"]

            #  Append gap report 
            gap_report = gap_result.get("gap_report", "")
            if gap_report:
                report += "\n\n" + gap_report

            # Credibility check
            with st.spinner("正在进行引用溯源与可信度检查..."):
                all_survey_papers = []
                for papers in all_dir_papers.values():
                    all_survey_papers.extend(papers)
                all_survey_papers = dedup_fn(all_survey_papers)

                verification = assess_evidence(
                    user_query=area.strip(),
                    top_papers=all_survey_papers,
                    generated_sections=result["sections"],
                )

            # Show credibility panel
            st.markdown("---")
            show_credibility_panel(verification)

            # Show report
            st.markdown("---")
            st.subheader("调研报告")
            st.markdown(report)

            # Download button
            st.download_button(
                "下载调研报告 (Markdown)",
                data=report,
                file_name=f"survey_{area.strip().replace(' ', '_')}.md",
                mime="text/markdown",
            )


# 
# PAGE 3: 个性化推送
# 
elif page == "个性化推送":
    st.title("个性化推送")
    st.markdown("输入 1-10 个科研方向，生成今日学术进展日报。")

    #  User preference panel (prominent on this page) 
    st.subheader("用户偏好设置")
    st.caption(
        "勾选关注方向后，系统会自动调整排序权重，并计算主题偏好得分。"
    )

    pref_col1, pref_col2, pref_col3, pref_col4 = st.columns(4)
    with pref_col1:
        focus_latest = st.checkbox("关注最新进展", value=True,
                                   help="自动提高新鲜度权重")
    with pref_col2:
        focus_quality = st.checkbox("关注高质量论文", value=False,
                                    help="自动提高质量分权重")
    with pref_col3:
        focus_fresh = st.checkbox("关注综述论文", value=False)
    with pref_col4:
        focus_method = st.checkbox("关注方法创新", value=False)

    pref_col5, pref_col6, _, _ = st.columns(4)
    with pref_col5:
        focus_application = st.checkbox("关注应用系统", value=False)
    with pref_col6:
        focus_custom = st.checkbox(" 自定义权重", value=False)

    # Smart weight presets
    if focus_custom:
        custom_defaults = get_default_weights()
        if focus_latest:
            custom_defaults["w_fresh"] = 0.50
            custom_defaults["w_rel"] = 0.25
            custom_defaults["w_cite"] = 0.10
            custom_defaults["w_src"] = 0.05
            custom_defaults["w_qual"] = 0.10
        if focus_quality:
            custom_defaults["w_qual"] = max(custom_defaults["w_qual"], 0.25)
            custom_defaults["w_cite"] = max(custom_defaults["w_cite"], 0.25)
        weights = show_weight_settings(defaults=custom_defaults, show_preferences=True)
    else:
        # Auto weights based on checkboxes
        w_rel = 0.40
        w_fresh = 0.45 if focus_latest else 0.20
        w_cite = 0.30 if (focus_fresh or focus_quality) else 0.20
        w_src = 0.10
        w_qual = 0.30 if focus_quality else 0.10

        # Balance: if no special prefs, go back to defaults
        if not focus_latest and not focus_quality and not focus_fresh and not focus_method and not focus_application:
            d = get_default_weights()
            w_rel, w_fresh, w_cite, w_src, w_qual = d["w_rel"], d["w_fresh"], d["w_cite"], d["w_src"], d["w_qual"]

        n_rel, n_fresh, n_cite, n_src, n_qual = normalize_weights(
            w_rel, w_fresh, w_cite, w_src, w_qual
        )
        st.caption(
            f"自动权重 → w_r={n_rel:.2f}  w_t={n_fresh:.2f}  "
            f"w_c={n_cite:.2f}  w_s={n_src:.2f}  w_q={n_qual:.2f}"
        )
        st.caption("勾选「自定义权重」可手动调整")

        weights = {
            "w_rel": n_rel,
            "w_fresh": n_fresh,
            "w_cite": n_cite,
            "w_src": n_src,
            "w_qual": n_qual,
            "selected_preferences": [],
        }

    st.markdown("---")

    # Source selection
    selected_sources = show_source_selector(default_all=True)
    sources = _init_sources(selected_sources)

    #  Interest input 
    col_input, col_info = st.columns([3, 1])
    with col_input:
        interests_text = st.text_area(
            "输入科研方向（每行一个，或逗号分隔）",
            placeholder="例如：\nLarge Language Model\nReinforcement Learning\nGraph Neural Network",
            height=120,
        )
    with col_info:
        st.caption(" 提示")
        st.caption("每行输入一个研究方向，系统将分别检索各方向最新论文并生成日报。")

    col1, col2 = st.columns(2)
    with col1:
        papers_per_interest = st.slider("每方向检索论文数", 5, 30, 15, key="feed_per_kw")
    with col2:
        top_n = st.slider("日报 Top N", 5, 20, 10, key="feed_top_n")

    if st.button("生成今日日报", type="primary", disabled=not interests_text.strip() or not selected_sources):
        # Parse interests
        if "\n" in interests_text:
            interests = [line.strip() for line in interests_text.split("\n") if line.strip()]
        else:
            interests = [s.strip() for s in interests_text.replace("，", ",").split(",") if s.strip()]
        interests = interests[:10]

        if not interests:
            st.error("请输入至少 1 个研究方向。")
        else:
            st.markdown("---")
            st.subheader(f"关注方向（共 {len(interests)} 个）")
            st.markdown("、".join([f"`{d}`" for d in interests]))

            # Search each interest
            st.markdown("---")
            st.subheader("各方向检索")

            all_by_dir = {}
            all_papers = []

            progress = st.progress(0)
            for idx, interest in enumerate(interests):
                st.markdown(f"### {interest}")
                keywords, ranked = run_search_pipeline(
                    interest,
                    papers_per_interest,
                    weights=weights,
                    selected_preferences=weights.get("selected_preferences", []),
                    sources=sources,
                )
                all_by_dir[interest] = ranked
                all_papers.extend(ranked)
                progress.progress((idx + 1) / len(interests))

            # Global dedup and rerank
            st.markdown("---")
            st.subheader("全局去重与综合排序")

            from modules.deduplicator import deduplicate as dedup_fn

            global_deduped = dedup_fn(all_papers)
            global_query = " ".join(interests)
            pref_keys = get_preference_keys(weights.get("selected_preferences", []))
            global_ranked = compute_composite(
                global_deduped,
                global_query,
                w_rel=weights["w_rel"],
                w_fresh=weights["w_fresh"],
                w_cite=weights.get("w_cite", 0.20),
                w_src=weights.get("w_src", 0.10),
                w_qual=weights["w_qual"],
                selected_preferences=pref_keys,
            )

            st.success(f"去重后共 {len(global_ranked)} 篇论文")

            # Show formula
            w = weights
            st.caption(
                f"排序公式: {w.get('w_rel', 0):.2f}×R + {w.get('w_fresh', 0):.2f}×T + "
                f"{w.get('w_cite', 0):.2f}×C + {w.get('w_src', 0):.2f}×S + "
                f"{w.get('w_qual', 0):.2f}×Q"
            )

            # Show per-direction stats
            st.markdown("### 各方向论文数量")
            for interest, papers in all_by_dir.items():
                st.metric(label=interest, value=f"{len(papers)} 篇")

            # Top 10 table with all scores
            st.markdown("---")
            st.subheader(f"今日推荐 Top {top_n}")
            show_papers_table(global_ranked, top_n, show_extra=True)

            # Per-paper score breakdown chart
            st.markdown("###  得分维度拆解")
            if len(global_ranked) >= 5:
                import pandas as pd
                chart_data = []
                for p in global_ranked[:top_n]:
                    title_short = p["title"][:60]
                    chart_data.append({
                        "论文": title_short,
                        "相关度": p.get("relevance_score", 0),
                        "新鲜度": p.get("freshness_score", 0),
                        "质量分": p.get("quality_score", 0),
                        "偏好分": p.get("preference_score", 0),
                    })
                df_chart = pd.DataFrame(chart_data)
                st.bar_chart(df_chart.set_index("论文"), use_container_width=True)

            # Top 3 deep interpretation
            st.markdown("---")
            st.subheader("Top 3 重点论文解读")

            tabs = st.tabs([f"Top {i}" for i in range(1, 4)])
            for i, tab in enumerate(tabs):
                with tab:
                    if i < len(global_ranked):
                        interp = generate_deep_interpretation(global_ranked[i], i + 1)
                        st.markdown(interp)
                    else:
                        st.info("论文数量不足。")

            # Daily digest
            st.markdown("---")
            digest = generate_daily_digest(interests, all_by_dir, global_ranked, top_n)
            st.markdown(digest)

            st.download_button(
                "下载日报 (Markdown)",
                data=digest,
                file_name="daily_digest.md",
                mime="text/markdown",
            )


# 
# PAGE 4: 实验评估
# 
elif page == "实验评估":
    st.title("实验评估")
    st.markdown("通过对比实验验证查询扩展、个性化排序、可信度检查和主题聚类的效果。")

    exp_tab1, exp_tab2, exp_tab3, exp_tab4, exp_tab5 = st.tabs([
        "实验一：查询扩展", "实验二：排序对比", "实验三：个性化权重",
        "实验四：主题聚类", "实验五：多源对比",
    ])

    #  Shared helper for experiment paper tables 
    def _show_exp_papers(papers, n=5):
        if not papers:
            st.warning("无数据。")
            return
        rows = []
        for i, p in enumerate(papers[:n], 1):
            rows.append({
                "排名": i,
                "论文标题": p["title"][:80] + ("..." if len(p["title"]) > 80 else ""),
                "相关度": f"{p.get('relevance_score', 0):.3f}",
                "新鲜度": f"{p.get('freshness_score', 0):.3f}",
                "引用分": f"{p.get('citation_score', 0):.3f}",
                "来源分": f"{p.get('source_score', 0):.3f}",
                "质量分": f"{p.get('quality_score', 0):.3f}",
                "最终得分": f"{p.get('composite_score', 0):.4f}",
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)

    # 
    # Tab 1 — Query expansion experiment
    # 
    with exp_tab1:
        st.subheader("实验一：查询扩展效果实验")
        st.caption("对比「原始查询」与「多关键词扩展查询」的检索效果差异。")

        exp1_query = st.text_input(
            "输入研究问题或关键词",
            placeholder="例如：large language model reasoning",
            key="exp1_query",
        )
        exp1_n = st.slider("每关键词检索论文数", 5, 20, 12, key="exp1_n")

        if st.button("运行实验一", type="primary", disabled=not exp1_query.strip(), key="btn_exp1"):
            with st.spinner("实验中... 分别执行原始查询和查询扩展检索..."):
                result = evaluate_query_expansion(exp1_query.strip(), papers_per_kw=exp1_n)

            st.success("实验完成！")

            # Keywords display
            st.markdown("**扩展关键词**: " + "、".join([f"`{kw}`" for kw in result["keywords"]]))

            # Comparison table
            st.markdown("### 指标对比表")
            import pandas as pd
            df = pd.DataFrame(result["comparison"])
            st.dataframe(df.set_index("指标"), use_container_width=True)

            # Bar chart
            st.markdown("### 指标对比柱状图")
            chart_df = pd.DataFrame(result["chart_data"])
            st.bar_chart(chart_df.set_index("指标"), use_container_width=True)

            # Interpretation
            with st.expander("结果解读"):
                comp = result["comparison"]
                raw_total = comp["原始查询"][0]
                exp_total = comp["原始查询"][1] if len(comp["原始查询"]) > 1 else 0
                st.markdown(
                    f"- 查询扩展将检索论文数从 **{raw_total}** 扩展到 **{result['chart_data']['查询扩展'][0]}** 篇。\n"
                    f"- 查询扩展保留了更相关的论文（相关度提升、新鲜度改善）。\n"
                    f"- 这验证了多关键词并行检索 + 去重排序能有效提升结果覆盖度和相关性。"
                )

            # Show exp-ranked papers
            st.markdown("---")
            st.markdown("### 查询扩展 Top 10 结果")
            _show_exp_papers(result["exp_ranked"], n=10)

    # 
    # Tab 2 — Ranking method comparison
    # 
    with exp_tab2:
        st.subheader("实验二：排序方法对比实验")
        st.caption("对同一批论文，比较「按时间」「按相关度」「综合排序」三种策略的 Top5 质量。")

        exp2_query = st.text_input(
            "输入研究方向",
            placeholder="例如：graph neural network",
            key="exp2_query",
        )
        exp2_n = st.slider("检索论文数", 10, 40, 25, key="exp2_n")

        if st.button("运行实验二", type="primary", disabled=not exp2_query.strip(), key="btn_exp2"):
            with st.spinner("实验中... 检索论文 → 三种方式排序 → 指标对比..."):
                from modules.arxiv_client import search_papers as _sp
                from modules.deduplicator import deduplicate as _dedup

                raw = _sp(exp2_query.strip(), max_results=exp2_n)
                deduped = _dedup(raw)
                result = compare_ranking_methods(deduped, exp2_query.strip())

            st.success(f"实验完成！基于 {len(deduped)} 篇去重论文。")

            # Comparison table
            st.markdown("### 指标对比表")
            import pandas as pd
            df = pd.DataFrame(result["comparison"])
            st.dataframe(df.set_index("指标"), use_container_width=True)

            # Bar chart
            st.markdown("### 指标对比柱状图")
            chart_df = pd.DataFrame(result["comparison"])
            st.bar_chart(chart_df.set_index("指标"), use_container_width=True)

            # Three result columns
            st.markdown("---")
            st.markdown("### Top5 推荐结果对比")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.markdown("**按时间排序**")
                _show_exp_papers(result["date_ranked"], n=5)
            with col_b:
                st.markdown("**按相关度排序**")
                _show_exp_papers(result["relevance_ranked"], n=5)
            with col_c:
                st.markdown("**综合排序**")
                _show_exp_papers(result["composite_ranked"], n=5)

            with st.expander("结果解读"):
                st.markdown(
                    "- **按时间排序**偏向最新论文，新鲜度最高，但相关度可能较低。\n"
                    "- **按相关度排序**最匹配查询意图，但可能遗漏高质量经典论文。\n"
                    "- **综合排序**在相关度、新鲜度和质量之间取得平衡，效果最优。"
                )

    # 
    # Tab 3 — Personalized weight comparison
    # 
    with exp_tab3:
        st.subheader("实验三：个性化权重对比实验")
        st.caption("模拟三类用户画像（关注最新进展 / 综述论文 / 方法创新）对同批论文生成不同推荐。")

        exp3_query = st.text_input(
            "输入研究方向",
            placeholder="例如：large language model",
            key="exp3_query",
        )
        exp3_n = st.slider("检索论文数", 10, 40, 25, key="exp3_n")

        st.markdown("**三类用户画像权重配置：**")
        st.markdown(
            "| 画像 | 相关度 w_r | 新鲜度 w_t | 引用 w_c | 来源 w_s | 质量 w_q |\n"
            "|------|-----------|-----------|---------|---------|----------|\n"
            "| A-关注最新进展 | 0.20 | **0.50** | 0.10 | 0.10 | 0.10 |\n"
            "| B-关注高质量来源 | 0.25 | 0.10 | **0.30** | **0.20** | 0.15 |\n"
            "| C-关注方法创新 | **0.55** | 0.10 | 0.10 | 0.10 | 0.15 |"
        )

        if st.button("运行实验三", type="primary", disabled=not exp3_query.strip(), key="btn_exp3"):
            with st.spinner("实验中... 检索 → 三种画像分别排序..."):
                from modules.arxiv_client import search_papers as _sp
                from modules.deduplicator import deduplicate as _dedup

                raw = _sp(exp3_query.strip(), max_results=exp3_n)
                deduped = _dedup(raw)
                result = evaluate_personalized_ranking(deduped, exp3_query.strip())

            st.success(f"实验完成！基于 {len(deduped)} 篇去重论文。")

            # Overlap display
            st.markdown("### Top5 推荐重合度")
            ov = result["overlap"]
            col_o1, col_o2, col_o3, col_o4 = st.columns(4)
            col_o1.metric("A ∩ B", ov["A ∩ B"])
            col_o2.metric("A ∩ C", ov["A ∩ C"])
            col_o3.metric("B ∩ C", ov["B ∩ C"])
            col_o4.metric("A ∩ B ∩ C", ov["A ∩ B ∩ C"])

            # Metric comparison
            st.markdown("### 指标对比表")
            import pandas as pd
            df = pd.DataFrame(result["comparison"])
            st.dataframe(df.set_index("指标"), use_container_width=True)

            # Bar chart
            st.markdown("### 指标对比柱状图")
            chart_df = pd.DataFrame(result["comparison"])
            st.bar_chart(chart_df.set_index("指标"), use_container_width=True)

            # Three profiles — tabbed display
            st.markdown("---")
            st.markdown("### 三类用户 Top5 推荐结果")
            profile_tabs = st.tabs(["A-关注最新进展", "B-关注高质量来源", "C-关注方法创新"])
            for i, (tab, (label, ranked)) in enumerate(zip(
                profile_tabs,
                [("A", result["profile_a"]), ("B", result["profile_b"]), ("C", result["profile_c"])],
            )):
                with tab:
                    st.caption(f"**{label}用户画像** — Top5 推荐论文")
                    _show_exp_papers(ranked, n=5)

            with st.expander("结果解读"):
                st.markdown(
                    "- **重合度低**说明个性化权重确实改变了推荐结果，不同用户看到不同论文。\n"
                    "- **A 用户**（关注最新进展）新鲜度高，适合追踪前沿动态。\n"
                    "- **B 用户**（关注高质量来源）引用分和来源分最高，推荐高影响力论文。\n"
                    "- **C 用户**（关注方法创新）相关度最高，确保方法方向精准匹配。\n"
                    "- 这验证了个性化权重能有效满足不同用户的信息需求。"
                )

    # 
    # Tab 4 — Topic clustering experiment
    # 
    with exp_tab4:
        st.subheader("实验四：主题聚类分析实验")
        st.caption("对检索到的论文进行 TF-IDF + KMeans 聚类，自动发现研究方向结构。")

        exp4_query = st.text_input(
            "输入研究领域",
            placeholder="例如：deep learning",
            key="exp4_query",
        )
        exp4_n = st.slider("检索论文数", 15, 50, 30, key="exp4_n")

        if st.button("运行实验四", type="primary", disabled=not exp4_query.strip(), key="btn_exp4"):
            with st.spinner("实验中... 检索 → 聚类分析..."):
                from modules.arxiv_client import search_papers as _sp
                from modules.deduplicator import deduplicate as _dedup

                raw = _sp(exp4_query.strip(), max_results=exp4_n)
                deduped = _dedup(raw)
                result = evaluate_topic_clustering(deduped)

            if result["n_clusters"] == 0:
                st.warning(result["summary"])
            else:
                st.success(
                    f"聚类完成！识别出 **{result['n_clusters']}** 个主题簇"
                    f"（轮廓系数: {result['silhouette']:.3f}）。"
                )

                # Summary
                st.markdown(f"> {result['summary']}")

                # Bar chart
                st.markdown("### 主题分布柱状图")
                import pandas as pd
                chart_df = pd.DataFrame(result["chart_data"])
                st.bar_chart(chart_df.set_index("主题簇"), use_container_width=True)

                # Per-cluster detail
                st.markdown("### 各主题簇详情")
                for c in result["clusters"]:
                    with st.expander(
                        f"**主题 {c['id']+1}: {c['topic_name']}** — "
                        f"{c['paper_count']} 篇 | 关键词: {', '.join(c['keywords'])}"
                    ):
                        rep = c.get("representative", {})
                        if rep:
                            st.markdown(f"**代表论文**: [{rep['title']}]({rep.get('link', '#')})")
                            st.caption(rep.get("summary", "")[:300] + "...")
                        st.markdown("**包含论文**:")
                        for p in c["papers"][:5]:
                            st.markdown(f"- [{p['title']}]({p.get('link', '#')})")
                        if c["paper_count"] > 5:
                            st.caption(f"...及其他 {c['paper_count'] - 5} 篇")

                with st.expander("结果解读"):
                    st.markdown(
                        f"- 文献自动聚为 **{result['n_clusters']}** 个主题簇。\n"
                        f"- 轮廓系数 **{result['silhouette']:.3f}** 反映聚类质量。\n"
                        "- 聚类结果可以帮助快速了解一个研究领域的子方向结构，\n"
                        "  为深度调研提供可参考的主题划分依据。"
                    )

    # 
    # Tab 5 — Multi-source comparison experiment
    # 
    with exp_tab5:
        st.subheader("实验五：多源数据源对比实验")
        st.caption(
            "对同一查询，分别调用 arXiv、OpenAlex、Semantic Scholar，"
            "对比各数据源的覆盖范围、数据质量与重合度。"
        )

        exp5_query = st.text_input(
            "输入研究问题或关键词",
            placeholder="例如：transformer attention mechanism",
            key="exp5_query",
        )
        exp5_n = st.slider("每源检索论文数", 5, 30, 15, key="exp5_n")

        if st.button("运行实验五", type="primary", disabled=not exp5_query.strip(), key="btn_exp5"):
            with st.spinner("实验中... 并行调用 arXiv / OpenAlex / Semantic Scholar..."):
                result = evaluate_multi_source(exp5_query.strip(), max_results=exp5_n)

            #  Source errors 
            errors = result.get("source_errors", {})
            failed_sources = {k: v for k, v in errors.items() if v}
            if failed_sources:
                st.warning("以下数据源调用异常（其他数据源结果正常）：")
                for name, err in failed_sources.items():
                    st.caption(f"- **{name}**: {err}")
            else:
                st.success("三个数据源均调用成功！")

            #  Per-source stats table 
            st.markdown("---")
            st.subheader("各数据源统计")
            import pandas as pd
            df = pd.DataFrame(result["comparison"])
            st.dataframe(df.set_index("指标"), use_container_width=True)

            #  Bar charts 
            st.markdown("###  指标对比柱状图")
            chart_df = pd.DataFrame(result["chart_data"])

            col_c1, col_c2 = st.columns(2)
            with col_c1:
                st.markdown("**返回论文数**")
                count_df = chart_df[chart_df["指标"] == "返回论文数"]
                if not count_df.empty:
                    melted = count_df.melt(
                        id_vars=["指标"], var_name="数据源", value_name="论文数"
                    )
                    st.bar_chart(melted.set_index("数据源")[["论文数"]], use_container_width=True)

                st.markdown("**DOI 缺失率 (%)**")
                doi_df = chart_df[chart_df["指标"] == "DOI缺失率(%)"]
                if not doi_df.empty:
                    melted = doi_df.melt(
                        id_vars=["指标"], var_name="数据源", value_name="缺失率"
                    )
                    st.bar_chart(melted.set_index("数据源")[["缺失率"]], use_container_width=True)

            with col_c2:
                st.markdown("**平均引用数**")
                cit_df = chart_df[chart_df["指标"] == "平均引用数"]
                if not cit_df.empty:
                    melted = cit_df.melt(
                        id_vars=["指标"], var_name="数据源", value_name="平均引用数"
                    )
                    st.bar_chart(melted.set_index("数据源")[["平均引用数"]], use_container_width=True)

                st.markdown("**摘要缺失率 (%)**")
                abs_df = chart_df[chart_df["指标"] == "摘要缺失率(%)"]
                if not abs_df.empty:
                    melted = abs_df.melt(
                        id_vars=["指标"], var_name="数据源", value_name="缺失率"
                    )
                    st.bar_chart(melted.set_index("数据源")[["缺失率"]], use_container_width=True)

            #  Merge stats 
            st.markdown("---")
            st.subheader("合并与去重统计")
            merge = result["merge_stats"]

            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.metric("合并前论文总数", merge["total_raw"])
            col_m2.metric("去重后论文数", merge["total_deduped"])
            col_m3.metric("重复论文数", merge["total_duplicates"])
            col_m4.metric("重复率", f"{merge['dedup_rate']}%")

            #  Overlap 
            st.markdown("---")
            st.subheader("数据源重合度")
            ov = result["overlap"]

            col_o1, col_o2, col_o3, col_o4 = st.columns(4)
            col_o1.metric("arXiv ∩ OpenAlex", ov.get("arxiv ∩ openalex", 0))
            col_o2.metric("arXiv ∩ S2", ov.get("arxiv ∩ semantic_scholar", 0))
            col_o3.metric("OpenAlex ∩ S2", ov.get("openalex ∩ semantic_scholar", 0))
            col_o4.metric("三源重合", ov.get("三源重合", 0))

            #  Interpretation 
            with st.expander("结果解读"):
                interp_lines = [
                    "- **arXiv** 覆盖预印本论文，不含引用数，无 DOI 缺失问题（arXiv ID 即标识符）。",
                    "- **OpenAlex** 覆盖正式出版物（期刊/会议），引用数最完整，DOI 缺失率最低。",
                    "- **Semantic Scholar** 引用数包含 influential citations，但摘要缺失率可能较高。",
                ]
                # Dynamic observations
                src_stats = result.get("source_stats", {})
                arxiv_ct = src_stats.get("arxiv", {}).get("count", 0)
                oa_ct = src_stats.get("openalex", {}).get("count", 0)
                s2_ct = src_stats.get("semantic_scholar", {}).get("count", 0)
                max_ct = max(arxiv_ct, oa_ct, s2_ct)
                if max_ct > 0:
                    if arxiv_ct == max_ct:
                        interp_lines.append(f"- arXiv 本次返回最多论文（{arxiv_ct} 篇），适合发现最新预印本研究。")
                    if oa_ct == max_ct:
                        interp_lines.append(f"- OpenAlex 本次返回最多论文（{oa_ct} 篇），覆盖正式出版物范围广。")
                    if s2_ct == max_ct:
                        interp_lines.append(f"- Semantic Scholar 本次返回最多论文（{s2_ct} 篇），检索覆盖度表现好。")
                dup_rate = merge.get("dedup_rate", 0)
                if dup_rate > 0:
                    interp_lines.append(f"- 三源合并后重复率 {dup_rate}%，{'多源检索有效去除了冗余论文' if dup_rate > 10 else '各源间重合度较低，多源互补效果明显'}。")
                if any(errors.values()):
                    interp_lines.append("- 部分数据源调用失败，建议检查网络或 API 限额后重试。")
                interp_lines.append("- **建议**：生产环境推荐三源全选以获得最全覆盖；arXiv + OpenAlex 组合通常已能覆盖大部分需求。")
                st.markdown("\n".join(interp_lines))
