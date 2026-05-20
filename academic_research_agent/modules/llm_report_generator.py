"""LLM-enhanced report generator — evidence-anchored survey report.

Integrates evidence_builder, prompt_templates, and llm_client to produce
a fully cited academic survey report.  Supports both:
- Single-pass evidence-anchored generation
- Data-driven section-by-section generation
"""

from modules.evidence_builder import build_evidence_pack, format_evidence_pack_for_prompt
from modules.prompt_templates import (
    EVIDENCE_ANCHORED_SYSTEM_PROMPT,
    build_evidence_anchored_user_prompt,
)
from modules.llm_client import LLMClient
from modules.report_validator import validate_report
from modules.section_templates import SectionTemplate


def generate_evidence_anchored_report(
    user_query: str,
    papers: list[dict],
    llm_client: LLMClient,
    report_plan: str | None = None,
    max_papers: int = 15,
    temperature: float = 0.3,
    max_tokens: int = 4096,
) -> dict:
    """Generate an evidence-anchored survey report using LLM.

    Pipeline:
    1. Build evidence pack from top papers (assign P1..Pn IDs, extract findings).
    2. Fill the evidence-anchored prompt template.
    3. Call the LLM.
    4. Return the report text together with the evidence pack and paper-id map
       (for downstream validation and rendering).

    Parameters
    ----------
    user_query : str
        The research area or question.
    papers : list[dict]
        Ranked paper list (top papers will be used).  Should already be scored.
    llm_client : LLMClient
        Configured LLM client.
    report_plan : str | None
        Optional user-specified report structure / section outline.
    max_papers : int
        Maximum papers in the evidence pack (default 15).
    temperature : float
        LLM sampling temperature.
    max_tokens : int
        Max output tokens.

    Returns
    -------
    dict with keys:
        report_text : str
            The LLM-generated markdown report.
        evidence_pack : dict
            The evidence pack used (for display).
        paper_id_map : dict
            P1 → {title, authors, year, venue, citation_count, url}.
    """
    if not papers:
        return {
            "report_text": f"## 调研报告：{user_query}\n\n未检索到相关论文，无法生成报告。",
            "evidence_pack": {"paper_id_map": {}, "papers": [], "total_papers": 0},
            "paper_id_map": {},
        }

    # 1. Build evidence pack
    evidence_pack = build_evidence_pack(papers, max_papers=max_papers)
    evidence_text = format_evidence_pack_for_prompt(evidence_pack)

    # 2. Build prompts
    system_prompt = EVIDENCE_ANCHORED_SYSTEM_PROMPT
    user_prompt = build_evidence_anchored_user_prompt(
        user_query=user_query,
        evidence_pack_text=evidence_text,
        report_plan=report_plan,
    )

    # 3. Call LLM
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    report_text = llm_client.chat(
        messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # 4. Return
    return {
        "report_text": report_text,
        "evidence_pack": evidence_pack,
        "paper_id_map": evidence_pack.get("paper_id_map", {}),
    }


# ── Backward-compatible placeholder ──────────────────────────────────────

def generate_template_report_fallback(
    user_query: str,
    papers: list[dict],
) -> dict:
    """Fallback when LLM is unavailable — returns a simple template report
    with the same structure as the LLM version so frontend code can consume it.
    """
    from modules.evidence_builder import build_evidence_pack

    evidence_pack = build_evidence_pack(papers, max_papers=15)

    n = evidence_pack["total_papers"]
    lines = [
        f"## 调研报告：{user_query}",
        "",
        f"*基于 {n} 篇论文的模板化报告（未接入 LLM）*",
        "",
        "### 代表性论文",
        "",
    ]
    for ep in evidence_pack.get("papers", [])[:10]:
        lines.append(
            f"- **{ep['paper_id']}**: {ep['title']} — {ep['authors']} ({ep['year']})"
        )
        if ep["key_findings"]:
            for kf in ep["key_findings"][:2]:
                lines.append(f"  - {kf[:120]}")
        lines.append("")

    lines.append("---")
    lines.append("*启用「AI 增强报告」以生成含证据锚定的完整调研报告。*")

    return {
        "report_text": "\n".join(lines),
        "evidence_pack": evidence_pack,
        "paper_id_map": evidence_pack.get("paper_id_map", {}),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Data-driven section-by-section generation
# ═══════════════════════════════════════════════════════════════════════════

def generate_section_with_llm(
    section_template: SectionTemplate,
    section_papers: list[dict],
    section_data: dict,
    llm_client: LLMClient,
    temperature: float = 0.3,
    max_tokens: int = 2048,
) -> dict:
    """Generate a single report section using LLM.

    Parameters
    ----------
    section_template : SectionTemplate
        The section template defining prompts and format.
    section_papers : list[dict]
        Papers selected for this section (already with P# IDs).
    section_data : dict
        Section-specific data for filling the prompt template.
    llm_client : LLMClient
        Configured LLM client.
    temperature : float
        LLM sampling temperature.
    max_tokens : int
        Max output tokens.

    Returns
    -------
    dict with keys:
        section_title : str
        section_text : str
        validation : dict
        success : bool
        error : str | None
    """
    if not section_papers:
        return {
            "section_title": section_template.title,
            "section_text": f"## {section_template.title}\n\n（无可用的证据论文，跳过此章节。）",
            "validation": {"risk_level": "低风险", "stats": {}},
            "success": True,
            "error": None,
        }

    # Build evidence pack for this section's papers
    evidence_pack = build_evidence_pack(section_papers, max_papers=section_template.max_papers)
    evidence_text = format_evidence_pack_for_prompt(evidence_pack)

    # Fill the user prompt template
    try:
        user_prompt = section_template.user_prompt_template.format(
            section_evidence=evidence_text,
            **section_data,
        )
    except KeyError as e:
        # Missing placeholder — use a best-effort fill
        user_prompt = section_template.user_prompt_template.replace(
            "{section_evidence}", evidence_text
        )
        for key, val in section_data.items():
            user_prompt = user_prompt.replace(f"{{{key}}}", str(val))

    messages = [
        {"role": "system", "content": section_template.system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        section_text = llm_client.chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as e:
        return {
            "section_title": section_template.title,
            "section_text": f"## {section_template.title}\n\n（LLM 调用失败：{e}）",
            "validation": {"risk_level": "低风险", "stats": {}},
            "success": False,
            "error": str(e),
        }

    # Prepend section heading if LLM didn't include one
    if not section_text.strip().startswith("##"):
        section_text = f"## {section_template.title}\n\n{section_text}"

    # Validate section citations
    valid_ids = set(evidence_pack.get("paper_id_map", {}).keys())
    validation = validate_report(section_text, valid_ids) if section_template.needs_citation else {}

    return {
        "section_title": section_template.title,
        "section_text": section_text,
        "validation": validation,
        "success": True,
        "error": None,
    }


def generate_data_driven_report(
    report_plan: dict,
    llm_client: LLMClient,
    temperature: float = 0.3,
    max_tokens_per_section: int = 2048,
) -> dict:
    """Generate a full data-driven report section by section.

    Parameters
    ----------
    report_plan : dict
        Output from build_data_driven_report_plan(), containing
        sections with template, section_data, selected_papers.
    llm_client : LLMClient
        Configured LLM client.
    temperature : float
        LLM sampling temperature.
    max_tokens_per_section : int
        Max tokens per section output.

    Returns
    -------
    dict with keys:
        report_text : str
            The complete concatenated report.
        sections : list[dict]
            Per-section results: {title, text, validation, success, error}.
        validation_summary : dict
            Aggregated validation results.
        risk_level : str
            Overall risk level.
    """
    section_plans = report_plan.get("sections", [])
    if not section_plans:
        return {
            "report_text": "（未能生成报告计划，无法生成报告。）",
            "sections": [],
            "validation_summary": {},
            "risk_level": "低风险",
        }

    section_results: list[dict] = []
    report_parts: list[str] = []
    total_cited = 0
    total_uncited = 0
    total_invalid = 0
    highest_risk = "低风险"

    for sp in section_plans:
        template = sp.get("template")
        section_papers = sp.get("selected_papers", [])
        section_data = sp.get("section_data", {})

        if template is None:
            continue

        result = generate_section_with_llm(
            section_template=template,
            section_papers=section_papers,
            section_data=section_data,
            llm_client=llm_client,
            temperature=temperature,
            max_tokens=max_tokens_per_section,
        )

        section_results.append(result)
        if result.get("success"):
            report_parts.append(result["section_text"])
            report_parts.append("")

        # Aggregate validation
        v = result.get("validation", {})
        stats = v.get("stats", {})
        total_cited += stats.get("unique_papers_cited", 0)
        total_uncited += stats.get("uncited_claim_count", 0)
        total_invalid += stats.get("invalid_ref_count", 0)

        risk = v.get("risk_level", "低风险")
        if risk == "高风险":
            highest_risk = "高风险"
        elif risk == "中风险" and highest_risk == "低风险":
            highest_risk = "中风险"

    report_text = "\n\n".join(report_parts)

    return {
        "report_text": report_text,
        "sections": section_results,
        "validation_summary": {
            "total_cited_papers": total_cited,
            "total_uncited_claims": total_uncited,
            "total_invalid_refs": total_invalid,
        },
        "risk_level": highest_risk,
    }
