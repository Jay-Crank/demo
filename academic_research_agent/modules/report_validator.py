"""Report validator — citation audit and uncited-claim detection.

Parses LLM-generated report text to:
- Extract all [P#] citations
- Flag citations to non-existent papers (hallucination)
- Identify sentences that make factual claims without citations
- Assign an overall risk level
"""

import re


# ── Citation extraction ──────────────────────────────────────────────────

_CITATION_RE = re.compile(r"\[(P\d+)\]")

# Patterns that suggest a sentence contains a factual claim
_NUMBER_PATTERN = re.compile(
    r"\d{1,3}(?:\.\d{1,2})?\s*%|"             # percentages
    r"\b\d{2,4}\b|"                              # standalone numbers (years, counts)
    r"\d+(?:\.\d+)?\s*[×xX]\s*"                  # multipliers
)

_COMPARISON_WORDS = re.compile(
    r"\b(outperform|outperforms|outperformed|better than|worse than|"
    r"higher than|lower than|faster than|slower than|superior|inferior|"
    r"exceed|exceeds|exceeded|surpass|surpasses|surpassed|"
    r"achieve|achieves|achieved|obtain|obtains|obtained|"
    r"report|reports|reported|demonstrate|demonstrates|demonstrated|"
    r"propose|proposes|proposed|introduce|introduces|introduced|"
    r"reduce|reduces|reduced|improve|improves|improved|"
    r"state-of-the-art|SOTA|best|highest|lowest|record|"
    r"优于|高于|低于|超过|达到|提出|引入|降低|提升|改进|"
    r"最佳|最高|最低|首次|首个)\b",
    re.IGNORECASE,
)

_METRIC_TERMS = re.compile(
    r"\b(accuracy|precision|recall|F1|F1-score|AUC|ROC|BLEU|ROUGE|"
    r"METEOR|RMSE|MAE|MSE|MAPE|perplexity|throughput|latency|"
    r"准确率|精确率|召回率|性能|指标|得分|分数)\b",
    re.IGNORECASE,
)

_AUTHOR_YEAR_PATTERN = re.compile(
    r"\b[A-Z][a-z]+\s*(?:et\s*al\.?)?\s*\((?:19|20)\d{2}[a-z]?\)|"   # Smith et al. (2024)
    r"\([A-Z][a-z]+\s*(?:et\s*al\.?)?,?\s*(?:19|20)\d{2}[a-z]?\)",     # (Smith et al., 2024)
)

# Lines that should be excluded from citation checking
_EXCLUDE_PATTERNS = [
    re.compile(r"^#{1,6}\s+"),                          # markdown headings
    re.compile(r"^[-*]\s+"),                            # bullet points (structure only)
    re.compile(r"^\|"),                                  # table rows
    re.compile(r"^---+$"),                               # horizontal rules
    re.compile(r"^[>\s]*$"),                             # blockquotes / empty
    re.compile(r"^基于已有文献尚无法确定"),                  # explicit insufficiency
    re.compile(r"^\*.*\*$"),                             # italic-only lines
    re.compile(r"^\[P\d+\].*"),                          # lines starting with citation
]

# Escape-hatch phrases that are allowed without citations
_SAFE_PHRASES = [
    "基于已有文献尚无法确定",
    "证据库中未提供",
    "暂未检索到",
    "尚需进一步研究",
    "目前无法判断",
    "有待进一步验证",
]


def _split_report_sentences(text: str) -> list[tuple[str, int]]:
    """Split report into sentences, returning (sentence, line_number) pairs.

    Respects markdown structure by splitting on newlines first, then
    sentence boundaries within each line.
    """
    lines = text.split("\n")
    results: list[tuple[str, int]] = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        # For short lines (headers, bullets), keep whole
        if len(stripped) < 80 and any(
            stripped.startswith(p) for p in ("#", "-", "*", "|", ">", "[")):
            results.append((stripped, i + 1))
        else:
            # Split long lines by sentence boundaries
            sents = re.split(r"(?<=[。！？.!?])\s*", stripped)
            for s in sents:
                s = s.strip()
                if s:
                    results.append((s, i + 1))
    return results


def _should_exclude(sentence: str) -> bool:
    """Check if a sentence should be excluded from citation checking."""
    for pat in _EXCLUDE_PATTERNS:
        if pat.match(sentence):
            return True
    for phrase in _SAFE_PHRASES:
        if phrase in sentence:
            return True
    return False


def _has_factual_signals(sentence: str) -> bool:
    """Return True if the sentence contains signals of a factual claim."""
    if _NUMBER_PATTERN.search(sentence):
        return True
    if _COMPARISON_WORDS.search(sentence):
        return True
    if _METRIC_TERMS.search(sentence):
        return True
    if _AUTHOR_YEAR_PATTERN.search(sentence):
        return True
    return False


def _has_citation(sentence: str) -> bool:
    """Return True if the sentence contains at least one [P#] citation."""
    return bool(_CITATION_RE.search(sentence))


# ── Main validator ───────────────────────────────────────────────────────

def validate_report(
    report_text: str,
    valid_paper_ids: set[str],
) -> dict:
    """Validate an evidence-anchored report for citation integrity.

    Parameters
    ----------
    report_text : str
        The full LLM-generated markdown report.
    valid_paper_ids : set[str]
        Set of valid paper IDs, e.g. {"P1", "P2", ..., "P15"}.

    Returns
    -------
    dict with keys:
        cited_papers : list[str]
            Paper IDs that were actually cited in the report.
        invalid_refs : list[str]
            Paper IDs referenced in the report that do not exist in valid_paper_ids.
        uncited_claims : list[dict]
            Each entry: {"sentence": str, "line": int, "reason": str}
        risk_level : str
            "高风险" | "中风险" | "低风险"
        suggestions : list[str]
            Human-readable recommendations.
        stats : dict
            Summary counts.
    """
    cited: set[str] = set()
    invalid: list[str] = []
    uncited: list[dict] = []

    # ── 1. Extract all citations ──
    for m in _CITATION_RE.finditer(report_text):
        pid = m.group(1)
        if pid in valid_paper_ids:
            cited.add(pid)
        else:
            if pid not in invalid:
                invalid.append(pid)

    # ── 2. Detect uncited factual claims ──
    sentences = _split_report_sentences(report_text)
    for sent, line_num in sentences:
        if _should_exclude(sent):
            continue
        if _has_citation(sent):
            continue
        if _has_factual_signals(sent):
            # Determine why flagged
            reasons = []
            if _NUMBER_PATTERN.search(sent):
                reasons.append("包含数字/百分比")
            if _COMPARISON_WORDS.search(sent):
                reasons.append("包含比较/声称动词")
            if _METRIC_TERMS.search(sent):
                reasons.append("包含指标术语")
            if _AUTHOR_YEAR_PATTERN.search(sent):
                reasons.append("包含作者-年份引用")
            uncited.append({
                "sentence": sent[:200],
                "line": line_num,
                "reason": "、".join(reasons) if reasons else "疑似事实性陈述",
            })

    # ── 3. Risk assessment ──
    risk_level = "低风险"
    suggestions: list[str] = []

    if invalid:
        risk_level = "高风险"
        suggestions.append(
            f"报告引用了 {len(invalid)} 个不存在的论文编号 "
            f"({'、'.join(sorted(invalid))})，这属于严重幻觉，建议重新生成报告。"
        )

    if len(uncited) > 10:
        if risk_level != "高风险":
            risk_level = "高风险"
        suggestions.append(
            f"发现 {len(uncited)} 条无引用的疑似事实性陈述，占比过高，"
            f"建议调整提示词或提高 temperature 后重试。"
        )
    elif len(uncited) > 5:
        if risk_level == "低风险":
            risk_level = "中风险"
        suggestions.append(
            f"发现 {len(uncited)} 条无引用的疑似事实性陈述。"
            f"建议人工审查以下句子，必要时要求 LLM 补充引用。"
        )
    elif len(uncited) > 0:
        suggestions.append(
            f"发现 {len(uncited)} 条无引用的疑似事实性陈述，数量较少，"
            f"可能为过渡性表述或常识性内容，建议人工确认。"
        )

    if not invalid and len(uncited) <= 2:
        risk_level = "低风险"
        if len(cited) == 0:
            suggestions.append("报告未引用任何证据论文，请确认 LLM 是否正确遵循了引用规则。")
        else:
            suggestions.append(f"报告引用质量良好，共引用 {len(cited)} 篇论文，无明显问题。")

    # ── 4. Stats ──
    total_citation_marks = len(_CITATION_RE.findall(report_text))
    unique_valid = len(cited)
    coverage = unique_valid / max(len(valid_paper_ids), 1)

    return {
        "cited_papers": sorted(cited, key=lambda x: int(x[1:])),
        "invalid_refs": sorted(invalid, key=lambda x: int(x[1:]) if x[1:].isdigit() else 0),
        "uncited_claims": uncited,
        "risk_level": risk_level,
        "suggestions": suggestions,
        "stats": {
            "total_citation_marks": total_citation_marks,
            "unique_papers_cited": unique_valid,
            "uncited_claim_count": len(uncited),
            "invalid_ref_count": len(invalid),
            "paper_coverage": round(coverage, 3),
        },
    }


def format_validation_summary(validation: dict) -> str:
    """Render validation results as a short Markdown summary for the UI."""
    stats = validation.get("stats", {})
    risk = validation.get("risk_level", "低风险")
    emoji = {"高风险": "🔴", "中风险": "🟡", "低风险": "🟢"}.get(risk, "⚪")

    lines = [
        f"### {emoji} 引用审计结果 — 风险等级：{risk}",
        "",
        f"- **引用标记总数**：{stats.get('total_citation_marks', 0)}",
        f"- **独立引用论文数**：{stats.get('unique_papers_cited', 0)}",
        f"- **证据覆盖率**：{stats.get('paper_coverage', 0):.0%}",
        f"- **无效引用数**：{stats.get('invalid_ref_count', 0)}",
        f"- **无据陈述数**：{stats.get('uncited_claim_count', 0)}",
    ]

    suggestions = validation.get("suggestions", [])
    if suggestions:
        lines.append("")
        for s in suggestions:
            lines.append(f"- 💡 {s}")

    return "\n".join(lines)
