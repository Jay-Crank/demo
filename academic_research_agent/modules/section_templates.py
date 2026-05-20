"""Section templates for data-driven report generation.

Defines SectionTemplate dataclass and the 10 standard section templates
used by report_planner to structure evidence-anchored survey reports.
"""

from dataclasses import dataclass, field


@dataclass
class SectionTemplate:
    """Blueprint for one section of a data-driven report.

    Each template specifies what data feeds the section, how papers are
    selected, the LLM prompts, and the expected output format.
    """

    title: str
    data_source: str
    paper_selector: str
    max_papers: int = 5
    system_prompt: str = ""
    user_prompt_template: str = ""
    needs_citation: bool = True
    output_format: str = "markdown"


# ── Shared system prompt base ────────────────────────────────────────────

_BASE_SYSTEM = """你是严谨的学术调研报告撰写助手。你只能基于提供的「事实证据库」中的论文信息进行写作。
不得引入证据库之外的任何论文、作者、实验结果或数据。
每一个事实性陈述必须在其后标注论文编号，例如 [P1]。
如果证据不足，必须写「基于已有文献尚无法确定」。
使用简体中文撰写，专业术语可保留英文原名。"""


# ── Section 1: 研究背景 ─────────────────────────────────────────────────

SEC_BACKGROUND = SectionTemplate(
    title="研究背景",
    data_source="sub_directions、论文总数统计、高引用论文元数据",
    paper_selector="citation_score top 3",
    max_papers=3,
    system_prompt=_BASE_SYSTEM,
    user_prompt_template="""## 研究领域
{user_query}

## 子方向
{sub_directions}

## 整体统计
论文总数: {total_papers}，子方向数: {n_directions}

## 高影响力论文
{section_evidence}

## 任务
请撰写「研究背景」章节。内容应包括：
1. 该领域的研究意义和当前关注度
2. 系统检索与分析的子方向概述
3. 引用高影响力论文说明领域现状
4. 本报告的结构说明（将涵盖哪些方面）

要求：每个事实性陈述必须引用 [P#]，不少于 2 个自然段。""",
    needs_citation=True,
    output_format="markdown",
)


# ── Section 2: 核心问题 ─────────────────────────────────────────────────

SEC_CORE_PROBLEMS = SectionTemplate(
    title="核心问题",
    data_source="gap_analyzer.recurring_problems",
    paper_selector="包含 challenge、limitation、future work、open problem 关键词的论文",
    max_papers=5,
    system_prompt=_BASE_SYSTEM,
    user_prompt_template="""## 研究领域
{user_query}

## 反复出现的问题与挑战
{recurring_problems}

## 相关论文证据
{section_evidence}

## 任务
请撰写「核心问题」章节。内容应包括：
1. 该领域当前面临的核心技术挑战
2. 各子方向反复出现的问题
3. 这些问题为何重要，解决它们对领域的意义

要求：每个事实性陈述必须引用 [P#]。如果证据不足，写「基于已有文献尚无法确定」。""",
    needs_citation=True,
    output_format="markdown",
)


# ── Section 3: 主要技术路线 ─────────────────────────────────────────────

SEC_TECHNICAL_ROUTES = SectionTemplate(
    title="主要技术路线",
    data_source="comparison_matrix.method_type、method_distribution",
    paper_selector="每类方法 top 2",
    max_papers=10,
    system_prompt=_BASE_SYSTEM,
    user_prompt_template="""## 研究领域
{user_query}

## 方法分布
{method_distribution}

## 各技术路线代表论文
{section_evidence}

## 任务
请撰写「主要技术路线」章节。内容应包括：
1. 当前主流技术路线的分类与概述
2. 每条路线的核心思想与代表工作
3. 各路线的优势与适用场景
4. 技术路线的演进趋势

要求：每个事实性陈述必须引用 [P#]。可适当使用小标题组织不同路线。""",
    needs_citation=True,
    output_format="markdown",
)


# ── Section 4: 研究方向图谱 ─────────────────────────────────────────────

SEC_TOPIC_MAP = SectionTemplate(
    title="研究方向图谱",
    data_source="topic_clusterer.clusters",
    paper_selector="每个主题簇 top 3",
    max_papers=15,
    system_prompt=_BASE_SYSTEM,
    user_prompt_template="""## 研究领域
{user_query}

## 主题聚类结果
{cluster_summary}

## 各簇代表论文
{section_evidence}

## 任务
请撰写「研究方向图谱」章节。内容应包括：
1. 自动识别的各研究主题簇概述
2. 每个主题簇的核心关注点和代表工作
3. 主题簇之间的关系与差异
4. 各主题的研究活跃度对比

要求：每个事实性陈述必须引用 [P#]。用自然段落描述，不要简单罗列。""",
    needs_citation=True,
    output_format="markdown",
)


# ── Section 5: 代表性工作 ───────────────────────────────────────────────

SEC_REPRESENTATIVE = SectionTemplate(
    title="代表性工作",
    data_source="composite_score top 10",
    paper_selector="综合分 top 10",
    max_papers=10,
    system_prompt=_BASE_SYSTEM,
    user_prompt_template="""## 研究领域
{user_query}

## 综合排名 Top 论文
{section_evidence}

## 任务
请撰写「代表性工作」章节。内容应包括：
1. 对综合排名最高的代表性论文进行逐一分析
2. 每篇论文的核心贡献、方法特点和实验结果
3. 这些工作的影响力和学术价值
4. 代表性工作之间的关联与差异

要求：每篇论文的分析必须引用其 [P#]。可对特别重要的论文进行更详细的分析。""",
    needs_citation=True,
    output_format="markdown",
)


# ── Section 6: 最新进展 ─────────────────────────────────────────────────

SEC_LATEST = SectionTemplate(
    title="最新进展",
    data_source="freshness_score top 10",
    paper_selector="新鲜度 top 10",
    max_papers=10,
    system_prompt=_BASE_SYSTEM,
    user_prompt_template="""## 研究领域
{user_query}

## 最新论文（按时间排序）
{section_evidence}

## 任务
请撰写「最新进展」章节。内容应包括：
1. 最近发表论文的研究趋势
2. 新提出的方法、基准或发现
3. 与之前工作的对比或改进
4. 最新进展揭示的未来可能方向

要求：每个事实性陈述必须引用 [P#]。优先关注近半年内发表的论文。""",
    needs_citation=True,
    output_format="markdown",
)


# ── Section 7: 方法对比矩阵 ─────────────────────────────────────────────

SEC_COMPARISON_MATRIX = SectionTemplate(
    title="方法对比矩阵",
    data_source="comparison_matrix",
    paper_selector="每个方法类别 top 5",
    max_papers=15,
    system_prompt=_BASE_SYSTEM + """
## 矩阵格式要求：

你必须为每个方法类别输出以下 Markdown 表格：

| 属性 | 说明 |
|------|------|
| 方法类别 | {类别名称} |
| 代表论文 | [P#] 论文标题（作者, 年份） |
| 核心思想 | 1-2 句话概括技术路线 |
| 适用场景 | 该方法最适合解决的问题类型 |
| 主要局限 | 已知的限制与不足 |
| 报告指标 | 论文中报告的关键性能指标 |
| 证据编号 | [P#] |

所有信息必须来自 evidence_pack。未知内容填「文献摘要中未明确说明」。""",
    user_prompt_template="""## 研究领域
{user_query}

## 方法分类与论文分配
{method_assignment}

## 证据库
{section_evidence}

## 任务
请为每个方法类别生成方法对比矩阵。每个类别一个表格。
表格必须包含：方法类别、代表论文、核心思想、适用场景、主要局限、报告指标或效果、证据编号。
所有内容必须来自 evidence_pack，不允许编造。未知信息填「文献摘要中未明确说明」。""",
    needs_citation=True,
    output_format="markdown_table",
)


# ── Section 8: 矛盾与争议 ───────────────────────────────────────────────

SEC_CONTROVERSIES = SectionTemplate(
    title="矛盾与争议",
    data_source="所有论文摘要和方法矩阵",
    paper_selector="可能在同一任务或指标上结论不同的论文",
    max_papers=8,
    system_prompt=_BASE_SYSTEM + """
## 重要：争议识别规则

1. 仅当多篇论文涉及同一 benchmark、同一任务或同一关键词，但报告的指标数值差异较大（如相差 >5%）时，才列为潜在争议。
2. 如果证据不足，必须写「现有摘要信息不足以判断是否存在明确矛盾」。
3. 不允许强行制造争议。如果各论文结论一致或不可直接比较，直接说明。
4. 每项争议必须同时引用双方论文 [P#][P#]。""",
    user_prompt_template="""## 研究领域
{user_query}

## 潜在矛盾信号
{contradiction_signals}

## 相关论文证据
{section_evidence}

## 任务
请撰写「矛盾与争议」章节。根据以上潜在矛盾信号，逐一分析：

1. 首先列出可能存在的矛盾点（如果检测到）
2. 对每个矛盾点，引用双方论文的证据
3. 判断该矛盾是实质性争议还是因实验设置不同导致
4. 如果无明显矛盾信号，写「现有摘要信息不足以判断是否存在明确矛盾」

禁止编造不存在的矛盾。禁止将不同任务/数据集上的结果强行对比。""",
    needs_citation=True,
    output_format="markdown",
)


# ── Section 9: 研究空白 ─────────────────────────────────────────────────

SEC_RESEARCH_GAPS = SectionTemplate(
    title="研究空白",
    data_source="research_gaps",
    paper_selector="小簇论文和 gap 信号论文",
    max_papers=8,
    system_prompt=_BASE_SYSTEM,
    user_prompt_template="""## 研究领域
{user_query}

## 检测到的研究空白
{research_gaps_summary}

## 相关论文证据
{section_evidence}

## 任务
请撰写「研究空白」章节。内容应包括：
1. 当前领域尚未被充分探索的方向
2. 论文中明确指出的未来工作和开放问题
3. 方法分布不均衡反映出的空白
4. 跨方向融合的缺失

要求：每个事实性陈述必须引用 [P#]。实事求是，不要过度推测。""",
    needs_citation=True,
    output_format="markdown",
)


# ── Section 10: 选题建议 ────────────────────────────────────────────────

SEC_TOPIC_SUGGESTIONS = SectionTemplate(
    title="选题建议",
    data_source="topic_suggestions",
    paper_selector="与选题相关的论文",
    max_papers=8,
    system_prompt=_BASE_SYSTEM,
    user_prompt_template="""## 研究领域
{user_query}

## 可行选题
{topic_suggestions}

## 相关论文证据
{section_evidence}

## 任务
请撰写「选题建议」章节。内容应包括：
1. 对未来研究方向的建议，按照可行性和创新级别分类
2. 每个建议方向的依据和支撑论文
3. 各方向的预估难度和潜在影响
4. 对研究者（尤其是研究生）的实用建议

要求：每个事实性陈述必须引用 [P#]。建议应具体、可操作。""",
    needs_citation=True,
    output_format="markdown",
)


# ── Template registry ────────────────────────────────────────────────────

ALL_SECTION_TEMPLATES: list[SectionTemplate] = [
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
]


def get_template_by_title(title: str) -> SectionTemplate | None:
    """Look up a section template by its title."""
    for t in ALL_SECTION_TEMPLATES:
        if t.title == title:
            return t
    return None
