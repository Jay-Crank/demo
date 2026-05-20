# 学术调研 Agent — Academic Research Agent

基于多轮检索、个性化排序与证据锚定的学术调研助手（课程项目 MVP v2.0）。

## 核心亮点

- **多源检索** — 同时检索 arXiv、OpenAlex、Semantic Scholar，自动去重合并
- **证据锚定** — 报告中每个事实性陈述必须标注论文编号 [P1]，可审计验证
- **数据驱动章节** — 10 个标准章节由系统分析数据后自动规划，非 LLM 自由发挥
- **全文精读** — 下载 arXiv PDF 全文，提取方法/实验/局限性等章节，生成精读卡片
- **引用审计** — 自动检测无效引用（幻觉）和无据陈述，红/黄/绿风险分级

## 功能页面

| 页面 | 说明 |
|------|------|
| 深度搜索 | 输入学术问题 → 扩展关键词 → 多源检索 → 去重排序 → 综合回答 + 可信度检查 |
| 深度调研 | 输入研究领域 → 拆分子方向 → 分别检索 → 主题聚类 → 研究空白分析 → 三种模式生成报告 |
| 个性化推送 | 输入 1-10 个研究方向 → 检索最新论文 → 全局排序 → 日报 + Top 3 深度解读 |
| 实验评估 | 5 组对照实验验证查询扩展、排序策略、个性化权重、主题聚类、多源对比效果 |

## 调研报告生成（三种模式）

| 模式 | 说明 |
|------|------|
| 模板报告 | 基于规则模板，无需 LLM，即刻生成 |
| 单次 LLM 报告 | LLM 一次性生成完整报告，事实性陈述锚定到 [P#] |
| 数据驱动分章节报告 | 系统规划 10 个标准章节 → 为每节自动选论文 → 逐节调用 LLM → 每节独立校验 → 拼接 |

### 10 个标准章节

1. **研究背景** — 高引用经典论文
2. **核心问题** — 含 limitation/challenge 信号的论文
3. **主要技术路线** — 按方法族分类（Transformer、Diffusion、GNN 等 15 类）
4. **研究方向图谱** — 基于主题聚类结果
5. **代表性工作** — 综合评分最高论文
6. **最新进展** — 新鲜度评分最高论文
7. **方法对比矩阵** — 方法族 × 指标对比
8. **矛盾与争议** — 同基准不同结论的论文
9. **研究空白** — 缺口分析与选题建议
10. **选题建议** — 可行研究方向推荐

## 全文精读

开启「精读模式」后，系统会：

1. 下载 Top N 论文的 arXiv PDF 全文（缓存至 `.cache/pdfs/`）
2. 提取全文文本（pymupdf，支持多列排版）
3. 解析论文章节（摘要、引言、方法、实验、讨论、结论、局限性）
4. 生成精读卡片（核心贡献、方法概述、关键发现、局限性、新颖性、适合读者）
5. 生成推荐阅读路径（快速了解、方法演进、应用落地）
6. 构建全文增强证据包，供 LLM 生成更深度的分析

## 技术栈

- **前端**: Streamlit
- **数据源**: arXiv API / OpenAlex API / Semantic Scholar API
- **文本处理**: scikit-learn（TF-IDF + KMeans 聚类）
- **PDF 解析**: pymupdf（fitz）
- **LLM 集成**: OpenAI 兼容 API（支持 DeepSeek / OpenAI / Ollama 等）
- **数据处理**: pandas, numpy

## 排序算法

综合得分公式（5 维加权，系统自动归一化）：

```
Score = w_r·R + w_t·T + w_c·C + w_s·S + w_q·Q
```

| 维度 | 说明 |
|------|------|
| R — 相关度 | TF-IDF 余弦相似度 |
| T — 新鲜度 | exp(-Δdays / 30) |
| C — 引用分 | log(1 + citations) 归一化 |
| S — 来源分 | 多源收录得分更高（max 3 源） |
| Q — 质量分 | 元数据完整度（DOI/摘要/期刊/URL） |

## 安装与运行

```bash
# 1. 进入项目目录
cd academic_research_agent

# 2. 安装依赖
pip install -r requirements.txt

# 3. （可选）配置 LLM API Key 以启用 AI 增强报告
# 支持 DeepSeek / OpenAI / Ollama 等兼容 API
set LLM_API_KEY=sk-your-key-here   # Windows
# export LLM_API_KEY=sk-your-key-here   # macOS/Linux

# 4. 运行
streamlit run app.py
```

浏览器打开 http://localhost:8501 即可使用。

## 项目结构

```
academic_research_agent/
├── app.py                         # Streamlit 主界面（4 个页面）
├── requirements.txt               # Python 依赖
├── modules/
│   ├── sources/                   # 数据源模块
│   │   ├── arxiv_client.py        #   arXiv API
│   │   ├── openalex_client.py     #   OpenAlex API
│   │   └── semantic_scholar_client.py  # Semantic Scholar API
│   ├── source_aggregator.py       # 多源聚合检索
│   ├── query_planner.py           # 查询扩展与子方向生成
│   ├── deduplicator.py            # 论文去重（DOI / arXiv ID / 标题相似度）
│   ├── reranker.py                # 综合排序（5 维加权）
│   ├── topic_clusterer.py         # TF-IDF + KMeans 主题聚类
│   ├── paper_card_generator.py    # 论文精读卡片
│   ├── evidence_builder.py        # 证据包构建（P1..Pn 编号映射）
│   ├── prompt_templates.py        # LLM 提示词模板
│   ├── llm_client.py              # LLM 客户端（OpenAI 兼容）
│   ├── llm_report_generator.py    # 单次 + 数据驱动分章节 LLM 报告生成
│   ├── report_planner.py          # 数据驱动报告计划（10 章节选文）
│   ├── section_templates.py       # SectionTemplate 数据类 + 10 个标准模板
│   ├── report_validator.py        # 引用校验（无效引用 / 无据陈述检测）
│   ├── report_generator.py        # 模板报告生成
│   ├── feed_generator.py          # 日报 + 深度解读
│   ├── research_gap_analyzer.py   # 研究空白分析
│   ├── deep_reader.py             # PDF 下载 / 全文提取 / 章节解析 / 精读卡片
│   ├── verifier.py                # 引用溯源与可信度检查
│   ├── evaluator.py               # 实验评估逻辑（5 组对照实验）
│   └── utils.py                   # 工具函数
├── .cache/pdfs/                   # PDF 缓存目录
└── README.md
```

## 注意事项

- 模板报告无需 LLM，AI 增强报告需配置 `LLM_API_KEY`（支持 DeepSeek/OpenAI/Ollama）
- 精读模式需联网下载 arXiv PDF，首次加载较慢（已下载 PDF 自动缓存）
- arXiv API 有频率限制，连续大量请求可能被限速
- 多源检索时 OpenAlex / Semantic Scholar 可能因网络或 API 限额偶发失败，系统会自动降级
