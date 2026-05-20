# 学术调研 Agent — Academic Research Agent

基于多轮检索与个性化排序的学术调研助手（课程项目 MVP）。

## 功能

| 功能 | 说明 |
|------|------|
| 🔍 深度搜索 | 输入学术问题 → 扩展关键词 → 检索 arXiv → 去重排序 → 综合回答 |
| 📊 深度调研 | 输入研究领域 → 拆分子方向 → 分别检索 → 生成结构化调研报告 |
| 📬 个性化推送 | 输入科研方向 → 检索最新论文 → 全局排序 → 日报 + Top 3 解读 |

## 技术栈

- **前端**: Streamlit
- **数据源**: arXiv API（通过 `arxiv` Python 库）
- **文本处理**: scikit-learn（TF-IDF）
- **数据处理**: pandas, numpy
- **缓存**: CSV

## 排序算法

综合得分公式：

```
Score(p, u) = 0.55 × R(p,u) + 0.30 × T(p) + 0.15 × Q(p)
```

- **R(p,u)**: 论文与用户问题的 TF-IDF 余弦相似度
- **T(p)**: 论文新鲜度，T(p) = exp(-Δt / 30)
- **Q(p)**: 论文质量分（摘要长度 + 链接有效性 + 信息完整性）

## 安装与运行

```bash
# 1. 进入项目目录
cd academic_research_agent

# 2. 安装依赖
pip install -r requirements.txt

# 3. 运行
streamlit run app.py
```

浏览器打开 http://localhost:8501 即可使用。

## 项目结构

```
academic_research_agent/
├── app.py                    # Streamlit 主界面（3 个页面）
├── requirements.txt          # Python 依赖
├── modules/
│   ├── arxiv_client.py       # arXiv API 检索
│   ├── query_planner.py      # 查询扩展与子方向生成
│   ├── deduplicator.py       # 论文去重
│   ├── reranker.py           # TF-IDF 相关度 + 新鲜度 + 质量分
│   ├── report_generator.py   # 搜索回答 + 调研报告生成
│   ├── feed_generator.py     # 日报 + 深度解读生成
│   └── utils.py              # 工具函数
├── data/
│   └── cache.csv             # 本地缓存
└── README.md
```

## 注意事项

- 当前版本为 MVP，摘要和报告使用模板化生成，未接入大语言模型
- 代码中预留了 LLM 接口扩展点，后续可接入 Claude API 等
- 数据源仅限 arXiv，不读取论文全文
- arXiv API 有频率限制，连续大量请求可能被限速
