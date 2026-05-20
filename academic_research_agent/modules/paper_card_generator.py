"""Paper card generator — structured reading cards from paper metadata.

Template-based: extracts sections from title + abstract via keyword matching.
LLM-ready: generate_paper_card_with_llm() reserved for future integration.
"""

import re
from typing import Callable

# ── Knowledge bases for section extraction ───────────────────────────

_DATASETS = [
    "ImageNet", "CIFAR-10", "CIFAR-100", "COCO", "MNIST", "SVHN",
    "GLUE", "SuperGLUE", "SQuAD", "CNN/DailyMail", "WMT", "Multi30k",
    "WikiText", "Penn Treebank", "PubMed", "MIMIC", "CheXpert",
    "OpenAI Gym", "Atari", "MuJoCo", "MoleculeNet", "QM9",
    "LibriSpeech", "VoxCeleb", "Cityscapes", "KITTI", "Waymo",
    "MS MARCO", "Natural Questions", "TriviaQA", "HotpotQA",
    "HumanEval", "MBPP", "GSM8K", "MATH", "MMLU", "ARC",
    "HellaSwag", "BoolQ", "PIQA", "WinoGrande",
]

_METRICS = [
    "accuracy", "F1", "F1-score", "precision", "recall", "AUC", "ROC",
    "BLEU", "ROUGE", "METEOR", "perplexity", "BPB", "word error rate",
    "mean average precision", "mAP", "IoU", "Dice", "PSNR", "SSIM",
    "success rate", "win rate", "Elo", "pass@k", "exact match",
]

_METHOD_KEYWORDS = {
    "Transformer / Attention": ["transformer", "attention", "self-attention", "multi-head", "encoder-decoder"],
    "CNN / Convolution": ["convolution", "CNN", "ResNet", "feature map", "pooling"],
    "RNN / LSTM / GRU": ["RNN", "LSTM", "GRU", "recurrent", "sequence model"],
    "Graph Neural Network": ["graph neural", "GNN", "GCN", "GAT", "message passing", "graph convolution"],
    "Reinforcement Learning": ["reinforcement", "policy gradient", "Q-learning", "actor-critic", "reward", "DQN"],
    "Diffusion Model": ["diffusion", "denoising", "score-based", "DDPM", "score matching"],
    "Contrastive Learning": ["contrastive", "SimCLR", "MoCo", "positive pair", "negative sample"],
    "Self-Supervised Learning": ["self-supervised", "pretext task", "masked", "pre-training", "pre-train"],
    "Fine-tuning / PEFT": ["fine-tun", "LoRA", "adapter", "prompt tuning", "prefix tuning"],
    "Retrieval-Augmented": ["retrieval-augment", "RAG", "retriever", "dense retrieval"],
    "Chain-of-Thought": ["chain-of-thought", "reasoning", "step-by-step", "CoT"],
    "Knowledge Distillation": ["distill", "teacher", "student model", "knowledge transfer"],
}

# ── Sentence-level extraction helper ─────────────────────────────────

def _extract_sentences(text: str, keywords: list[str], max_sentences: int = 3) -> str:
    """Return sentences from `text` that contain any of `keywords`."""
    if not text:
        return "无法从摘要中提取。"
    # Split on sentence boundaries
    sents = re.split(r"(?<=[.!?])\s+", text)
    matched = [s.strip() for s in sents if any(kw.lower() in s.lower() for kw in keywords)]
    if not matched:
        return "摘要中未明确提及。"
    return " ".join(matched[:max_sentences])


def _match_keywords(text: str, keyword_groups: dict[str, list[str]]) -> list[str]:
    """Return the group names whose keywords appear in text."""
    found = []
    lower = text.lower()
    for group, kws in keyword_groups.items():
        if any(kw.lower() in lower for kw in kws):
            found.append(group)
    return found


def _find_terms(text: str, terms: list[str]) -> list[str]:
    """Return terms that appear in text."""
    lower = text.lower()
    return [t for t in terms if t.lower() in lower]


# ── Main card generator ──────────────────────────────────────────────

def generate_paper_card(paper: dict) -> dict:
    """
    Generate a structured reading card for a single paper.

    Parameters
    ----------
    paper : dict
        Must have: title, summary, authors, published, link (or url).

    Returns
    -------
    dict with keys:
      title, authors, published, link,
      research_problem, core_method, key_innovation,
      tasks_datasets, evaluation_metrics, main_conclusions,
      possible_limitations, reading_reason,
      raw_summary (truncated original abstract).
    """
    title = paper.get("title", "Untitled") or "Untitled"
    summary = paper.get("summary", "") or ""
    authors = paper.get("authors", "Unknown") or "Unknown"
    published = paper.get("published")
    link = paper.get("link", "") or paper.get("url", "") or "#"

    combined = f"{title} {summary}"

    # ── 1. Research problem ──
    problem_kw = [
        "problem", "challenge", "address", "tackle", "solve", "goal",
        "aim", "objective", "focus on", "study", "investigate", "explore",
        "question", "issue", "limitation of", "lack of",
    ]
    research_problem = _extract_sentences(summary, problem_kw, max_sentences=3)

    # ── 2. Core method ──
    matched_methods = _match_keywords(combined, _METHOD_KEYWORDS)
    method_kw = [
        "method", "approach", "framework", "model", "architecture",
        "algorithm", "technique", "pipeline", "scheme", "strategy",
    ]
    method_sents = _extract_sentences(summary, method_kw, max_sentences=2)
    core_method = method_sents
    if matched_methods:
        core_method = f"**技术路线**: {', '.join(matched_methods[:5])}\n\n{method_sents}"

    # ── 3. Key innovation ──
    innovation_kw = [
        "novel", "new", "first", "propose", "introduce", "improve",
        "outperform", "state-of-the-art", "advance", "contribution",
        "key insight", "different from", "unlike", "better than",
    ]
    key_innovation = _extract_sentences(summary, innovation_kw, max_sentences=3)

    # ── 4. Tasks / Datasets ──
    task_kw = [
        "task", "dataset", "benchmark", "corpus", "evaluated on",
        "experiment on", "test on", "applied to", "domain",
    ]
    task_sents = _extract_sentences(summary, task_kw, max_sentences=2)
    found_datasets = _find_terms(combined, _DATASETS)
    tasks_datasets = task_sents
    if found_datasets:
        tasks_datasets += f"\n\n**识别到的数据集**: {', '.join(found_datasets[:8])}"

    # ── 5. Evaluation metrics ──
    found_metrics = _find_terms(combined, _METRICS)
    if found_metrics:
        evaluation_metrics = f"论文中提及以下评价指标: {', '.join(sorted(set(found_metrics))[:10])}。"
    else:
        evaluation_metrics = "摘要中未明确提及具体评价指标。"

    # ── 6. Main conclusions ──
    conclusion_kw = [
        "result", "show", "demonstrate", "achieve", "obtain",
        "outperform", "exceed", "surpass", "improve", "reduce",
        "increase", "better", "higher", "lower", "effective",
    ]
    main_conclusions = _extract_sentences(summary, conclusion_kw, max_sentences=4)

    # ── 7. Possible limitations ──
    limitation_kw = [
        "limitation", "future work", "however", "although",
        "remain", "challenge", "drawback", "weakness", "fail",
        "not yet", "further investigation", "need to be",
    ]
    possible_limitations = _extract_sentences(summary, limitation_kw, max_sentences=3)

    # ── 8. Reading recommendation ──
    reasons = []
    if any(kw in combined.lower() for kw in ["state-of-the-art", "SOTA", "outperform"]):
        reasons.append("该工作在实验中取得了领先性能")
    if any(kw in combined.lower() for kw in ["novel", "new", "first"]):
        reasons.append("提出了创新性的方法或框架")
    if any(kw in combined.lower() for kw in ["survey", "review", "overview"]):
        reasons.append("提供了该领域的全面综述")
    if any(kw in combined.lower() for kw in ["benchmark", "dataset", "open-source", "code"]):
        reasons.append("可能提供了公开的数据集或代码")
    if matched_methods:
        reasons.append(f"涉及 {', '.join(matched_methods[:3])} 等热点技术")
    if not reasons:
        reasons.append("与查询主题相关，可作为背景文献阅读")
    reading_reason = "；".join(reasons) + "。"

    # ── Publish date ──
    pub_str = "N/A"
    if published:
        try:
            from datetime import datetime as _dt
            if isinstance(published, _dt):
                pub_str = published.strftime("%Y-%m-%d")
            else:
                pub_str = str(published)[:10]
        except Exception:
            pub_str = "N/A"

    return {
        "title": title,
        "authors": authors,
        "published": pub_str,
        "link": link,
        "research_problem": research_problem,
        "core_method": core_method,
        "key_innovation": key_innovation,
        "tasks_datasets": tasks_datasets,
        "evaluation_metrics": evaluation_metrics,
        "main_conclusions": main_conclusions,
        "possible_limitations": possible_limitations,
        "reading_reason": reading_reason,
        "raw_summary": summary[:500] + ("..." if len(summary) > 500 else ""),
    }


# ── LLM-ready interface (reserved) ───────────────────────────────────

def generate_paper_card_with_llm(
    paper: dict,
    llm_call: Callable[[str], str],
) -> dict:
    """
    Generate a paper card using an LLM.

    Parameters
    ----------
    paper : dict
        Paper metadata.
    llm_call : callable
        A function that takes a prompt string and returns the LLM response.
        Example: lambda prompt: anthropic_client.messages.create(...)

    Returns
    -------
    dict — same structure as generate_paper_card.
    """
    title = paper.get("title", "")
    summary = paper.get("summary", "")
    prompt = f"""You are an academic paper reviewer. Analyze the following paper and produce a structured reading card in Chinese.

Paper Title: {title}
Abstract: {summary}

Please output exactly these 8 sections in Chinese, each 2-4 sentences:

1. 研究问题：
2. 核心方法：
3. 关键创新：
4. 实验任务或数据集：
5. 评价指标：
6. 主要结论：
7. 可能局限：
8. 推荐阅读理由："""

    response = llm_call(prompt)

    # Parse structured response into dict
    card = {
        "title": title,
        "authors": paper.get("authors", ""),
        "published": str(paper.get("published", ""))[:10],
        "link": paper.get("link", "") or paper.get("url", ""),
        "research_problem": "", "core_method": "", "key_innovation": "",
        "tasks_datasets": "", "evaluation_metrics": "", "main_conclusions": "",
        "possible_limitations": "", "reading_reason": "",
        "raw_summary": summary,
    }

    # Map numbered sections to dict keys
    section_map = {
        "研究问题": "research_problem",
        "核心方法": "core_method",
        "关键创新": "key_innovation",
        "实验任务或数据集": "tasks_datasets",
        "评价指标": "evaluation_metrics",
        "主要结论": "main_conclusions",
        "可能局限": "possible_limitations",
        "推荐阅读理由": "reading_reason",
    }

    current_key = None
    for line in response.split("\n"):
        line = line.strip()
        if not line:
            continue
        for label, key in section_map.items():
            if label in line:
                current_key = key
                # Extract content after the label
                content = line.split("：", 1)[-1].split(":", 1)[-1].strip()
                if content:
                    card[current_key] = content
                break
        else:
            if current_key:
                card[current_key] += line

    return card


# ── Markdown formatter ───────────────────────────────────────────────

def format_card_markdown(card: dict) -> str:
    """Render a paper card as formatted Markdown string."""
    lines = [
        f"# 📄 论文精读卡片",
        "",
        f"**{card['title']}**",
        "",
        f"*{card['authors']}* | {card['published']} | [arXiv]({card['link']})",
        "",
        "---",
        "",
        "## 🔍 研究问题",
        "",
        card["research_problem"],
        "",
        "## ⚙️ 核心方法",
        "",
        card["core_method"],
        "",
        "## 💡 关键创新",
        "",
        card["key_innovation"],
        "",
        "## 📊 实验任务 / 数据集",
        "",
        card["tasks_datasets"],
        "",
        "## 📏 评价指标",
        "",
        card["evaluation_metrics"],
        "",
        "## ✅ 主要结论",
        "",
        card["main_conclusions"],
        "",
        "## ⚠️ 可能局限",
        "",
        card["possible_limitations"],
        "",
        "## 📌 推荐阅读理由",
        "",
        card["reading_reason"],
        "",
        "---",
        "",
        "### 原始摘要",
        "",
        card["raw_summary"],
    ]
    return "\n".join(lines)
