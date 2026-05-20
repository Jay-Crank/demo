"""Query planner — generate search keywords and decompose research areas."""

import re

# Domain → (keywords, sub_directions)
_DOMAIN_KB = {
    "large language model": {
        "keywords": [
            "large language model",
            "LLM pretraining",
            "instruction tuning LLM",
            "RLHF alignment",
            "LLM reasoning",
        ],
        "sub_directions": [
            "Large Language Model Pretraining",
            "Instruction Tuning and Fine-tuning",
            "RLHF and Alignment",
            "LLM Reasoning and Chain-of-Thought",
            "Retrieval-Augmented Generation",
            "LLM Evaluation and Benchmarking",
        ],
    },
    "reinforcement learning": {
        "keywords": [
            "reinforcement learning",
            "deep Q-network",
            "policy gradient methods",
            "actor-critic reinforcement learning",
            "multi-agent reinforcement learning",
        ],
        "sub_directions": [
            "Deep Q-Networks and Value-Based RL",
            "Policy Gradient and Actor-Critic Methods",
            "Model-Based Reinforcement Learning",
            "Multi-Agent Reinforcement Learning",
            "Offline and Safe Reinforcement Learning",
        ],
    },
    "computer vision": {
        "keywords": [
            "computer vision",
            "image classification deep learning",
            "object detection transformer",
            "image segmentation",
            "visual representation learning",
        ],
        "sub_directions": [
            "Image Classification and Backbone Networks",
            "Object Detection and Instance Segmentation",
            "Image Generation and Diffusion Models",
            "Video Understanding",
            "Self-Supervised Visual Representation Learning",
            "3D Vision and Neural Radiance Fields",
        ],
    },
    "natural language processing": {
        "keywords": [
            "natural language processing",
            "text classification transformer",
            "named entity recognition",
            "machine translation neural",
            "text summarization",
        ],
        "sub_directions": [
            "Pretrained Language Models",
            "Machine Translation",
            "Text Summarization and Generation",
            "Information Extraction and NER",
            "Sentiment Analysis and Text Classification",
            "Question Answering Systems",
        ],
    },
    "graph neural network": {
        "keywords": [
            "graph neural network",
            "graph convolutional network",
            "graph attention network",
            "graph representation learning",
            "GNN node classification",
        ],
        "sub_directions": [
            "Graph Convolutional Networks",
            "Graph Attention Mechanisms",
            "Graph Representation Learning",
            "GNN for Molecular Modeling",
            "Dynamic and Temporal Graph Learning",
            "Scalable Graph Neural Networks",
        ],
    },
    "diffusion model": {
        "keywords": [
            "diffusion model",
            "denoising diffusion probabilistic model",
            "score-based generative model",
            "text-to-image diffusion",
            "diffusion model sampling acceleration",
        ],
        "sub_directions": [
            "Denoising Diffusion Probabilistic Models",
            "Score-Based Generative Models",
            "Text-to-Image Generation",
            "Video and 3D Generation with Diffusion",
            "Diffusion for Scientific Applications",
            "Fast Sampling and Distillation Methods",
        ],
    },
    "recommender system": {
        "keywords": [
            "recommender system",
            "collaborative filtering",
            "deep learning recommendation",
            "sequential recommendation",
            "graph neural network recommendation",
        ],
        "sub_directions": [
            "Collaborative Filtering Methods",
            "Deep Learning for Recommendation",
            "Sequential and Session-Based Recommendation",
            "Graph-Based Recommendation",
            "Cold-Start and Cross-Domain Recommendation",
            "Fairness and Explainability in Recommendation",
        ],
    },
    "federated learning": {
        "keywords": [
            "federated learning",
            "federated optimization",
            "differential privacy federated",
            "heterogeneous federated learning",
            "federated learning communication efficient",
        ],
        "sub_directions": [
            "Federated Optimization Algorithms",
            "Privacy and Security in Federated Learning",
            "Heterogeneous Federated Learning",
            "Communication-Efficient Federated Learning",
            "Federated Learning for Healthcare",
            "Personalized Federated Learning",
        ],
    },
    "knowledge graph": {
        "keywords": [
            "knowledge graph",
            "knowledge graph embedding",
            "knowledge graph completion",
            "knowledge graph reasoning",
            "entity alignment knowledge graph",
        ],
        "sub_directions": [
            "Knowledge Graph Embedding Methods",
            "Knowledge Graph Completion",
            "Knowledge Graph Reasoning",
            "Entity Alignment and Fusion",
            "Temporal Knowledge Graphs",
            "Knowledge Graph Construction",
        ],
    },
}

# Additional generic keyword patterns for queries not in the KB
_GENERIC_KEYWORD_PATTERNS = [
    "survey",
    "review",
    "state of the art",
    "benchmark",
    "novel method",
    "deep learning",
    "transformer",
    "attention mechanism",
    "neural network",
]


def _extract_english(text: str) -> str:
    """Extract English words from mixed Chinese-English text."""
    eng = re.findall(r"[a-zA-Z][a-zA-Z\s+\-]+", text)
    if eng:
        return " ".join(eng).strip()
    return text


def _find_domain(text: str) -> str | None:
    """Match text to known domains in the knowledge base."""
    lower = text.lower()
    for domain in _DOMAIN_KB:
        if domain in lower:
            return domain
    return None


def generate_keywords(user_query: str, n: int = 5) -> list[str]:
    """Generate 3-5 English search keywords from a user query."""
    domain = _find_domain(user_query)
    if domain:
        keywords = list(_DOMAIN_KB[domain]["keywords"])
        return keywords[:n]

    # Fallback: extract English terms + generic patterns
    eng = _extract_english(user_query)
    terms = [t.strip() for t in re.split(r"[,;，；]", eng) if t.strip()]
    keywords = terms[:3] if terms else [user_query]
    # Pad with generic patterns if needed
    for kw in keywords:
        for pat in _GENERIC_KEYWORD_PATTERNS:
            if len(keywords) >= n:
                break
            composed = f"{kw} {pat}"
            if composed not in keywords:
                keywords.append(composed)
    return keywords[:n]


def generate_sub_directions(research_area: str, n: int = 5) -> list[str]:
    """Decompose a research area into 4-6 sub-directions."""
    domain = _find_domain(research_area)
    if domain:
        subs = list(_DOMAIN_KB[domain]["sub_directions"])
        return subs[:n]

    # Fallback: generate generic sub-direction templates
    eng = _extract_english(research_area)
    templates = [
        f"{eng} Methods and Architectures",
        f"{eng} for Real-World Applications",
        f"Efficient and Scalable {eng}",
        f"{eng} with Limited Supervision",
        f"Evaluation and Benchmarks for {eng}",
        f"Theory and Foundations of {eng}",
    ]
    return templates[:n]
