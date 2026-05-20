"""Citation tracing and evidence verification module.

For each section of a generated report, matches it against the
retrieved paper corpus to find supporting evidence and assess
credibility using heuristic rules.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _build_paper_text(paper: dict) -> str:
    return f"{paper['title']} {paper['summary']}"


def assess_evidence(
    user_query: str,
    top_papers: list[dict],
    generated_sections: dict[str, str],
    relevance_threshold: float = 0.1,
) -> dict:
    """
    Assess credibility of each generated report section against the paper corpus.

    Parameters
    ----------
    user_query : str
        The original user query.
    top_papers : list[dict]
        Ranked paper list, each with at minimum: title, summary, composite_score.
    generated_sections : dict[str, str]
        Mapping of section name → section text content.

    Returns
    -------
    dict with keys:
        sections : list[dict]
            Per-section evidence (name, credibility, papers, counts, avg_relevance).
        overall : dict
            Overall credibility summary.
    """
    if not top_papers or not generated_sections:
        return {
            "sections": [],
            "overall": {
                "score": 0.0,
                "level": "低",
                "total_papers": len(top_papers),
                "high_count": 0,
                "mid_count": 0,
                "low_count": len(generated_sections),
            },
        }

    paper_texts = [_build_paper_text(p) for p in top_papers]
    section_names = list(generated_sections.keys())
    section_texts = [generated_sections[name] for name in section_names]

    # Fit TF-IDF on paper corpus + all sections
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    try:
        all_texts = paper_texts + section_texts
        tfidf_matrix = vectorizer.fit_transform(all_texts)
    except ValueError:
        return _empty_result(generated_sections, top_papers)

    paper_vecs = tfidf_matrix[: len(paper_texts)]
    section_vecs = tfidf_matrix[len(paper_texts):]

    # Cosine similarity between each section and each paper
    sim_matrix = cosine_similarity(section_vecs, paper_vecs)  # (n_sections, n_papers)

    section_results = []
    high_count = mid_count = low_count = 0

    for i, name in enumerate(section_names):
        sims = sim_matrix[i]
        # Papers exceeding relevance threshold
        relevant_mask = sims >= relevance_threshold
        relevant_indices = np.where(relevant_mask)[0]
        relevant_papers = [top_papers[j] for j in relevant_indices]
        relevant_sims = sims[relevant_indices]

        paper_count = len(relevant_papers)
        avg_relevance = float(np.mean(relevant_sims)) if len(relevant_sims) > 0 else 0.0

        # Heuristic credibility rules
        if paper_count >= 3 and avg_relevance >= 0.5:
            credibility = "高"
            high_count += 1
        elif paper_count >= 1:
            credibility = "中"
            mid_count += 1
        else:
            credibility = "低"
            low_count += 1

        # Build compact evidence paper list
        evidence_papers = []
        for j in relevant_indices:
            p = top_papers[j]
            evidence_papers.append({
                "title": p["title"],
                "authors": p.get("authors", "").split(",")[0].strip(),
                "arxiv_id": p.get("arxiv_id", ""),
                "link": p.get("link", ""),
                "relevance_score": round(float(sims[j]), 4),
            })
        # Sort evidence papers by relevance descending
        evidence_papers.sort(key=lambda x: x["relevance_score"], reverse=True)

        section_results.append({
            "name": name,
            "credibility": credibility,
            "paper_count": paper_count,
            "avg_relevance": round(avg_relevance, 4),
            "evidence_papers": evidence_papers,
        })

    # Overall credibility
    total_sections = len(section_results)
    if total_sections == 0:
        overall_level = "低"
        overall_score = 0.0
    else:
        # Weighted: 高=1.0, 中=0.5, 低=0.0
        overall_score = (high_count * 1.0 + mid_count * 0.5) / total_sections
        if overall_score >= 0.66:
            overall_level = "高"
        elif overall_score >= 0.33:
            overall_level = "中"
        else:
            overall_level = "低"

    return {
        "sections": section_results,
        "overall": {
            "score": round(overall_score, 2),
            "level": overall_level,
            "total_papers": len(top_papers),
            "high_count": high_count,
            "mid_count": mid_count,
            "low_count": low_count,
        },
    }


def _empty_result(sections: dict, papers: list) -> dict:
    """Return empty result when TF-IDF fails (e.g. empty sections)."""
    sec_results = [
        {
            "name": name,
            "credibility": "低",
            "paper_count": 0,
            "avg_relevance": 0.0,
            "evidence_papers": [],
        }
        for name in sections
    ]
    return {
        "sections": sec_results,
        "overall": {
            "score": 0.0,
            "level": "低",
            "total_papers": len(papers),
            "high_count": 0,
            "mid_count": 0,
            "low_count": len(sections),
        },
    }


def format_credibility_badge(level: str) -> str:
    """Return a markdown badge string for a credibility level."""
    colors = {
        "高": "green",
        "中": "orange",
        "低": "red",
    }
    c = colors.get(level, "grey")
    return f"![可信度: {level}](https://img.shields.io/badge/可信度-{level}-{c})"
