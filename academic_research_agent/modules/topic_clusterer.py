"""Topic clustering — TF-IDF + KMeans to discover research themes from papers.

Groups papers into clusters, extracts per-cluster keywords, and generates
short topic names from the top keywords.  When paper count is low the
cluster count is automatically reduced.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity


def _build_text(paper: dict) -> str:
    title = paper.get("title", "") or ""
    summary = paper.get("summary", "") or ""
    return f"{title} {summary}"


def _extract_cluster_keywords(
    tfidf_matrix,
    vectorizer,
    labels: np.ndarray,
    cluster_id: int,
    top_n: int = 5,
) -> list[str]:
    """Extract top-N TF-IDF keywords for a single cluster."""
    mask = labels == cluster_id
    if not mask.any():
        return []
    cluster_vecs = tfidf_matrix[mask]
    mean_weights = np.asarray(cluster_vecs.mean(axis=0)).flatten()
    top_indices = mean_weights.argsort()[::-1][:top_n]
    feature_names = vectorizer.get_feature_names_out()
    result = [feature_names[i] for i in top_indices if mean_weights[i] > 0]
    return result if result else ["unknown"]


def _pick_representative(
    tfidf_matrix,
    labels: np.ndarray,
    papers: list[dict],
    cluster_id: int,
) -> dict:
    """Pick the paper closest to the cluster centroid as representative."""
    mask = labels == cluster_id
    if not mask.any():
        return papers[0] if papers else {}
    cluster_vecs = tfidf_matrix[mask]
    centroid = np.asarray(cluster_vecs.mean(axis=0))
    dense_vecs = np.asarray(cluster_vecs.todense())
    sims = cosine_similarity(dense_vecs, centroid.reshape(1, -1)).flatten()
    best_local_idx = int(sims.argmax())
    global_indices = np.where(mask)[0]
    return papers[global_indices[best_local_idx]]


def _auto_k(n_papers: int, requested: int | None, min_k: int, max_k: int) -> int:
    """Determine a safe k value given the number of papers."""
    upper = max(min_k, min(max_k, n_papers - 1))
    upper = max(upper, 1)
    if requested is not None:
        return max(1, min(requested, upper))
    return upper if upper >= min_k else 1


def cluster_papers(
    papers: list[dict],
    n_clusters: int | None = None,
    min_k: int = 3,
    max_k: int = 6,
) -> dict:
    """
    Cluster papers using TF-IDF + KMeans.

    Parameters
    ----------
    papers : list[dict]
        Each paper must have 'title' and 'summary'.
    n_clusters : int | None
        Fixed cluster count. If None, auto-select best k (3–6) via silhouette.
    min_k, max_k : int
        Range for auto-selection.

    Returns
    -------
    dict with: clusters, n_clusters, total_papers, silhouette_score.
    """
    n = len(papers)

    # ── Fewer than 6 papers → single cluster ──
    if n < 6:
        return {
            "clusters": [{
                "id": 0,
                "topic_name": "All Papers",
                "keywords": ["insufficient", "data"],
                "paper_count": n,
                "papers": papers,
                "representative": papers[0] if papers else None,
            }] if papers else [],
            "n_clusters": 1 if papers else 0,
            "total_papers": n,
            "silhouette_score": 0.0,
        }

    texts = [_build_text(p) for p in papers]

    # All-empty texts guard
    if all(not t.strip() for t in texts):
        return {
            "clusters": [{
                "id": 0,
                "topic_name": "All Papers",
                "keywords": ["no", "text"],
                "paper_count": n,
                "papers": papers,
                "representative": papers[0],
            }],
            "n_clusters": 1,
            "total_papers": n,
            "silhouette_score": 0.0,
        }

    vectorizer = TfidfVectorizer(stop_words="english", max_features=3000)
    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
    except ValueError:
        return _fallback_single(papers)

    # ── Auto-select k ──
    safe_max = _auto_k(n, n_clusters, min_k, max_k)
    if safe_max <= 1:
        return _fallback_single(papers)

    if n_clusters is None:
        best_k = min(min_k, safe_max)
        best_sil = -1.0
        for k in range(max(2, min_k), safe_max + 1):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(tfidf_matrix)
            if len(set(labels)) < 2:
                continue
            sil = silhouette_score(tfidf_matrix, labels)
            if sil > best_sil:
                best_sil = sil
                best_k = k
        final_k = best_k
    else:
        final_k = safe_max

    # ── Final clustering ──
    km = KMeans(n_clusters=final_k, random_state=42, n_init=10)
    labels = km.fit_predict(tfidf_matrix)
    sil = silhouette_score(tfidf_matrix, labels) if len(set(labels)) >= 2 else 0.0

    # ── Build per-cluster results ──
    clusters = []
    for cid in range(final_k):
        mask = labels == cid
        cluster_papers_list = [p for i, p in enumerate(papers) if mask[i]]
        keywords = _extract_cluster_keywords(tfidf_matrix, vectorizer, labels, cid, top_n=5)
        topic_name = " ".join(keywords[:3]).title() if keywords else f"Cluster {cid + 1}"
        representative = _pick_representative(tfidf_matrix, labels, papers, cid)

        clusters.append({
            "id": cid,
            "topic_name": topic_name,
            "keywords": keywords,
            "paper_count": len(cluster_papers_list),
            "papers": cluster_papers_list,
            "representative": representative,
        })

    # Sort by paper count descending
    clusters.sort(key=lambda c: c["paper_count"], reverse=True)
    for i, c in enumerate(clusters):
        c["id"] = i

    return {
        "clusters": clusters,
        "n_clusters": final_k,
        "total_papers": n,
        "silhouette_score": round(float(sil), 4),
    }


def _fallback_single(papers: list[dict]) -> dict:
    n = len(papers)
    return {
        "clusters": [{
            "id": 0,
            "topic_name": "All Papers",
            "keywords": [],
            "paper_count": n,
            "papers": papers,
            "representative": papers[0] if papers else None,
        }] if papers else [],
        "n_clusters": 1 if papers else 0,
        "total_papers": n,
        "silhouette_score": 0.0,
    }
