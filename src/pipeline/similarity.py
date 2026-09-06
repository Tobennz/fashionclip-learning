import numpy as np

from src.pipeline.fusion import l2_normalize


def cosine_scores(query, catalog):
    query = np.asarray(query, dtype=np.float32).reshape(1, -1)
    catalog = np.asarray(catalog, dtype=np.float32)

    normalized_query = l2_normalize(query)
    normalized_catalog = l2_normalize(catalog)

    return normalized_catalog @ normalized_query[0]


def top_k_similar(query, catalog, k=5):
    scores = cosine_scores(query, catalog)

    best_indices = np.argsort(scores)[::-1][:k]

    return best_indices, scores[best_indices]