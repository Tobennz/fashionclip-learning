import numpy as np 

from src.pipeline.similarity import cosine_scores, top_k_similar

def test_cosine_scores():
    query = np.array([1.0, 0.0])

    catalog = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [-1.0, 0.0],
    ])

    scores = cosine_scores(query, catalog)

    assert np.allclose(scores, [1.0, 0.0, -1.0])

def test_top_k_similar():
    query = np.array([1.0, 0.0])

    catalog = np.array([
        [0.0, 1.0],
        [1.0, 0.0],
        [0.8, 0.2],
    ])

    indices, scores = top_k_similar(query, catalog, k=2)

    assert indices.tolist() == [1, 2]
    assert scores[0] > scores[1]