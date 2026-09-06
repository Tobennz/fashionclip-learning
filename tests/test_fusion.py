import numpy as np 
import pytest 

from src.pipeline.fusion import concat_fused, l2_normalize 

def test_l2_normalize():
    vectors = np.array([[3.0, 4.0]])
    normalized = l2_normalize(vectors)

    assert np.allclose(normalized, [[0.6, 0.8]])
    assert np.allclose(
        np.linalg.norm(normalized, axis=1),
        [1.0],
    )

def test_concat_fused():
    image_features = np.array([[3.0, 4.0]])
    text_features = np.array([[0.0, 5.0]])

    fused = concat_fused(
        image_features,
        text_features,
        alpha=0.5,
    )

    assert fused.shape == (1, 4)
    assert fused.dtype == np.float32
    assert np.allclose(
        np.linalg.norm(fused, axis=1),
        [1.0],
    )

def test_invalid_alpha():
    image_features = np.array([[1.0, 0.0]])
    text_features = np.array([[0.0, 1.0]])

    with pytest.raises(ValueError):
        concat_fused(
            image_features,
            text_features,
            alpha=1.5,
        )