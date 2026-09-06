import numpy as np 

def l2_normalize(vectors):
    vectors = np.asarray(vectors, dtype=np.float32)

    lengths = np.linalg.norm(
        vectors, 
        axis=1,
        keepdims=True,
    )

    lengths = np.maximum(lengths, 1e-12)

    return vectors / lengths 

def concat_fused(image_features, text_features, alpha=0.5):
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")

    image_features = l2_normalize(image_features)
    text_features = l2_normalize(text_features)

    image_weight = np.sqrt(alpha) 
    text_weight = np.sqrt(1 - alpha)

    fused = np.concatenate(
        [
            image_weight * image_features,
            text_weight * text_features,
        ],
        axis=1,
    )

    return fused.astype(np.float32, copy=False)