import hashlib

import numpy as np


class FakeEmbedder:
    def __init__(self, dimension=4):
        self.dimension = dimension

    def embed_texts(self, texts):
        return np.stack([
            self._make_vector(text)
            for text in texts
        ])

    def embed_images(self, image_names):
        return np.stack([
            self._make_vector(image_name)
            for image_name in image_names
        ])

    def _make_vector(self, value):
        digest = hashlib.sha256(str(value).encode()).digest()
        seed = int.from_bytes(digest[:4], byteorder="little")

        generator = np.random.default_rng(seed)

        return generator.normal(
            size=self.dimension
        ).astype(np.float32)