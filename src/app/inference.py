import numpy as np

from src.app.index_store import load_catalog_index
from src.pipeline.fusion import concat_fused
from src.pipeline.similarity import top_k_similar


class InferenceService:
    def __init__(
        self,
        index_dir,
        embedder,
        alpha=0.5,
    ):
        self.embedder = embedder
        self.alpha = alpha

        self.catalog_embeddings, self.products = (
            load_catalog_index(index_dir)
        )

        self.modality_dimension = (
            self.catalog_embeddings.shape[1] // 2
        )

    def recommend(
        self,
        image=None,
        text=None,
        k=5,
    ):
        if image is None and not text:
            raise ValueError("Provide an image or text")

        image_embedding = self._image_embedding(image)
        text_embedding = self._text_embedding(text)

        query_embedding = concat_fused(
            image_embedding,
            text_embedding,
            alpha=self.alpha,
        )[0]

        indices, scores = top_k_similar(
            query=query_embedding,
            catalog=self.catalog_embeddings,
            k=min(k, len(self.products)),
        )

        results = []

        for index, score in zip(indices, scores):
            product = self.products[int(index)].copy()
            product["score"] = float(score)
            results.append(product)

        return results

    def _image_embedding(self, image):
        if image is None:
            return self._zero_embedding()

        return self.embedder.embed_images([image])

    def _text_embedding(self, text):
        if not text:
            return self._zero_embedding()

        return self.embedder.embed_texts([text])

    def _zero_embedding(self):
        return np.zeros(
            (1, self.modality_dimension),
            dtype=np.float32,
        )