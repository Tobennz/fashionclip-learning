from src.pipeline.fusion import concat_fused 
from src.pipeline.similarity import top_k_similar 

class ProductRecommender:
    def __init__(self, products, embedder, alpha=0.5):
        self.products = products
        self.embedder = embedder
        self.alpha = alpha

        self.catalog_embeddings = self._embed_products(products)

    def _embed_products(self, products):
        image_names = [
            product["image"]
            for product in products
        ]

        descriptions = [
            product["description"]
            for product in products 
        ]

        image_embeddings = self.embedder.embed_images(image_names)
        text_embeddings = self.embedder.embed_texts(descriptions)

        return concat_fused(
            image_embeddings,
            text_embeddings,
            alpha=self.alpha,
        )

    def recommend(self, image_name, description, k=3):
        image_embedding = self.embedder.embed_images([image_name])
        text_embedding = self.embedder.embed_texts([description])

        query_embedding = concat_fused(
            image_embedding, 
            text_embedding,
            alpha=self.alpha,
        )[0]

        indices, scores = top_k_similar(
            query_embedding,
            self.catalog_embeddings,
            k=k,
        )

        recommendations = []

        for index, score in zip(indices, scores):
            product = self.products[int(index)].copy()
            product["score"] = float(score)
            recommendations.append(product)

        return recommendations 