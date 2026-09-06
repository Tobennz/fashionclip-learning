import numpy as np 

from src.embedders.fashionclip_embedder import FashionClipEmbedder 

def main():
    texts = [
        "navy blue casual shirt",
        "blue cotton button-up shirt",
        "red rynning shoes",
    ]

    embedder = FashionClipEmbedder()

    embeddings = embedder.embed_texts(texts)
    similarities = embeddings @ embeddings.T

    np.set_printoptions(precision=3, suppress=True)

    print("\nEmbedding shape:")
    print(embeddings.shape)

    print("\nSimilarity matrix:")
    print(similarities)

if __name__ == "__main__":
    main()