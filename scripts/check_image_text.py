from pathlib import Path

import numpy as np
from PIL import Image

from src.embedders.fashionclip_embedder import FashionClipEmbedder


def main():
    project_root = Path(__file__).parent.parent
    image_dir = project_root / "data" / "images"

    image_paths = [
        image_dir / "navy-shirt.jpg",
        image_dir / "red-shoes.jpg",
        image_dir / "black-handbag.jpg",
    ]

    texts = [
        "a navy blue shirt",
        "a pair of red shoes",
        "a black handbag",
    ]

    images = [
        Image.open(path).convert("RGB")
        for path in image_paths
    ]

    embedder = FashionClipEmbedder()

    image_embeddings = embedder.embed_images(images)
    text_embeddings = embedder.embed_texts(texts)

    similarities = image_embeddings @ text_embeddings.T

    np.set_printoptions(precision=3, suppress=True)

    print("\nImage embedding shape:")
    print(image_embeddings.shape)

    print("\nText embedding shape:")
    print(text_embeddings.shape)

    print("\nImage-to-text similarity matrix:")
    print(similarities)


if __name__ == "__main__":
    main()