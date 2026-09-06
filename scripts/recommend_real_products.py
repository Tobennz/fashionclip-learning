from pathlib import Path 

from PIL import Image 

from src.data.fashion_dataset import load_fashion_products 
from src.embedders.fashionclip_embedder import FashionClipEmbedder
from src.pipeline.fusion import concat_fused 
from src.pipeline.similarity import top_k_similar 

def main():
    project_root = Path(__file__).parent.parent 

    products = load_fashion_products(
        csv_path=project_root / "data" / "fashion" / "styles.csv",
        images_dir=project_root / "data" / "fashion" / "images",
        limit=20,
    )

    images = [
        Image.open(product["image_path"]).convert("RGB")
        for product in products 
    ]

    descriptions = [
        product["description"]
        for product in products 
    ]

    print(f"Loaded {len(products)} real products")

    embedder = FashionClipEmbedder()

    image_embeddings = embedder.embed_images(images)
    text_embeddings = embedder.embed_texts(descriptions)

    catalog_embeddings = concat_fused(
        image_embeddings,
        text_embeddings,
        alpha=0.5,
    )

    query_index = 0 
    query_embedding = catalog_embeddings[query_index]

    indices, scores = top_k_similar(
        query=query_embedding,
        catalog=catalog_embeddings,
        k=6,
    )

    query_product = products[query_index]

    print("\nQuery product:")
    print(query_product["name"])
    print(query_product["description"])
    print(query_product["image_path"])

    print("\nSimilar products:")

    rank = 1

    for index, score in zip(indices, scores):
        index = int(index)

        if index == query_index:
            continue 

        product = products[index]

        print(
            f"{rank}. {product['name']} "
            f"(score: {float(score):.3f})"
        )

        rank += 1
if __name__ == "__main__":
    main()