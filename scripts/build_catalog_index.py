from pathlib import Path

from PIL import Image

from src.app.index_store import save_catalog_index
from src.data.fashion_dataset import load_fashion_products
from src.embedders.fashionclip_embedder import FashionClipEmbedder
from src.pipeline.fusion import concat_fused


def main():
    project_root = Path(__file__).parent.parent

    products = load_fashion_products(
        csv_path=project_root / "data" / "fashion" / "styles.csv",
        images_dir=project_root / "data" / "fashion" / "images",
        limit=100,
    )

    print(f"Loaded {len(products)} products")

    images = [
        Image.open(product["image_path"]).convert("RGB")
        for product in products
    ]

    descriptions = [
        product["description"]
        for product in products
    ]

    embedder = FashionClipEmbedder()

    print("Embedding images...")
    image_embeddings = embedder.embed_images(images)

    print("Embedding descriptions...")
    text_embeddings = embedder.embed_texts(descriptions)

    print("Fusing embeddings...")
    catalog_embeddings = concat_fused(
        image_embeddings,
        text_embeddings,
        alpha=0.5,
    )

    output_dir = project_root / "outputs" / "catalog"

    embeddings_path, products_path = save_catalog_index(
        embeddings=catalog_embeddings,
        products=products,
        output_dir=output_dir,
    )

    print("\nCatalog index saved")
    print(f"Shape: {catalog_embeddings.shape}")
    print(f"Embeddings: {embeddings_path}")
    print(f"Products: {products_path}")


if __name__ == "__main__":
    main()