from pathlib import Path 

from src.app.recommender import ProductRecommender 
from src.data.product_loader import load_products
from tests.fixtures.fake_embedder import FakeEmbedder 

def main():
    project_root = Path(__file__).parent 
    csv_path = project_root / "data" / "products.csv"

    products = load_products(csv_path)
    embedder = FakeEmbedder(dimension=4)

    recommender = ProductRecommender(
        products=products,
        embedder=embedder,
    )

    query_image = "navy-shirt.jpg"
    query_description = (
        "Navy Shirt. "
        "Men Casual Shirts in Navy Blue for Fall"
    )

    results = recommender.recommend(
        image_name=query_image,
        description=query_description,
        k=3,
    )

    print("\nRecommendations:\n")

    for position, product in enumerate(results, start=1):
        print(
            f"{position}. {product['name']} "
            f"(score: {product['score']:.3f})"
        )

if __name__=="__main__":
    main()