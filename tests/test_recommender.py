from src.app.recommender import ProductRecommender 
from tests.fixtures.fake_embedder import FakeEmbedder 

def test_recommender_returns_matching_product_first():
    products = [
        {
            "id": 1,
            "name": "Navy Shirt",
            "image": "navy-shirt.jpg",
            "description": "Men Casual Shirts in Navy Blue for Fall",
        },
        {
            "id": 2,
            "name": "Red Shoes",
            "image": "red-shoes.jpg",
            "description": "Men Sports Shoes in Red for Summer",
        },
        {
            "id": 3,
            "name": "Black Handbag",
            "image": "black-handbag.jpg",
            "description": "Women Casual Handbags in Black for Fall",
        },
    ]

    embedder = FakeEmbedder(dimension=4)

    recommender = ProductRecommender(
        products=products,
        embedder=embedder,
    )

    results = recommender.recommend(
        image_name="navy-shirt.jpg",
        description="Men Casual Shirts in Navy Blue for Fall",
        k=2,
    )

    assert len(results) == 2
    assert results[0]["id"] == 1
    assert results[0]["name"] == "Navy Shirt"
    assert results[0]["score"] > results[1]["score"]

from pathlib import Path

from src.data.product_loader import load_products

def test_recommender_with_csv_catalog():
    project_root = Path(__file__).parent.parent 
    csv_path = project_root /"data" / "products.csv"

    products = load_products(csv_path)
    embedder = FakeEmbedder(dimension=4)

    recommender = ProductRecommender(
        products=products,
        embedder=embedder,
    )

    results = recommender.recommend(
        image_name="navy-shirt.jpg",
        description="Navy Shirt. Men Casual Shirts in Navy Blue for Fall",
        k=2,
    )

    assert len(results) == 2
    assert results[0]["id"] == 1
    assert results[0]["name"] == "Navy Shirt"