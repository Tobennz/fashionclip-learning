from pathlib import Path

from src.data.product_loader import load_products


def test_load_products():
    project_root = Path(__file__).parent.parent
    csv_path = project_root / "data" / "products.csv"

    products = load_products(csv_path)

    assert len(products) == 3
    assert products[0]["id"] == 1
    assert products[0]["name"] == "Navy Shirt"
    assert products[0]["image"] == "navy-shirt.jpg"

    assert products[0]["description"] == (
        "Navy Shirt. "
        "Men Casual Shirts in Navy Blue for Fall"
    )