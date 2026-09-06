from pathlib import Path 
import pytest

from src.data.fashion_dataset import load_fashion_products 

def test_load_real_fashion_subset():
    project_root = Path(__file__).parent.parent
    csv_path = project_root / "data" / "fashion" / "styles.csv"
    images_dir = project_root / "data" / "fashion" / "images"

    if not csv_path.exists() or not images_dir.exists():
        pytest.skip("Kaggle fashion dataset is not installed.")

    products = load_fashion_products(
        csv_path=csv_path,
        images_dir=images_dir,
        limit=5,
    )

    assert len(products) == 5

    for product in products:
        assert isinstance(product["id"], int)
        assert product["name"]
        assert product["description"]
        assert Path(product["image_path"]).exists()

    product_ids = [
        product["id"]
        for product in products
    ]

    assert len(product_ids) == len(set(product_ids))