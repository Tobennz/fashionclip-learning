import numpy as np

from src.app.index_store import (
    load_catalog_index,
    save_catalog_index,
)


def test_save_and_load_catalog_index(tmp_path):
    embeddings = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ],
        dtype=np.float32,
    )

    products = [
        {"id": 1, "name": "Navy Shirt"},
        {"id": 2, "name": "Red Shoes"},
    ]

    save_catalog_index(
        embeddings=embeddings,
        products=products,
        output_dir=tmp_path,
    )

    loaded_embeddings, loaded_products = load_catalog_index(
        tmp_path
    )

    assert np.allclose(
        loaded_embeddings,
        embeddings,
    )

    assert loaded_products == products