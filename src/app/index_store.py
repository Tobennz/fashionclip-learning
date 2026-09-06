import json
from pathlib import Path

import numpy as np


def save_catalog_index(
    embeddings,
    products,
    output_dir,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    embeddings_path = output_dir / "catalog_embeddings.npy"
    products_path = output_dir / "catalog_products.json"

    np.save(embeddings_path, embeddings)

    with products_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            products,
            file,
            indent=2,
            ensure_ascii=False,
        )

    return embeddings_path, products_path


def load_catalog_index(output_dir):
    output_dir = Path(output_dir)

    embeddings_path = output_dir / "catalog_embeddings.npy"
    products_path = output_dir / "catalog_products.json"

    embeddings = np.load(
        embeddings_path,
        mmap_mode="r",
    )

    with products_path.open(
        encoding="utf-8",
    ) as file:
        products = json.load(file)

    return embeddings, products