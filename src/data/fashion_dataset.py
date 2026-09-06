import csv
from pathlib import Path

from src.data.text_template import build_text


TEXT_FIELDS = [
    "gender",
    "usage",
    "articleType",
    "baseColour",
    "season",
]


def load_fashion_products(
    csv_path,
    images_dir,
    limit=None,
):
    csv_path = Path(csv_path)
    images_dir = Path(images_dir)

    products = []

    with csv_path.open(
        newline="",
        encoding="utf-8",
        errors="replace",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            try:
                product_id = int(row["id"])
            except (KeyError, TypeError, ValueError):
                continue

            image_path = images_dir / f"{product_id}.jpg"

            if not image_path.exists():
                continue

            for field in TEXT_FIELDS:
                if not row.get(field):
                    row[field] = "Unknown"

            if not row.get("productDisplayName"):
                row["productDisplayName"] = f"Product {product_id}"

            row["id"] = product_id
            row["name"] = row["productDisplayName"]
            row["image_path"] = str(image_path)
            row["description"] = build_text(row)

            products.append(row)

            if limit is not None and len(products) >= limit:
                break

    return products