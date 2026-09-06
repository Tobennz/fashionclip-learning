import csv

from src.data.text_template import build_text


def load_products(csv_path):
    products = []

    with open(csv_path, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            row["id"] = int(row["id"])
            row["name"] = row["productDisplayName"]
            row["description"] = build_text(row)

            products.append(row)

    return products
    