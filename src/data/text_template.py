METADATA_TEMPLATE = (
    "{gender} {usage} {articleType} in {baseColour} for {season}"
)


def build_text(product):
    metadata_text = METADATA_TEMPLATE.format(
        gender=product["gender"],
        usage=product["usage"],
        articleType=product["articleType"],
        baseColour=product["baseColour"],
        season=product["season"],
    )

    product_name = product["productDisplayName"]

    return f"{product_name}. {metadata_text}"