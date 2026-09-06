from src.app.index_store import save_catalog_index
from src.app.inference import InferenceService
from src.pipeline.fusion import concat_fused
from tests.fixtures.fake_embedder import FakeEmbedder


def test_inference_service_uses_saved_index(tmp_path):
    products = [
        {"id": 1, "name": "Navy Shirt"},
        {"id": 2, "name": "Red Shoes"},
    ]

    image_names = [
        "navy-shirt.jpg",
        "red-shoes.jpg",
    ]

    descriptions = [
        "navy blue casual shirt",
        "red running shoes",
    ]

    embedder = FakeEmbedder(dimension=4)

    image_embeddings = embedder.embed_images(image_names)
    text_embeddings = embedder.embed_texts(descriptions)

    catalog_embeddings = concat_fused(
        image_embeddings,
        text_embeddings,
        alpha=0.5,
    )

    save_catalog_index(
        embeddings=catalog_embeddings,
        products=products,
        output_dir=tmp_path,
    )

    service = InferenceService(
        index_dir=tmp_path,
        embedder=embedder,
    )

    results = service.recommend(
        image="navy-shirt.jpg",
        text="navy blue casual shirt",
        k=1,
    )

    assert len(results) == 1
    assert results[0]["id"] == 1
    assert results[0]["name"] == "Navy Shirt"