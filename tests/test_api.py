import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.app.api import app, get_service


class FakeService:
    def __init__(self):
        self.products = [
            {"id": 1, "name": "Navy Shirt"},
            {"id": 2, "name": "Red Shoes"},
            {"id": 3, "name": "Black Handbag"},
        ]

    def recommend(self, image=None, text=None, k=5):
        results = [
            {
                "id": 1,
                "name": "Navy Shirt",
                "image_path": "1.jpg",
                "score": 0.9,
            },
            {
                "id": 2,
                "name": "Red Shoes",
                "image_path": "2.jpg",
                "score": 0.2,
            },
        ]

        return results[:k]


@pytest.fixture
def client():
    app.dependency_overrides[get_service] = (
        lambda: FakeService()
    )

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


def test_health(client: TestClient):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "product_count": 3,
    }


def test_recommend(client: TestClient):
    response = client.post(
        "/recommend",
        json={
            "text": "navy blue checked men's shirt",
            "k": 2,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert len(data["items"]) == 2
    assert data["items"][0]["name"] == "Navy Shirt"
    assert data["items"][0]["image_url"] == "/images/1.jpg"


def create_test_image():
    image_bytes = io.BytesIO()

    Image.new(
        mode="RGB",
        size=(16, 16),
        color="navy",
    ).save(image_bytes, format="JPEG")

    return image_bytes.getvalue()


def test_recommend_upload(client: TestClient):
    response = client.post(
        "/recommend/upload",
        data={
            "text": "navy blue shirt",
            "k": "2",
        },
        files={
            "image": (
                "shirt.jpg",
                create_test_image(),
                "image/jpeg",
            ),
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert len(data["items"]) == 2
    assert data["items"][0]["name"] == "Navy Shirt"
    assert data["items"][0]["image_url"] == "/images/1.jpg"


def test_recommend_upload_requires_input(client: TestClient):
    response = client.post(
        "/recommend/upload",
        data={"k": "2"},
    )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "Provide an image or text."
    }