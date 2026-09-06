import io
from pathlib import Path

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from PIL import Image
from pydantic import BaseModel, Field
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from src.app.inference import InferenceService
from src.embedders.fashionclip_embedder import FashionClipEmbedder


class RecommendationRequest(BaseModel):
    text: str
    k: int = Field(default=5, ge=1, le=20)


project_root = Path(__file__).parent.parent.parent
index_dir = project_root / "outputs" / "catalog"
image_dir = project_root / "data" / "fashion" / "images"

real_service = InferenceService(
    index_dir=index_dir,
    embedder=FashionClipEmbedder(),
    alpha=0.5,
)

app = FastAPI(title="Fashion Product Recommender")

app.mount(
    "/images",
    StaticFiles(directory=image_dir),
    name="images",
)


def get_service():
    return real_service

@app.get("/ui", include_in_schema=False)
def user_interface():
     return FileResponse(project_root / "frontend" /"index.html")

def add_image_urls(results):
    items = []

    for result in results:
        item = result.copy()
        filename = Path(item["image_path"]).name
        item["image_url"] = f"/images/{filename}"
        items.append(item)

    return items 

@app.get("/")
def home():
    return {
        "message": "Fashion Product Recommender API",
        "docs": "/docs",
    }


@app.get("/health")
def health(service=Depends(get_service)):
    return {
        "status": "ok",
        "product_count": len(service.products),
    }

@app.post("/recommend")
def recommend(
    request: RecommendationRequest,
    service=Depends(get_service),
):
    results = service.recommend(
        text=request.text,
        k=request.k,
    )

    return {"items": add_image_urls(results)}

@app.post("/recommend/upload")
def recommend_upload(
    image: UploadFile | None = File(default=None),
    text: str | None = Form(default=None),
    k: int = Form(default=5),
    service=Depends(get_service),
):
    clean_text = text.strip() if text else None 

    if image is None and clean_text is None:
        raise HTTPException(
            status_code=400,
            detail="Provide an image or text.",
        )

    if not 1 <= k <= 20:
        raise HTTPException(
            status_code=400,
            detail="k must be between 1 and 20.",
        )

    query_image = None 

    if image is not None:
        try: 
            image_bytes = image.file.read()
            query_image = Image.open(
                io.BytesIO(image_bytes)
            ).convert("RGB")
        except Exception as error:
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is not a valid image",
            ) from error 
    results = service.recommend(
        image=query_image,
        text=clean_text,
        k=k,
    )

    return {"items": add_image_urls(results)}