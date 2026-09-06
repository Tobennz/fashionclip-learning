from pathlib import Path 

from src.app.inference import InferenceService 
from src.embedders.fashionclip_embedder import FashionClipEmbedder 

def main():
    project_root = Path(__file__).parent.parent 
    index_dir = project_root / "outputs" / "catalog"

    embedder = FashionClipEmbedder()

    service = InferenceService(
        index_dir=index_dir,
        embedder=embedder,
        alpha=0.5,
    )

    query = "navy blue checked men's shirt"

    print(f"\nQuery: {query}")
    print("Searching saved catalog...\n")

    results = service.recommend(
        text=query,
        k=5,
    )

    for rank, product in enumerate(results, start=1):
        print(
            f"{rank}. {product['name']} "
            f"(score: {product['score']:.3f})"
        )

if __name__=="__main__":
    main()