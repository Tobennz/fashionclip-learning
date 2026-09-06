from src.embedders.fashionclip_embedder import FashionClipEmbedder 

def test_fashionclip_embedder_loads_lazily():
    embedder = FashionClipEmbedder()

    assert embedder.model is None 
    assert embedder.processor is None 
    assert embedder.device in {"cpu", "cuda"}