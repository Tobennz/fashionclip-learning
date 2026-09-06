import numpy as np 

from tests.fixtures.fake_embedder import FakeEmbedder 

def test_fake_text_embeddings_have_correct_shape():
    embedder = FakeEmbedder(dimension=4)

    embeddings = embedder.embed_texts([
        "navy blue shirt",
        "red running shoes",
    ])

    assert embeddings.shape == (2, 4)

def test_same_input_produces_same_embedding():
    embedder = FakeEmbedder(dimension=4)

    first = embedder.embed_texts(["navy blue shirt"])
    second = embedder.embed_texts(["navy blue shirt"])

    assert np.allclose(first, second)

def test_different_inputs_produce_different_emveddings():
    embedder = FakeEmbedder(dimension=4)

    embeddings = embedder.embed_texts([
        "navy blue shirt",
        "red running shoes",
    ])

    assert not np.allclose(embeddings[0], embeddings[1])
