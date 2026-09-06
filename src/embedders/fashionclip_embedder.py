import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor

from src.pipeline.fusion import l2_normalize


class FashionClipEmbedder:
    def __init__(
        self,
        model_name="patrickjohncyh/fashion-clip",
        device=None,
    ):
        self.model_name = model_name

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.model = None
        self.processor = None

    def load_model(self):
        if self.model is not None:
            return

        print(f"Loading {self.model_name} on {self.device}...")

        self.model = CLIPModel.from_pretrained(
            self.model_name
        ).to(self.device)

        self.processor = CLIPProcessor.from_pretrained(
            self.model_name
        )

        self.model.eval()

    def embed_texts(self, texts):
        self.load_model()

        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )

        inputs = {
            name: tensor.to(self.device)
            for name, tensor in inputs.items()
        }

        with torch.inference_mode():
            features = self.model.get_text_features(**inputs)

        features = self._extract_tensor(features)

        return l2_normalize(
            features.cpu().numpy().astype(np.float32)
        )

    def embed_images(self, images):
        self.load_model()

        inputs = self.processor(
            images=images,
            return_tensors="pt",
        )

        inputs = {
            name: tensor.to(self.device)
            for name, tensor in inputs.items()
        }

        with torch.inference_mode():
            features = self.model.get_image_features(**inputs)

        features = self._extract_tensor(features)

        return l2_normalize(
            features.cpu().numpy().astype(np.float32)
        )

    def _extract_tensor(self, features):
        if hasattr(features, "pooler_output"):
            return features.pooler_output

        return features