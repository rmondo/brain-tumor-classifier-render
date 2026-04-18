import os
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms


CLASS_NAMES = ["glioma", "meningioma", "pituitary", "notumor"]
IMAGE_SIZE = (224, 224)
DROPOUT = 0.30

INFERENCE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

class BrainTumorClassifier(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.30):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=None)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)


def build_model(num_classes: int) -> nn.Module:
    return BrainTumorClassifier(num_classes, DROPOUT)


def _extract_state_dict(checkpoint: object) -> dict:
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
        if all(isinstance(k, str) for k in checkpoint.keys()):
            return checkpoint
    raise TypeError("Unsupported checkpoint format")


def load_checkpoint_into_model(
    model: nn.Module,
    checkpoint_path: str,
    device: torch.device,
) -> nn.Module:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = _extract_state_dict(checkpoint)

    cleaned_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith("module."):
            new_key = new_key[len("module."):]
        cleaned_state_dict[new_key] = value

    missing, unexpected = model.load_state_dict(cleaned_state_dict, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint mismatch. Missing keys: {missing} | Unexpected keys: {unexpected}"
        )

    model.to(device)
    model.eval()
    return model


def get_gradcam_target_layer(model: nn.Module) -> nn.Module:
    common_paths = [
        ["backbone", "features"],
        ["model", "features"],
        ["features"],
    ]
    for path in common_paths:
        obj = model
        ok = True
        for attr in path:
            if hasattr(obj, attr):
                obj = getattr(obj, attr)
            else:
                ok = False
                break
        if ok:
            try:
                return obj[-1]
            except Exception:
                pass

    last_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module

    if last_conv is not None:
        return last_conv

    raise AttributeError("Unable to locate a Conv2d target layer for Grad-CAM on model")


def prepare_image(
    image: Image.Image,
    device: torch.device,
) -> Tuple[torch.Tensor, np.ndarray]:
    image = image.convert("RGB")
    original_rgb = np.array(image)
    input_tensor = INFERENCE_TRANSFORM(image).unsqueeze(0).to(device)
    return input_tensor, original_rgb


@torch.no_grad()
def run_prediction(
    model: nn.Module,
    input_tensor: torch.Tensor,
    class_names: List[str],
) -> Tuple[int, str, float, Dict[str, float]]:
    model.eval()
    logits = model(input_tensor)
    probabilities = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

    pred_idx = int(np.argmax(probabilities))
    predicted_class = class_names[pred_idx]
    confidence = float(probabilities[pred_idx])

    all_probabilities = {
        class_name: float(prob)
        for class_name, prob in zip(class_names, probabilities)
    }

    return pred_idx, predicted_class, confidence, all_probabilities