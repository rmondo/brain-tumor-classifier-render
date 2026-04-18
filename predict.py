import os
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms


CLASS_NAMES = ["glioma", "meningioma", "pituitary", "notumor"]
IMAGE_SIZE = (224, 224)

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


def build_model(num_classes: int) -> nn.Module:
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def load_checkpoint_into_model(
    model: nn.Module,
    checkpoint_path: str,
    device: torch.device,
) -> None:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    cleaned_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith("module."):
            new_key = new_key[len("module."):]
        if new_key.startswith("backbone."):
            new_key = new_key[len("backbone."):]
        cleaned_state_dict[new_key] = value

    missing, unexpected = model.load_state_dict(cleaned_state_dict, strict=False)

    if missing:
        print("WARNING: Missing keys while loading model:", missing)
    if unexpected:
        print("WARNING: Unexpected keys while loading model:", unexpected)


def get_gradcam_target_layer(model: nn.Module) -> nn.Module:
    return model.features[-1]


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
    logits = model(input_tensor)
    probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()

    pred_idx = int(np.argmax(probabilities))
    predicted_class = class_names[pred_idx]
    confidence = float(probabilities[pred_idx])

    all_probabilities = {
        class_name: float(prob)
        for class_name, prob in zip(class_names, probabilities)
    }

    return pred_idx, predicted_class, confidence, all_probabilities
