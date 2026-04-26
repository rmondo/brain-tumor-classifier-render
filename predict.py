import os
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from efficientnet_pytorch import EfficientNet

CLASS_NAMES = ["glioma", "meningioma", "pituitary", "notumor"]
IMAGE_SIZE = (224, 224)
OVERLAY_MAX_SIDE = int(os.environ.get("OVERLAY_MAX_SIDE", "512"))

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
    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = EfficientNet.from_name("efficientnet-b0")
        in_features = self.backbone._fc.in_features
        self.backbone._fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


def build_model(num_classes: int) -> nn.Module:
    return BrainTumorClassifier(num_classes)


def _extract_state_dict(checkpoint: object) -> dict:
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint and isinstance(checkpoint["model_state_dict"], dict):
            return checkpoint["model_state_dict"]
        if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            return checkpoint["state_dict"]
        if all(isinstance(k, str) for k in checkpoint.keys()):
            return checkpoint
    raise TypeError("Unsupported checkpoint format")


def _rebuild_fc_from_checkpoint(model: nn.Module, state_dict: dict) -> None:
    keys = state_dict.keys()
    has_seq_head = (
        "backbone._fc.0.weight" in keys
        and "backbone._fc.2.weight" in keys
        and "backbone._fc.4.weight" in keys
        and "backbone._fc.6.weight" in keys
    )

    if not has_seq_head:
        return

    bn0_features = state_dict["backbone._fc.0.weight"].shape[0]
    lin2_out = state_dict["backbone._fc.2.weight"].shape[0]
    bn4_features = state_dict["backbone._fc.4.weight"].shape[0]
    lin6_out = state_dict["backbone._fc.6.weight"].shape[0]

    lin2_in = state_dict["backbone._fc.2.weight"].shape[1]
    lin6_in = state_dict["backbone._fc.6.weight"].shape[1]

    if bn0_features != lin2_in:
        raise RuntimeError(
            f"Checkpoint head mismatch: bn0_features={bn0_features} but lin2_in={lin2_in}"
        )
    if bn4_features != lin6_in:
        raise RuntimeError(
            f"Checkpoint head mismatch: bn4_features={bn4_features} but lin6_in={lin6_in}"
        )

    model.backbone._fc = nn.Sequential(
        nn.BatchNorm1d(bn0_features),
        nn.ReLU(inplace=True),
        nn.Linear(lin2_in, lin2_out),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(bn4_features),
        nn.ReLU(inplace=True),
        nn.Linear(lin6_in, lin6_out),
    )


def load_checkpoint_into_model(
    model: nn.Module,
    checkpoint_path: str,
    device: torch.device,
) -> None:
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

    _rebuild_fc_from_checkpoint(model, cleaned_state_dict)
    missing, unexpected = model.load_state_dict(cleaned_state_dict, strict=True)

    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint mismatch. Missing keys: {missing} | Unexpected keys: {unexpected}"
        )

    model.to(device)
    model.eval()


def get_gradcam_target_layer(model: nn.Module) -> nn.Module:
    return model.backbone._conv_head


def _resize_rgb_for_overlay(image: Image.Image, max_side: int = OVERLAY_MAX_SIDE) -> Image.Image:
    image = image.convert("RGB")
    width, height = image.size
    longest = max(width, height)
    if longest <= max_side:
        return image.copy()
    scale = max_side / float(longest)
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def prepare_image(
    image: Image.Image,
    device: torch.device,
) -> Tuple[torch.Tensor, np.ndarray]:
    image = image.convert("RGB")
    overlay_image = _resize_rgb_for_overlay(image, max_side=OVERLAY_MAX_SIDE)
    original_rgb = np.asarray(overlay_image, dtype=np.uint8).copy()
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
