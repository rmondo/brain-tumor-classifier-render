from pathlib import Path
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn

# Ensure project imports work whether this file is run from repo root,
# a scripts directory, or an app entrypoint.
CWD = Path.cwd().resolve()
THIS_FILE = Path(__file__).resolve()
CANDIDATES = [THIS_FILE.parent, CWD, THIS_FILE.parent.parent, CWD.parent]
REPO_ROOT = next((p for p in CANDIDATES if (p / "pyproject.toml").exists()), THIS_FILE.parent)
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
for p in [REPO_ROOT, NOTEBOOKS_DIR]:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from brain_tumor import config as cfg
from brain_tumor.data.dataset import get_val_transform
from brain_tumor.models.classifier import BrainTumorClassifier


CLASS_NAMES = cfg.CLASS_NAMES
IMAGE_SIZE = cfg.IMG_SIZE
INFERENCE_TRANSFORM = get_val_transform()


def build_model(num_classes: int) -> nn.Module:
    model = BrainTumorClassifier(num_classes, cfg.DROPOUT)
    return model


def _extract_state_dict(checkpoint: object) -> dict:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
        if all(isinstance(k, str) for k in checkpoint.keys()):
            return checkpoint
    raise TypeError("Unsupported checkpoint format: expected state_dict or checkpoint dict")


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
            f"Checkpoint mismatch while loading {checkpoint_path}. "
            f"Missing keys: {missing} | Unexpected keys: {unexpected}"
        )

    model.to(device)
    model.eval()
    return model


def get_gradcam_target_layer(model: nn.Module) -> nn.Module:
    """
    Return the best layer for Grad-CAM.

    Prefer the last convolutional feature layer from common wrapper structures,
    and fall back to the final Conv2d found anywhere in the model.
    """
    # Common wrapper patterns first.
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

    # Robust fallback: last Conv2d anywhere in the model.
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
