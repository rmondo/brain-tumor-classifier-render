import base64
import gc
import io

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


class _GradCAMHook:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._forward_handle = None
        self._backward_handle = None
        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(_module, _inputs, output):
            self.activations = output.detach()

        def backward_hook(_module, _grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self._forward_handle = self.target_layer.register_forward_hook(forward_hook)
        self._backward_handle = self.target_layer.register_full_backward_hook(backward_hook)

    def remove(self) -> None:
        if self._forward_handle is not None:
            self._forward_handle.remove()
            self._forward_handle = None
        if self._backward_handle is not None:
            self._backward_handle.remove()
            self._backward_handle = None
        self.activations = None
        self.gradients = None


def _compute_gradcam(
    model: torch.nn.Module,
    target_layer: torch.nn.Module,
    input_tensor: torch.Tensor,
    target_class_idx: int,
) -> np.ndarray:
    hook = _GradCAMHook(model=model, target_layer=target_layer)
    output = None
    score = None
    cam_tensor = None

    try:
        model.eval()
        model.zero_grad(set_to_none=True)

        with torch.enable_grad():
            output = model(input_tensor)
            score = output[:, target_class_idx].sum()
            score.backward()

        if hook.activations is None or hook.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations or gradients.")

        activations = hook.activations[0]
        gradients = hook.gradients[0]

        weights = gradients.mean(dim=(1, 2))
        cam_tensor = (weights[:, None, None] * activations).sum(dim=0)
        cam_tensor = F.relu(cam_tensor)

        cam = cam_tensor.detach().cpu().numpy()
        max_value = float(cam.max()) if cam.size else 0.0
        if max_value > 0:
            cam = cam / max_value

        return cam.astype(np.float32)

    finally:
        hook.remove()
        try:
            del output
            del score
            del cam_tensor
        except Exception:
            pass
        model.zero_grad(set_to_none=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _resize_if_needed(image_rgb: np.ndarray, max_side: int = 512) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return image_rgb
    scale = max_side / float(longest)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _overlay_heatmap(
    original_rgb: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.40,
) -> np.ndarray:
    original_rgb = _resize_if_needed(original_rgb, max_side=512)
    h, w = original_rgb.shape[:2]
    cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)

    heatmap = np.uint8(255 * cam_resized)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = np.clip((1.0 - alpha) * original_rgb + alpha * heatmap, 0, 255).astype(np.uint8)
    return overlay


def _rgb_to_base64_jpeg(image_rgb: np.ndarray, quality: int = 82) -> str:
    pil_image = Image.fromarray(image_rgb)
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def generate_gradcam_base64(
    model: torch.nn.Module,
    target_layer: torch.nn.Module,
    input_tensor: torch.Tensor,
    target_class_idx: int,
    original_rgb: np.ndarray,
) -> str:
    cam = None
    overlay = None
    try:
        cam = _compute_gradcam(
            model=model,
            target_layer=target_layer,
            input_tensor=input_tensor,
            target_class_idx=target_class_idx,
        )
        overlay = _overlay_heatmap(original_rgb=original_rgb, cam=cam)
        return _rgb_to_base64_jpeg(overlay, quality=82)
    finally:
        try:
            del cam
            del overlay
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
