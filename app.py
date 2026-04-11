"""
src/app/app.py
──────────────
Brain Tumor MRI Classifier — Flask inference server.
Production-ready for Render.com deployment.

Endpoints
---------
GET  /         → HTML upload UI
POST /predict  → multipart/form-data { file: image } → JSON
GET  /health   → JSON { status, device, model_loaded }
"""

import io
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from PIL import Image

app = Flask(__name__, template_folder="templates")
CORS(app)

# ── Device (Render free tier is CPU-only) ─────────────────────────────────────
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)

# ── Model path: env var takes priority, then repo-relative discovery ─────────
# On Render, set MODEL_PATH in the environment to the absolute path of the
# uploaded .pth file, OR use Render Disk and mount at /data/models/.
_MODEL_NAME = "brain_tumor_efficientnetb0_final.pth"
_APP_DIR = Path(__file__).resolve().parent
_CANDIDATE_ROOTS = [_APP_DIR, _APP_DIR.parent, _APP_DIR.parent.parent]

def _default_model_path() -> Path:
    for root in _CANDIDATE_ROOTS:
        candidate = root / "models" / _MODEL_NAME
        if candidate.exists():
            return candidate
    return _APP_DIR / "models" / _MODEL_NAME

MODEL_PATH = Path(os.environ.get("MODEL_PATH", str(_default_model_path())))

_model = None
_meta  = None

print(f"[INIT] Device      : {DEVICE}", flush=True)
print(f"[INIT] Model path  : {MODEL_PATH}", flush=True)
print(f"[INIT] Model exists: {MODEL_PATH.exists()}", flush=True)


# ── Error handler ─────────────────────────────────────────────────────────────
@app.errorhandler(Exception)
def _handle_error(err):
    traceback.print_exc()
    return jsonify({
        "error"    : str(err),
        "type"     : type(err).__name__,
        "traceback": traceback.format_exc(),
    }), 500


# ── Model loading (lazy, cached) ──────────────────────────────────────────────
def _load_model():
    global _model, _meta
    if _model is not None:
        return _model, _meta

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {MODEL_PATH}\n"
            "Set the MODEL_PATH environment variable to the correct path."
        )

    from efficientnet_pytorch import EfficientNet

    print(f"[MODEL] Loading from {MODEL_PATH} …", flush=True)
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    _meta = ckpt

    n_cls, drop = ckpt["num_classes"], ckpt["dropout"]
    backbone    = EfficientNet.from_pretrained("efficientnet-b0", num_classes=n_cls)
    in_f        = backbone._fc.in_features
    backbone._fc = nn.Sequential(
        nn.BatchNorm1d(in_f), nn.Dropout(drop),
        nn.Linear(in_f, 256), nn.ReLU(inplace=True),
        nn.BatchNorm1d(256),  nn.Dropout(drop / 2),
        nn.Linear(256, n_cls),
    )

    # Strip 'backbone.' prefix added by the training wrapper
    state = {
        (k[len("backbone."):] if k.startswith("backbone.") else k): v
        for k, v in ckpt["model_state_dict"].items()
    }
    backbone.load_state_dict(state)
    backbone.eval().to(DEVICE)
    _model = backbone
    print(f"[MODEL] Loaded on {DEVICE}", flush=True)
    return _model, _meta


def _preprocess(img: Image.Image, meta: dict) -> torch.Tensor:
    return T.Compose([
        T.Resize((meta["image_size"], meta["image_size"])),
        T.ToTensor(),
        T.Normalize(meta["mean"], meta["std"]),
    ])(img.convert("RGB")).unsqueeze(0).to(DEVICE)


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({
        "status"      : "ok",
        "device"      : str(DEVICE),
        "model_loaded": _model is not None,
        "model_path"  : str(MODEL_PATH),
    })


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Send 'file' in multipart/form-data"}), 400
    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400

    try:
        img = Image.open(io.BytesIO(f.read()))
    except Exception as exc:
        return jsonify({"error": f"Cannot read image: {exc}"}), 400

    try:
        model, meta = _load_model()
        x = _preprocess(img, meta)
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": f"Model load/preprocess failed: {exc}"}), 500

    try:
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1).cpu().numpy()[0]
        pred_idx = int(probs.argmax())
        return jsonify({
            "predicted_class"  : meta["class_names"][pred_idx],
            "confidence"       : float(probs[pred_idx]),
            "all_probabilities": {
                cls: float(p) for cls, p in zip(meta["class_names"], probs)
            },
        })
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": f"Inference failed: {exc}"}), 500


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Render injects $PORT; default to 5001 for local dev
    port = int(os.environ.get("PORT", 5001))
    print(f"[MAIN] Starting on 0.0.0.0:{port}", flush=True)
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False, threaded=True)
