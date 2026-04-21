import os
import traceback

from flask import Flask, jsonify, render_template, request
from PIL import Image
import torch

from predict import (
    CLASS_NAMES,
    build_model,
    get_gradcam_target_layer,
    load_checkpoint_into_model,
    prepare_image,
    run_prediction,
)
from gradcam import generate_gradcam_base64


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.join(BASE_DIR, "models", "brain_tumor_efficientnetb0_final.pth"),
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Environment-aware mode:
#   APP_ENV=development  -> local Debug behavior
#   APP_ENV=production   -> release / deployed behavior
APP_ENV = os.environ.get("APP_ENV", "development").strip().lower()
IS_DEBUG = APP_ENV in {"dev", "debug", "development", "local"}

SERVICE_NAME = "NeuroScan AI LOCAL FLASK" if IS_DEBUG else "NeuroScan AI"

app = Flask(__name__)

_model = None


def get_model() -> torch.nn.Module:
    global _model
    if _model is None:
        model = build_model(num_classes=len(CLASS_NAMES))
        load_checkpoint_into_model(model, MODEL_PATH, DEVICE)
        model.to(DEVICE)
        model.eval()
        _model = model
    return _model


@app.route("/", methods=["GET"])
def index():
    try:
        return render_template("index.html")
    except Exception:
        return jsonify(
            {
                "service": SERVICE_NAME,
                "status": "ok",
                "message": "Upload a brain MRI image to POST /predict using form field 'file'.",
                "environment": APP_ENV,
            }
        )


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "service": SERVICE_NAME,
            "environment": APP_ENV,
            "device": str(DEVICE),
            "model_loaded": _model is not None,
            "model_path": MODEL_PATH if IS_DEBUG else os.path.basename(MODEL_PATH),
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "Missing upload field 'file'."}), 400

        file = request.files["file"]
        if not file or not file.filename:
            return jsonify({"error": "No file selected."}), 400

        try:
            image = Image.open(file.stream).convert("RGB")
        except Exception:
            return jsonify({"error": "Uploaded file is not a valid image."}), 400

        model = get_model()

        input_tensor, original_rgb = prepare_image(image=image, device=DEVICE)

        pred_idx, predicted_class, confidence, all_probabilities = run_prediction(
            model=model,
            input_tensor=input_tensor,
            class_names=CLASS_NAMES,
        )

        target_layer = get_gradcam_target_layer(model)

        grad_cam_base64 = generate_gradcam_base64(
            model=model,
            target_layer=target_layer,
            input_tensor=input_tensor,
            target_class_idx=pred_idx,
            original_rgb=original_rgb,
        )

        return jsonify(
            {
                "predicted_class": predicted_class,
                "confidence": confidence,
                "all_probabilities": all_probabilities,
                "grad_cam_base64": grad_cam_base64,
            }
        ), 200

    except FileNotFoundError:
        payload = {
            "error": "Model checkpoint not found.",
        }
        if IS_DEBUG:
            payload["model_path"] = MODEL_PATH
        return jsonify(payload), 500

    except Exception as exc:
        payload = {
            "error": str(exc),
            "type": exc.__class__.__name__,
        }
        if IS_DEBUG:
            payload["traceback"] = traceback.format_exc()
        return jsonify(payload), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050 if IS_DEBUG else 5000))
    app.run(host="0.0.0.0", port=port, debug=IS_DEBUG)
