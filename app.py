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
                "service": "NeuroScan AI",
                "status": "ok",
                "message": "Upload a brain MRI image to POST /predict using form field 'file'.",
            }
        )


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "device": str(DEVICE),
            "model_loaded": _model is not None,
            "model_path": MODEL_PATH,
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
        return jsonify(
            {
                "error": "Model checkpoint not found.",
                "model_path": MODEL_PATH,
            }
        ), 500
    except Exception as exc:
        return jsonify(
            {
                "error": str(exc),
                "type": exc.__class__.__name__,
                "traceback": traceback.format_exc(),
            }
        ), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
