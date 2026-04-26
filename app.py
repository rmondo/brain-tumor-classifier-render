import gc
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
torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "1")))

APP_ENV = os.environ.get("APP_ENV", "development").strip().lower()
IS_DEBUG = APP_ENV in {"dev", "debug", "development", "local"}
SERVICE_NAME = "NeuroScan AI LOCAL FLASK" if IS_DEBUG else "NeuroScan AI"

app = Flask(__name__)

MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "6"))
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

_model = None


def get_model() -> torch.nn.Module:
    """Load model exactly once per worker. Do not load inside every /predict request."""
    global _model
    if _model is None:
        print("Loading NeuroScan AI model...")
        model = build_model(num_classes=len(CLASS_NAMES))
        load_checkpoint_into_model(model, MODEL_PATH, DEVICE)
        model.to(DEVICE)
        model.eval()
        _model = model
        print("Model loaded.")
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
            "max_upload_mb": MAX_UPLOAD_MB,
        }
    )


@app.errorhandler(413)
def file_too_large(_error):
    return jsonify(
        {
            "error": f"Uploaded file is too large. Limit is {MAX_UPLOAD_MB} MB.",
            "type": "PayloadTooLarge",
        }
    ), 413


@app.route("/predict", methods=["POST"])
def predict():
    input_tensor = None
    original_rgb = None

    try:
        if request.content_length and request.content_length > app.config["MAX_CONTENT_LENGTH"]:
            return jsonify(
                {
                    "error": f"Uploaded file is too large. Limit is {MAX_UPLOAD_MB} MB.",
                    "type": "PayloadTooLarge",
                }
            ), 413

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

        grad_cam_base64 = None
        grad_cam_error = None

        try:
            target_layer = get_gradcam_target_layer(model)
            grad_cam_base64 = generate_gradcam_base64(
                model=model,
                target_layer=target_layer,
                input_tensor=input_tensor,
                target_class_idx=pred_idx,
                original_rgb=original_rgb,
            )
        except Exception as grad_exc:
            grad_cam_error = str(grad_exc)
            if IS_DEBUG:
                grad_cam_error = f"{grad_exc}\n{traceback.format_exc()}"

        payload = {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "all_probabilities": all_probabilities,
            "grad_cam_base64": grad_cam_base64,
        }

        if grad_cam_error:
            payload["grad_cam_error"] = grad_cam_error

        return jsonify(payload), 200

    except FileNotFoundError:
        payload = {
            "error": "Model checkpoint not found.",
            "type": "FileNotFoundError",
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

    finally:
        try:
            del input_tensor
            del original_rgb
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050 if IS_DEBUG else 5000))
    app.run(host="0.0.0.0", port=port, debug=IS_DEBUG)
