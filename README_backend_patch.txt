NeuroScan AI full backend patch (notumor naming)
================================================

Included files
--------------
- app.py
- predict.py
- gradcam.py
- render.yaml
- requirements.txt

Expected project root layout
----------------------------
- app.py
- predict.py
- gradcam.py
- render.yaml
- requirements.txt
- models/brain_tumor_efficientnetb0_final.pth
- templates/index.html   (optional)

API routes
----------
- GET /health
- POST /predict   (multipart form-data field name: file)

Response fields
---------------
- predicted_class
- confidence
- all_probabilities
- grad_cam_base64

Label convention
----------------
This patch preserves your current live label naming:
- glioma
- meningioma
- pituitary
- notumor

Notes
-----
1. This patch is tailored to EfficientNetB0.
2. The Grad-CAM target layer is model.features[-1].
3. The checkpoint loader strips common training prefixes like module. and backbone.
4. If your checkpoint was trained with a different final layer layout or label order, adjust CLASS_NAMES and/or build_model().
5. Test locally before deploying:
   BASE_URL=http://127.0.0.1:8000 bash test_predict.sh mri.jpg
