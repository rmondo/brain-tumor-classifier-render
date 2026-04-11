#!/usr/bin/env bash
# upload_model.sh
# ───────────────
# One-time script: uploads the trained model checkpoint to your Render service.
# Run this AFTER training completes locally and BEFORE the first web request.
#
# Prerequisites
# ─────────────
# 1. Install the Render CLI:  https://render.com/docs/cli
#    brew install render  (macOS)  or  npm i -g @render.com/cli
# 2. Authenticate:  render login
# 3. Replace SERVICE_NAME below with your Render service name
#    (visible in the Render dashboard URL: https://dashboard.render.com/web/<SERVICE_NAME>)
#
# Usage
# ─────
#   bash upload_model.sh
#   # or, to override the local model path:
#   MODEL_LOCAL=/path/to/model.pth bash upload_model.sh

set -euo pipefail

SERVICE_NAME="${SERVICE_NAME:-brain-tumor-classifier}"   # ← change this
MODEL_LOCAL="${MODEL_LOCAL:-./models/brain_tumor_efficientnetb0_final.pth}"
MODEL_REMOTE="/data/models/brain_tumor_efficientnetb0_final.pth"

echo "── Checking local model file ──────────────────────────────────────────"
if [[ ! -f "$MODEL_LOCAL" ]]; then
    echo "ERROR: model file not found at $MODEL_LOCAL"
    echo "Run the notebook through Section 17 first to generate the checkpoint."
    exit 1
fi
SIZE=$(du -sh "$MODEL_LOCAL" | cut -f1)
echo "  Found: $MODEL_LOCAL  ($SIZE)"

echo ""
echo "── Uploading to Render service: $SERVICE_NAME ─────────────────────────"
# render scp copies files to/from a running service via SSH
render scp "$MODEL_LOCAL" "$SERVICE_NAME:/data/models/"

echo ""
echo "── Verifying remote file ──────────────────────────────────────────────"
render ssh "$SERVICE_NAME" -- ls -lh /data/models/

echo ""
echo "Done. Hit /health to confirm the service sees the model:"
echo "  curl https://<your-service>.onrender.com/health"
