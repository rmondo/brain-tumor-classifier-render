#!/usr/bin/env bash
set -euo pipefail

# NeuroScan AI backend curl test script
#
# Usage examples:
#   bash test_predict.sh /path/to/mri.jpg
#   BASE_URL=https://brain-tumor-classifier-render.onrender.com bash test_predict.sh /path/to/mri.jpg
#
# Requirements:
#   - curl
#   - python3 (optional, for pretty JSON formatting)

BASE_URL="${BASE_URL:-http://127.0.0.1:5000}"
IMAGE_PATH="${1:-}"

if [[ -z "${IMAGE_PATH}" ]]; then
  echo "Usage: bash test_predict.sh /path/to/mri.jpg"
  exit 1
fi

if [[ ! -f "${IMAGE_PATH}" ]]; then
  echo "Error: file not found: ${IMAGE_PATH}"
  exit 1
fi

echo "==> Checking health endpoint: ${BASE_URL}/health"
curl -sS "${BASE_URL}/health" | python3 -m json.tool || curl -sS "${BASE_URL}/health"

echo
echo "==> Sending image to ${BASE_URL}/predict"
RESPONSE_FILE="$(mktemp)"

curl -sS -X POST "${BASE_URL}/predict"   -F "file=@${IMAGE_PATH}"   -o "${RESPONSE_FILE}"

echo "==> Raw JSON response:"
python3 -m json.tool "${RESPONSE_FILE}" || cat "${RESPONSE_FILE}"

echo
echo "==> Saving Grad-CAM image if present..."
python3 - <<'PY' "${RESPONSE_FILE}"
import sys, json, base64, pathlib

response_path = pathlib.Path(sys.argv[1])
data = json.loads(response_path.read_text())

grad = data.get("grad_cam_base64")
if not grad:
    print("No grad_cam_base64 field found in response.")
    raise SystemExit(0)

out_path = pathlib.Path("gradcam_from_response.png")
out_path.write_bytes(base64.b64decode(grad))
print(f"Saved Grad-CAM image to: {out_path.resolve()}")
PY

echo
echo "==> Done."
