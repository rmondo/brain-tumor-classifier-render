# Deploying the Brain Tumor MRI Classifier — Flask Inference Server

## What you're deploying

The **Flask inference server** (`app.py`) that:
- serves `GET /` → drag-and-drop MRI upload UI
- handles `POST /predict` → returns predicted class + all probabilities as JSON
- exposes `GET /health` → liveness check

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Trained model checkpoint | Place at `models/brain_tumor_efficientnetb0_final.pth` — obtained from training pipeline |
| GitHub / GitLab account | Render deploys from a Git remote |
| Render account | [render.com](https://render.com) — free tier available |
| Render CLI (optional) | Needed only for model upload on Starter plan via `upload_model.sh` |

---

## File checklist

Ensure these files exist at your **project root** before pushing:

```
brain-tumor-classifier-render/
├── render.yaml                          ← Render Blueprint (auto-configures the service)
├── requirements.txt                     ← Flask + PyTorch inference deps (CPU-only)
├── app.py                               ← Flask server (at root, not in src/)
├── config.py                            ← Minimal config (BASE_DIR, LOG_DIR, FLASK_HOST, FLASK_PORT)
├── templates/
│   └── index.html                       ← upload UI (Flask serves from root)
├── static/                              ← optional: static assets
└── models/
    └── brain_tumor_efficientnetb0_final.pth   ← trained weights
```

---

## Step 1 — Verify the repository

```bash
# From your project root
git add render.yaml requirements.txt app.py config.py templates/index.html
git commit -m "Add Render deployment files"
git push origin main
```

> **Model weights:** The `.pth` file is ~47 MB. Do **not** commit it to Git unless  
> you are on the free plan (no Disk available). See Step 4 for the recommended approach.

---

## Step 2 — Create the Render Web Service

### Option A — Blueprint (recommended, uses `render.yaml`)

1. Go to [dashboard.render.com](https://dashboard.render.com) → **New** → **Blueprint**
2. Connect your GitHub/GitLab repo
3. Render detects `render.yaml` and pre-fills all settings
4. Click **Apply** — the service is created and the first build starts

### Option B — Manual setup

1. **New** → **Web Service** → connect your repo
2. Set the following fields:

| Field | Value |
|---|---|
| **Runtime** | Python 3 |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120` |
| **Plan** | Free (or Starter for always-on) |

3. Add environment variable (optional — `app.py` auto-discovers model):

| Key | Value |
|---|---|
| `MODEL_PATH` | `/data/models/brain_tumor_efficientnetb0_final.pth` |

---

## Step 3 — Handle the model weights

### Free plan — commit the model to the repo

The free plan has no persistent disk, so the model must be in the repo or fetched at build time.

**Option A: commit the file** (only practical if < 100 MB and you're OK with it in history)
```bash
git lfs install                                 # enable Git LFS first
git lfs track "*.pth"
git add .gitattributes models/brain_tumor_efficientnetb0_final.pth
git commit -m "Add model weights via LFS"
git push
```
Then set `MODEL_PATH` to the repo-relative path:
```
MODEL_PATH=models/brain_tumor_efficientnetb0_final.pth
```

**Option B: download at build time** — add a `build.sh` and call it from `buildCommand`:
```bash
# build.sh
pip install -r requirements.txt
mkdir -p models
# Download from wherever you store the weights, e.g. a private S3 bucket:
curl -L "$MODEL_DOWNLOAD_URL" -o models/brain_tumor_efficientnetb0_final.pth
```
Set `MODEL_DOWNLOAD_URL` as a secret environment variable in the Render dashboard.

---

### Starter plan ($7/mo) — use a Render Disk

A Disk persists the model across deploys without re-downloading or bloating the Git repo.

**In `render.yaml`**, uncomment the `disk` block:
```yaml
disk:
  name: model-storage
  mountPath: /data/models
  sizeGB: 1
```

**Upload the checkpoint once** using the Render CLI:
```bash
# Install CLI
brew install render        # macOS
# or: npm i -g @render.com/cli

# Authenticate
render login

# Upload (replace <service-name> with your Render service name)
render scp ./models/brain_tumor_efficientnetb0_final.pth brain-tumor-classifier-render:/data/models/

# Or use the convenience script included in this repo:
bash upload_model.sh
```

Verify it landed:
```bash
render ssh brain-tumor-classifier-render -- ls -lh /data/models/
```

---

## Step 4 — Verify the deployment

Once the build goes green, hit your service URL:

```bash
# Health check — confirms the server is up and the model is found
curl https://brain-tumor-classifier-render.onrender.com/health
# Expected: {"status":"ok","device":"cpu","model_loaded":true}
# model_loaded becomes true after the first /predict request triggers lazy load

# Test inference
curl -X POST https://brain-tumor-classifier-render.onrender.com/predict \
     -F "file=@/path/to/test_mri.jpg"
# Expected: {"predicted_class":"glioma","confidence":0.94,"all_probabilities":{...}}

# Open the UI
open https://brain-tumor-classifier-render.onrender.com
```

---

## Environment variables reference

| Variable | Required | Default | Description |
|---|---|---|---|
| `PORT` | Auto | set by Render | Port gunicorn binds to — **do not set manually** |
| `MODEL_PATH` | No | auto-discovered | Absolute path to the `.pth` checkpoint; `app.py` searches `models/` first |
| `MODEL_DOWNLOAD_URL` | Only if using build-time download | — | Pre-signed URL or public URL to fetch weights during build |

---

## Render free tier limitations

| Limit | Impact |
|---|---|
| **Spins down after 15 min inactivity** | First request after sleep takes 30–60 s (model load + cold start) |
| **512 MB RAM** | CPU-only PyTorch is required; CUDA builds won't fit |
| **No persistent disk** | Model must be in the Git repo or fetched at build time |
| **750 build minutes/month** | Sufficient for occasional redeploys |

Upgrade to the **Starter** plan ($7/mo) for always-on instances and Disk support.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Build fails: `No matching distribution for torch` | Older torch version pinned | Ensure `requirements.txt` pins `torch==2.11.0  torchvision==0.26.0` (both available on PyPI) |
| `FileNotFoundError: Checkpoint not found` | `MODEL_PATH` env var wrong or file not uploaded | Check `MODEL_PATH` in dashboard → Environment; re-run `upload_model.sh`; or place at `models/` in repo |
| `502 Bad Gateway` on first request | gunicorn timeout during model load | Increase `--timeout` in start command (already 120 s) |
| App spins up but `/predict` returns 500 | `backbone.` key mismatch | The `app.py` in this repo already strips the prefix; make sure you're using this version |
| Free tier cold start is very slow | Normal — model loading takes ~10 s on CPU | Use Starter plan for always-on, or pre-warm with a scheduled health-check ping |

---

## Updating the model

After retraining, upload the new weights and redeploy:

```bash
# Upload new checkpoint (Starter plan with Disk)
bash upload_model.sh

# Trigger a redeploy (or just push any commit)
render deploys create brain-tumor-classifier-render
```

---

> ⚠️ **Medical Disclaimer** — This deployment is for research and educational purposes only.  
> Model outputs must not substitute for clinical diagnosis.
