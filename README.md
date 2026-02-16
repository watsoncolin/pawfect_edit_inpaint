# Pawfect Edit FLUX Inpainting Worker

Cloud-based inpainting service for the Pawfect Edit app using [FLUX.1-Fill-dev](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev). Handles large masked areas (retractable leash handles, harness clips) where the on-device LaMa model produces artifacts.

## Architecture

FastAPI service deployed on Cloud Run with an L4 GPU, following the same Pub/Sub push pattern as the SAM leash detection worker.

```
NestJS API  →  Pub/Sub (session-processing)  →  Cloud Run (this service)
                                                    ↕
                                              Firebase Storage
                                              (original.jpg, mask_auto.png → edited.jpg, preview.jpg)
```

### Pipeline

1. Receive Pub/Sub push message with `userId` + `sessionId`
2. Update Firestore: `processingStatus = 'processing'`
3. Download `original.jpg` + `mask_auto.png` from GCS
4. Resize to FLUX-compatible dimensions (multiples of 8, max 1024px)
5. Run FLUX.1-Fill-dev inference (28 steps, guidance_scale=30, float8 quantized)
6. Resize result back to original dimensions
7. Upload `edited.jpg` (quality 95) + `preview.jpg` (512px, quality 85)
8. Update Firestore: `processingStatus = 'completed'`

### Key Details

- **Model**: FLUX.1-Fill-dev with `optimum-quanto` float8 quantization + CPU offload
- **GPU**: NVIDIA L4 (24GB VRAM)
- **Concurrency**: 1 (single request at a time)
- **Prompt**: `"seamless natural background matching surrounding area, photorealistic, no leash, no rope, no cord"`

## Project Structure

```
app/
  main.py                  # FastAPI app + lifespan model loading
  routers/
    inpaint.py             # POST /inpaint — Pub/Sub push handler
  services/
    firebase.py            # Firestore + Storage (lazy singleton)
    flux_inpaint.py        # FLUX.1-Fill-dev loading + inference
    pipeline.py            # Download → inpaint → upload → update
  utils/
    image.py               # Decode, resize, JPEG encode
Dockerfile
cloudbuild.yaml
pyproject.toml
```

## Development

```bash
# Install dependencies
uv sync

# Run locally (requires GPU + FIREBASE_CREDENTIALS env var)
uv run uvicorn app.main:app --host 0.0.0.0 --port 8080
```

## Deployment

```bash
# Build and deploy via Cloud Build
gcloud builds submit --config cloudbuild.yaml

# Create Pub/Sub push subscription (one-time, after first deploy)
gcloud pubsub subscriptions create session-processing-push \
  --topic=session-processing \
  --push-endpoint=https://pawfect-edit-inpaint-<hash>.run.app/inpaint \
  --push-auth-service-account=<service-account>@pawfect-edit.iam.gserviceaccount.com
```

## Environment Variables

| Variable | Description |
|---|---|
| `FIREBASE_CREDENTIALS` | Base64-encoded Firebase service account JSON |
| `HF_TOKEN` | Hugging Face token (build-time only, for gated model access) |

## Endpoints

| Endpoint | Description |
|---|---|
| `GET /health` | Liveness probe |
| `GET /ready` | Readiness probe (OK when model is loaded) |
| `POST /inpaint` | Pub/Sub push handler |
