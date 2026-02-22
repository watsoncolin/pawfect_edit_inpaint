# External Integrations

**Analysis Date:** 2026-02-21

## APIs & External Services

**Machine Learning Models:**
- Hugging Face Hub - Model hosting and download
  - SDK/Client: `transformers`, `diffusers`, `huggingface_hub`
  - Auth: Environment variable `HF_TOKEN` (build-time only)
  - Models accessed:
    - `YarvixPA/FLUX.1-Fill-dev-GGUF` - Quantized GGUF model weights
    - `black-forest-labs/FLUX.1-Fill-dev` - Base model architecture, tokenizer, scheduler

**Message Queue:**
- Google Cloud Pub/Sub - Asynchronous job queue
  - Topic: `session-processing` (receives inpainting job messages)
  - Subscription: `session-processing-push` (Cloud Run push endpoint)
  - Message format: JSON with base64-encoded payload containing `userId` and `sessionId`
  - Authentication: Service account (configured in Cloud Run)
  - Handler: `POST /inpaint` in `app/routers/inpaint.py`

## Data Storage

**Databases:**
- Firestore (Google Cloud)
  - Connection: Firebase Admin SDK (initialized via `FIREBASE_CREDENTIALS`)
  - Client: `firebase_admin.firestore`
  - Collections:
    - `users/{userId}/sessions/{sessionId}` - Session documents with processing status
    - Fields: `processingStatus`, `editedImagePath`, `previewImagePath`, `cost`, `errorMessage`

**File Storage:**
- Firebase Cloud Storage (Google Cloud Storage)
  - Connection: Firebase Admin SDK
  - Client: `firebase_admin.storage`
  - Storage bucket: Dynamic, derived from Firebase project ID in credentials
  - Paths:
    - `users/{userId}/sessions/{sessionId}/original.jpg` - Input image
    - `users/{userId}/sessions/{sessionId}/mask_auto.png` - Segmentation mask
    - `users/{userId}/sessions/{sessionId}/edited.jpg` - FLUX inpainting output (quality 95)
    - `users/{userId}/sessions/{sessionId}/preview.jpg` - Thumbnail preview (512px, quality 85)

**Caching:**
- None - No explicit caching layer. Model weights cached locally in Docker image.

## Authentication & Identity

**Auth Provider:**
- Google Cloud Service Account (Firebase Admin SDK)
  - Implementation: Certificate-based authentication via `FIREBASE_CREDENTIALS` environment variable
  - Format: Base64-encoded JSON service account key
  - Scope: Full Firebase Admin access (Firestore, Cloud Storage)
  - Secret stored in: Google Cloud Secret Manager (`FIREBASE_CREDENTIALS:latest`)

**API Authentication:**
- Google Cloud Service Account (Cloud Run service account)
  - Pub/Sub messages verified via X-Goog-IAM-Authority-Selector and X-Goog-IAM-Authorization-Token headers
  - Cloud Run service account grants permissions to pull from Pub/Sub

## Monitoring & Observability

**Error Tracking:**
- None - Errors logged to standard output (Cloud Logging)

**Logs:**
- Google Cloud Logging (via Cloud Run)
  - Log level: INFO
  - Logging configured in `app/main.py` and throughout services
  - Logs include: model loading, Pub/Sub messages, inference timing, Firestore updates, errors
  - Log format: Python logging module standard format

**Health Checks:**
- Cloud Run startup probe: `GET /ready` (returns 503 until model loaded, succeeds after 60 attempts × 10s = 600s timeout)
- Cloud Run liveness probe: `GET /health` (always returns 200)

## CI/CD & Deployment

**Hosting:**
- Google Cloud Run (managed serverless container platform)
  - Service name: `pawfect-edit-inpaint`
  - Region: `us-east4`
  - Concurrency: 1 (handles one request at a time)
  - Timeout: 300 seconds (5 minutes per request)
  - GPU: 1 × NVIDIA L4
  - CPU: 8 cores
  - Memory: 32GB
  - Min instances: 0 (scales to zero when idle)
  - Max instances: 1 (prevents concurrent requests)

**CI Pipeline:**
- GitHub Actions workflow (`.github/workflows/deploy.yml`)
  - Triggers on: push to main branch OR manual `workflow_dispatch`
  - Steps:
    1. Checkout code
    2. Authenticate with GCP using `GCP_SA_KEY` secret
    3. Set up Google Cloud SDK
    4. Trigger Cloud Build via `gcloud builds submit`

- Google Cloud Build (managed CI/CD platform)
  - Config file: `cloudbuild.yaml`
  - Steps:
    1. Build Docker image (with `HF_TOKEN` build arg for gated model)
    2. Push to Container Registry (`gcr.io/pawfect-edit/pawfect-edit-inpaint`)
    3. Deploy to Cloud Run with GPU and secrets

**Build Environment:**
- Machine type: `E2_HIGHCPU_32` (high-CPU build machine for fast compilation)
- Build timeout: 3600 seconds (1 hour)

## Environment Configuration

**Required env vars (Runtime):**
- `FIREBASE_CREDENTIALS` - Base64-encoded Firebase service account JSON (secret)
  - Source: Google Cloud Secret Manager (`FIREBASE_CREDENTIALS:latest`)
  - Used by: `app/services/firebase.py`

**Required env vars (Build-time):**
- `HF_TOKEN` - Hugging Face API token for gated model access (secret)
  - Source: Google Cloud Secret Manager (`HF_TOKEN:latest`)
  - Used by: Dockerfile during model pre-download
  - Cleared from final image: `ENV HF_TOKEN=` (line 38 in Dockerfile)

**Build args:**
- `HF_TOKEN` - Passed as Docker build arg, not persisted in image

**Secrets location:**
- Google Cloud Secret Manager (project: `pawfect-edit`)
  - `FIREBASE_CREDENTIALS` - Firebase service account key
  - `HF_TOKEN` - Hugging Face token
  - Referenced in: `cloudbuild.yaml` (secretEnv) and Cloud Run deployment config

**Environment variables (Docker):**
- `PYTHONUNBUFFERED=1` - Unbuffered stdout
- `DEBIAN_FRONTEND=noninteractive` - Non-interactive apt installations
- `APP_HOME=/app` - Application directory
- `UV_COMPILE_BYTECODE=1` - Bytecode compilation
- `UV_LINK_MODE=copy` - Dependency linking mode
- `CUDA_MODULE_LOADING=LAZY` - Lazy CUDA loading
- `HF_HUB_ENABLE_HF_TRANSFER=1` - Optimized Hugging Face downloads

## Webhooks & Callbacks

**Incoming:**
- `POST /inpaint` - Google Cloud Pub/Sub push endpoint
  - Receives JSON envelope with base64-encoded message
  - Payload: `{"userId": "...", "sessionId": "..."}`
  - Pub/Sub service account: `<service-account>@pawfect-edit.iam.gserviceaccount.com`
  - Handler: `app/routers/inpaint.py:handle_inpaint()`

**Outgoing:**
- Firestore updates via Firebase Admin SDK
  - No direct webhooks, but Firestore can trigger Cloud Functions (not currently integrated)
- Cloud Storage updates via Firebase Admin SDK
  - No webhooks, but Cloud Storage can trigger Pub/Sub topics

---

*Integration audit: 2026-02-21*
