# Architecture

**Analysis Date:** 2026-02-21

## Pattern Overview

**Overall:** Pub/Sub Worker Pattern with Asynchronous Image Processing

**Key Characteristics:**
- Event-driven worker service responding to Google Cloud Pub/Sub messages
- Single-threaded GPU inference with background model loading
- Multi-step stateful pipeline: download → process → upload → update
- Lazy-loaded Firebase dependencies (Firestore + Storage)
- Graceful handling of session deletion during processing

## Layers

**API/Handler Layer:**
- Purpose: Receive and validate Pub/Sub push messages, coordinate request lifecycle
- Location: `app/routers/inpaint.py`
- Contains: HTTP endpoint (`POST /inpaint`), message envelope decoding, error classification
- Depends on: Pipeline orchestration, model readiness checks
- Used by: Pub/Sub service → Cloud Run

**Service/Business Logic Layer:**
- Purpose: Orchestrate the inpainting workflow, manage state transitions
- Location: `app/services/pipeline.py`
- Contains: Multi-step pipeline (session validation → download → resize → inpaint → upload → update), error handling with Firestore fallback, composite operation
- Depends on: Firebase (storage + firestore), FLUX model inference, image utilities
- Used by: Inpaint router, error recovery paths

**Infrastructure/Integration Layer:**
- Purpose: Manage external service integrations (Firebase Admin SDK, Hugging Face model loading)
- Location: `app/services/firebase.py`, `app/services/flux_inpaint.py`
- Contains: Firebase client initialization (lazy singleton), model lifecycle (load, is_ready), storage/firestore operations
- Depends on: Google Cloud credentials (via environment), Hugging Face Hub, PyTorch
- Used by: Pipeline, router, main

**Utility Layer:**
- Purpose: Reusable image processing operations
- Location: `app/utils/image.py`
- Contains: Decoding (with EXIF correction), resizing (FLUX-compatible dimensions), encoding (JPEG), preview generation
- Depends on: Pillow
- Used by: Pipeline

**Application Layer:**
- Purpose: Initialize FastAPI app, manage model lifespan, expose health probes
- Location: `app/main.py`
- Contains: FastAPI factory with lifespan context manager, background model loader thread, health + readiness endpoints
- Depends on: Flask, model loading service
- Used by: Cloud Run container runtime

## Data Flow

**Inpainting Request Flow:**

1. Pub/Sub service pushes message to `/inpaint` endpoint (Cloud Run ingress)
2. Router handler decodes base64-encoded JSON payload → extracts `userId`, `sessionId`
3. Model readiness check: if not ready, return 503 to trigger Pub/Sub retry
4. Async dispatch to thread pool (keeps event loop responsive for health probes)
5. Pipeline validates session exists in Firestore
6. Download `original.jpg` + `mask_auto.png` from Firebase Storage
7. Image preprocessing: decode → apply EXIF orientation → resize to FLUX dimensions (multiples of 8, ≤1024px)
8. FLUX.1-Fill-dev inference (28 steps, guidance_scale=10, Q4 GGUF transformer with CPU offload)
9. Postprocessing: resize result to original dimensions → composite with mask to preserve unmasked pixels
10. Encode outputs: high-quality final (`edited.jpg`, quality=95) + thumbnail (`preview.jpg`, 512px, quality=85)
11. Upload both to Firebase Storage
12. Update Firestore session: `processingStatus='completed'`, store paths and cost
13. Return 200 OK to Pub/Sub

**State Management:**

Session state machine in Firestore:
- `processingStatus: 'pending'` → Job received, queued
- `processingStatus: 'processing'` → Pipeline started, downloading assets
- `processingStatus: 'completed'` → Success, `editedImagePath` and `previewImagePath` set, cost recorded
- `processingStatus: 'failed'` → Error occurred, `errorMessage` set

Session deletion detection: If session document is removed during download phase, `firebase.session_exists()` returns false, pipeline exits gracefully without error.

**Error Handling Strategy:**

- **Model not ready (503)**: Handler returns 503, Pub/Sub retries
- **Missing fields in message (400)**: Handler logs, returns 200 to prevent infinite retries
- **Session deleted during processing (NotFound)**: Pipeline catches, logs warning, returns 200 to ACK
- **Transient errors (network, temp GPU issue)**: Router checks `deliveryAttempt` in envelope, returns 500 if < 5 attempts (Pub/Sub retries), returns 200 + error logged if >= 5 (max retries exceeded)
- **Permanent errors (invalid image, inference OOM)**: Pipeline catches, updates Firestore with error status, re-raises to return 500 and eventually hit max retries

## Key Abstractions

**Firebase Lazy Singleton:**
- Purpose: Defer Firebase Admin SDK initialization until first use, avoid credential parsing on cold start
- Examples: `firebase.py` — `_init()` guards initialization, `get_firestore_client()` and `get_storage_bucket()` call `_init()` on first use
- Pattern: Module-level `_initialized` flag, lazy factory functions

**Model Lifecycle State:**
- Purpose: Track whether FLUX.1-Fill-dev is ready to infer (loaded + warmup complete)
- Examples: `flux_inpaint.py` — `_pipe` module variable, `is_ready()` check, `load_model()` async load
- Pattern: Singleton pattern with module-level state

**Image Dimension Compatibility:**
- Purpose: Ensure all resizing operations produce FLUX-compatible dimensions (multiples of 8)
- Examples: `image.py` — `resize_for_flux(image, max_size=1024)` normalizes to 8px grid
- Pattern: Centralized utility function to enforce FLUX constraints

**Composite Operation:**
- Purpose: Apply only masked region from FLUX output, preserve original pixels outside mask
- Examples: `pipeline.py` — Line 60: `Image.composite(result, image, mask.convert("L"))`
- Pattern: Prevents color drift in unmasked regions by blending inference output with original

## Entry Points

**HTTP Endpoints:**
- Location: `app/main.py` + `app/routers/inpaint.py`
- `GET /health`: Liveness probe, always returns `{"status": "ok"}`
- `GET /ready`: Readiness probe, returns 200 if model loaded, 503 if still loading
- `POST /inpaint`: Pub/Sub message handler, triggers pipeline

**Application Startup:**
- Location: `app/main.py` — `lifespan` context manager
- Triggers: Cloud Run container startup
- Responsibilities: Spawn background thread for `load_model()`, log startup state

**Background Task:**
- Location: `app/main.py` — `_load_model_background()` thread
- Triggers: Called during FastAPI lifespan startup
- Responsibilities: Load FLUX.1-Fill-dev model in background, log success/failure

## Cross-Cutting Concerns

**Logging:**
- Framework: Python standard `logging` module
- Pattern: Module-level logger created with `logging.getLogger(__name__)`, configured at startup in `main.py` with `basicConfig(level=logging.INFO)`
- Scope: Info-level logs for key steps (model loading, request start/completion), warnings for transient issues (model not ready, session deleted), errors for failures

**Error Recovery:**
- Pub/Sub message acknowledgment: 200 OK = message processed successfully or error is permanent (bad message, max retries); 500 = transient error, Pub/Sub retries
- Firestore error state: Pipeline catches exceptions and updates session document with `processingStatus='failed'` and error message
- Session deletion fallback: Pipeline checks `session_exists()` before starting work, gracefully exits if removed

**Inference Concurrency:**
- Single GPU constraint: Cloud Run deployment with `--concurrency=1` and `--max-instances=1`
- Async event loop handling: Router runs pipeline in thread pool via `asyncio.to_thread()` to keep event loop responsive for health probes during long inference (28 steps = ~60-90 seconds)

---

*Architecture analysis: 2026-02-21*
