# Architecture Research

**Domain:** Async GPU inference worker with scene-labeling pre-processing and optimization pipeline
**Researched:** 2026-02-21
**Confidence:** MEDIUM (existing service is HIGH, new components are MEDIUM based on GCP/Diffusers docs + WebSearch)

---

## Standard Architecture

### System Overview

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              iOS App                                          │
│  User selects area → SAM2 on-device creates mask → large area → server       │
└───────────────────────────────────┬──────────────────────────────────────────┘
                                    │ POST /inpaint-request
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                          API Server (existing)                                │
│                                                                               │
│   ┌──────────────────┐        ┌───────────────────────────────┐               │
│   │  Enqueue Pub/Sub │        │  Trigger SAM3 Scene Labeling  │               │
│   │  message         │        │  (HTTP call to SAM3 service)  │               │
│   └─────────┬────────┘        └──────────────┬────────────────┘               │
│             │  (parallel fire-and-forget)     │                               │
└─────────────┼──────────────────────────────── ┼──────────────────────────────┘
              │                                  │
              ▼                                  ▼
┌─────────────────────┐          ┌──────────────────────────────────────────────┐
│   Google Pub/Sub    │          │          SAM3 Labeling Service                │
│   (push topic)      │          │  1. Receives image path                       │
└────────┬────────────┘          │  2. Runs SAM3 segmentation with surface       │
         │                       │     concept prompts ("grass","concrete",etc)  │
         │                       │  3. Classifies dominant background surface     │
         │                       │  4. Writes labels to Firestore session doc     │
         │                       └──────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                   Inpaint Service (this codebase — Cloud Run + T4)            │
│                                                                               │
│  POST /inpaint ← Pub/Sub push                                                 │
│        │                                                                      │
│        ▼                                                                      │
│  [1] Check model ready (503 → Pub/Sub nack → retry)                          │
│        │                                                                      │
│        ▼                                                                      │
│  [2] Read Firestore session doc                                               │
│        │                                                                      │
│        ├── sceneLabels present? ──YES──▶ [3] Build context-aware prompt       │
│        │                                                                      │
│        └── sceneLabels absent?  ──NO───▶ [3'] Requeue via 500 NACK           │
│              (if deliveryAttempt < max and age < threshold)                   │
│                                                                               │
│  [4] Download image + mask from Firebase Storage                              │
│        │                                                                      │
│        ▼                                                                      │
│  [5] Clean mask artifacts (morphological ops)                                 │
│        │                                                                      │
│        ▼                                                                      │
│  [6] Run inference (FluxFillPipeline / model registry)                        │
│        │                                                                      │
│        ▼                                                                      │
│  [7] Composite result + upload + update Firestore                             │
└──────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  Firebase Storage  │  Firestore     │
│  - original.jpg    │  - session doc │
│  - mask_auto.png   │  - sceneLabels │
│  - edited.jpg      │  - status      │
│  - preview.jpg     │  - cost        │
└─────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Current State |
|-----------|----------------|---------------|
| iOS App | Mask generation via SAM2 on-device, triggers server inpaint | Existing, unchanged |
| API Server | Receives inpaint requests, fans out Pub/Sub enqueue + SAM3 trigger in parallel | Existing; SAM3 trigger is new |
| Google Pub/Sub | Durable async queue with push delivery and exponential backoff retry | Existing |
| SAM3 Service | Accepts image path, runs segmentation with surface concept prompts, writes `sceneLabels` to Firestore | Existing (binary masks only); needs extension |
| Inpaint Service | GPU inference worker; checks for labels, builds prompt, runs FLUX, uploads output | Existing; needs label check + prompt logic |
| Firebase Storage | Image and mask binary storage; structured by `users/{uid}/sessions/{sid}/` | Existing |
| Firestore | Session state machine + metadata; `sceneLabels` field added as intermediate result | Existing; `sceneLabels` field is new |

---

## Component Boundaries

### What Talks to What

```
API Server        → Pub/Sub topic         (enqueue job)
API Server        → SAM3 Service HTTP     (trigger labeling, fire-and-forget)
SAM3 Service      → Firestore             (write sceneLabels to session doc)
Pub/Sub           → Inpaint Service HTTP  (push delivery to /inpaint)
Inpaint Service   → Firestore             (read sceneLabels, update status)
Inpaint Service   → Firebase Storage      (download image/mask, upload result)
```

### Boundaries That Must NOT Cross

- Inpaint Service must NOT call SAM3 Service directly (creates synchronous dependency on GPU worker)
- Inpaint Service must NOT block waiting for labels (async requeue instead)
- SAM3 Service must NOT publish to Pub/Sub (creates circular dependency)
- API Server must NOT wait for SAM3 response (fire-and-forget keeps request latency low)

---

## Data Flow

### Happy Path (Labels Ready)

```
1. iOS app → API server: POST /inpaint-request {userId, sessionId, imagePath}
2. API server (parallel):
   a. Enqueues Pub/Sub message {userId, sessionId}
   b. HTTP POST to SAM3 service {imagePath, sessionId} — no await
3. SAM3 service:
   a. Downloads image from Storage
   b. Runs SAM3 with concept prompts: ["grass","concrete","wood","sand","gravel","leaves","tile","water"]
   c. Identifies dominant background surface within masked region's neighborhood
   d. Writes to Firestore: session.sceneLabels = {surface: "grass", confidence: 0.87, detail: "patchy grass with fallen leaves"}
4. Pub/Sub delivers to Inpaint Service POST /inpaint
5. Inpaint service:
   a. Reads Firestore session doc
   b. Finds sceneLabels.surface = "grass", constructs prompt:
      "patchy grass with fallen leaves, natural ground surface, seamless texture"
   c. Downloads original.jpg + mask_auto.png
   d. Cleans mask artifacts (morphological erosion/dilation)
   e. Runs FluxFillPipeline inference (optimized)
   f. Composites result, uploads edited.jpg + preview.jpg
   g. Updates Firestore: {processingStatus: "completed", ...}
```

### Race Condition Path (Labels Not Yet Ready)

```
1-4. Same as above
5. Inpaint service reads Firestore session doc
6. sceneLabels field absent or null
7. Inpaint service checks deliveryAttempt:
   - If deliveryAttempt <= 3 AND job age < 60s: return HTTP 500 (Pub/Sub nacks, retries with backoff)
   - If deliveryAttempt > 3 OR job age > 60s: proceed with fallback generic prompt
   - If deliveryAttempt >= 5: ack (existing max retry guard)
8. Pub/Sub retries with exponential backoff (typically 10-30s interval at attempt 2-3)
```

**Key design insight:** Pub/Sub exponential backoff (10-600s configurable) provides natural wait time. At attempt 2, backoff is ~10s — enough for SAM3 to complete its 2-5s labeling. No custom delay infrastructure needed.

### Label Storage: Firestore Session Document (Recommended)

Store `sceneLabels` directly on the existing session Firestore document rather than as a subcollection, a separate Storage file, or in the Pub/Sub message payload.

**Rationale:**
- Session document already exists and is read by inpaint service at job start — zero extra reads
- Labels are small (< 1KB), well within document size constraints
- Pub/Sub message payload modification requires API server coordination and payload size discipline
- Storage file read adds latency and cold path complexity
- Subcollection is unnecessary complexity for a single small metadata object

**Firestore document structure addition:**

```
users/{userId}/sessions/{sessionId}:
  processingStatus: "pending" | "processing" | "completed" | "failed"
  sceneLabels:
    surface: "grass"
    detail: "patchy grass with fallen leaves"
    confidence: 0.87
    labeledAt: <timestamp>
  editedImagePath: "..."
  previewImagePath: "..."
  cost: 0.03
```

---

## Architectural Patterns

### Pattern 1: Fire-and-Forget Parallel Pre-computation

**What:** API server triggers SAM3 labeling asynchronously (no await) at the same time as Pub/Sub enqueue. Both happen in parallel before the response returns to the iOS app.

**When to use:** When a downstream computation (labeling) feeds a later worker (inpaint) but the upstream caller (API server) doesn't need the result.

**Trade-offs:**
- Pro: Zero added latency to the user-facing request
- Pro: SAM3 has the full Pub/Sub queue time (typically 1-10s) to complete before inpaint worker picks up the job
- Con: If SAM3 fails silently, inpaint worker must handle missing labels gracefully
- Con: No built-in confirmation that SAM3 was triggered successfully

**Example (API server, conceptual):**

```python
async def handle_inpaint_request(user_id: str, session_id: str, image_path: str):
    # Fire both in parallel, don't await SAM3
    await asyncio.gather(
        pubsub_client.publish(topic, {"userId": user_id, "sessionId": session_id}),
        trigger_sam3_no_wait(session_id, image_path),  # fire-and-forget
    )
    return {"status": "queued"}

async def trigger_sam3_no_wait(session_id: str, image_path: str):
    """Non-blocking SAM3 trigger — failure is logged but not propagated."""
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            await client.post(SAM3_URL, json={"sessionId": session_id, "imagePath": image_path})
    except Exception:
        logger.warning(f"SAM3 trigger failed for {session_id}, will fall back to generic prompt")
```

### Pattern 2: Graceful Label Wait via Pub/Sub Backoff Nack

**What:** Inpaint worker returns HTTP 500 when labels are not yet available. Pub/Sub nacks the message and retries with exponential backoff, naturally providing the wait time SAM3 needs.

**When to use:** When a pre-computation dependency has bounded latency (< 30s) and Pub/Sub retry timing aligns with that window.

**Trade-offs:**
- Pro: No new infrastructure (no Cloud Tasks, no polling loop, no sleep())
- Pro: Pub/Sub backoff is configurable (10s minimum, up to 600s maximum)
- Pro: Existing retry limit (5 attempts) prevents infinite loops
- Con: Adds 1 extra delivery cycle if SAM3 isn't done — increases wall-clock latency by one backoff interval
- Con: Must distinguish "waiting for labels" from "transient error" — use the same 500 NACK path but log differently

**Requeue guard logic:**

```python
def should_wait_for_labels(delivery_attempt: int, job_queued_at: datetime) -> bool:
    """Return True if we should nack and wait for SAM3 labels."""
    age_seconds = (datetime.utcnow() - job_queued_at).total_seconds()
    # Only wait on early attempts and within reasonable time window
    return delivery_attempt <= 3 and age_seconds < 90

# In pipeline:
scene_labels = session_doc.get("sceneLabels")
if not scene_labels:
    if should_wait_for_labels(delivery_attempt, job_queued_at):
        logger.info(f"Labels not ready, nacking for retry (attempt {delivery_attempt})")
        raise LabelNotReadyError()  # caller returns 500
    else:
        logger.warning("Labels never arrived, using fallback generic prompt")
        scene_labels = {"surface": "ground", "detail": "natural ground surface"}
```

### Pattern 3: Model Registry for A/B Testing and Switching

**What:** Decouple model loading from model selection. A registry maps model names to loader functions. An environment variable or Firestore config selects the active model at startup. Each model variant lives in its own module.

**When to use:** When evaluating multiple inpainting models (FLUX, SDXL-Inpaint, SD3, future models) without code changes between deployments.

**Trade-offs:**
- Pro: Safe A/B testing via Cloud Run traffic splitting (two revisions, each with different MODEL_VARIANT env var)
- Pro: Rollback is instant (traffic split revert in Cloud Run console)
- Pro: Clean separation — each model variant owns its prompt construction logic
- Con: Cannot switch models at runtime without restart (T4 VRAM can't hold two models)
- Con: Requires separate Docker builds per model variant that bakes weights at build time

**Implementation:**

```python
# app/services/model_registry.py
import os
from typing import Protocol
from PIL import Image

class InpaintModel(Protocol):
    def inpaint(self, image: Image.Image, mask: Image.Image, prompt: str) -> Image.Image: ...
    def is_ready(self) -> bool: ...

MODEL_VARIANT = os.getenv("MODEL_VARIANT", "flux_fill_q8")

def load_active_model() -> InpaintModel:
    if MODEL_VARIANT == "flux_fill_q8":
        from app.services.models.flux_fill import FluxFillModel
        return FluxFillModel()
    elif MODEL_VARIANT == "sdxl_inpaint":
        from app.services.models.sdxl_inpaint import SDXLInpaintModel
        return SDXLInpaintModel()
    else:
        raise ValueError(f"Unknown MODEL_VARIANT: {MODEL_VARIANT}")
```

**Cloud Run A/B setup (no code changes):**
```
Revision A: MODEL_VARIANT=flux_fill_q8  → receives 80% traffic
Revision B: MODEL_VARIANT=sdxl_inpaint  → receives 20% traffic
```

Metrics (inference time, quality flags from user feedback) are compared across revisions.

### Pattern 4: Prompt Builder with Scene Labels

**What:** A dedicated prompt construction function that takes `sceneLabels` from Firestore and produces an inpainting prompt. Keeps prompt logic centralized and testable.

**Trade-offs:**
- Pro: Unit testable without GPU
- Pro: Easy to iterate on prompt phrasing without touching inference code
- Con: Adds indirection between label data and inference call

**Example:**

```python
# app/services/prompt_builder.py
SURFACE_PROMPTS = {
    "grass": "natural grass, ground, lawn, seamless grass texture",
    "concrete": "smooth concrete pavement, ground surface",
    "wood": "wooden floor, wood grain texture, natural wood",
    "sand": "sandy ground, beach sand texture",
    "leaves": "fallen leaves on ground, autumn leaves, natural ground",
    "gravel": "gravel path, small stones on ground",
    "tile": "tile floor, smooth tiles, seamless tile pattern",
    "water": "clear water, water surface, natural water",
}

FALLBACK_PROMPT = "empty ground, nothing here, just the natural ground surface continuing seamlessly"

def build_inpaint_prompt(scene_labels: dict | None) -> str:
    if not scene_labels:
        return FALLBACK_PROMPT
    surface = scene_labels.get("surface", "ground")
    detail = scene_labels.get("detail", "")
    base = SURFACE_PROMPTS.get(surface, FALLBACK_PROMPT)
    if detail:
        return f"{detail}, {base}"
    return base
```

### Pattern 5: Inference Optimization — torch.compile on Transformer Only

**What:** Apply `torch.compile` to the FLUX transformer (DiT) which dominates inference compute. Skip compiling text encoders and VAE, which are cheaper and benefit less.

**When to use:** When the model runs repeated inference calls (production worker) and the startup compilation cost (one-time) is acceptable.

**Trade-offs:**
- Pro: 15-53% documented speedup on Flux.1-Dev (H100 benchmarks); benefit on T4 is lower but present for int8/int4 quantized models
- Pro: Integrates with torchao quantization — `int8wo` + `torch.compile` is the recommended pairing for non-FP8 GPUs
- Con: T4 has compute capability 7.5 (Turing) — FP8 quantization not supported; limited to int4/int8
- Con: First inference call triggers compilation (5-60s warmup); Cloud Run startup probe must account for this
- Con: `dynamic=True` is required if image sizes vary (avoids recompilation per resolution)
- Con: GGUF quantized weights loaded via `from_single_file` may not be compatible with `torch.compile` — requires switching to torchao int8 quantization as the optimization path

**Implementation (replacing GGUF with torchao):**

```python
import torch
from diffusers import FluxFillPipeline, AutoModel
from torchao.quantization import Int8WeightOnlyConfig
from diffusers import TorchAoConfig

def load_model():
    quantization_config = TorchAoConfig(Int8WeightOnlyConfig())
    transformer = AutoModel.from_pretrained(
        "black-forest-labs/FLUX.1-Fill-dev",
        subfolder="transformer",
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
    )
    pipe = FluxFillPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Fill-dev",
        transformer=transformer,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    # Compile only the transformer (DiT) — not text encoders or VAE
    pipe.transformer = torch.compile(
        pipe.transformer,
        mode="reduce-overhead",
        dynamic=True,  # Avoid recompile on different image sizes
    )
    return pipe
```

**Note on GGUF vs torchao:** The current codebase uses Q4/Q8 GGUF quantization loaded via `from_single_file`. This path is incompatible with `torch.compile`. To unlock compile-based optimization, switch to torchao `int8wo` (int8 weight-only quantization) which is natively supported and compatible with `torch.compile`. The memory footprint difference is manageable on a 16GB T4. This is a build-order dependency: inference optimization requires the torchao migration as a prerequisite.

---

## Build Order (Dependency Graph)

Components must be built in this order due to dependencies:

```
Phase 1: SAM3 Extension (prerequisite for everything else)
  ├── Extend SAM3 service to run surface concept prompts and return labels
  ├── Write sceneLabels to Firestore session doc
  └── Validate label schema (surface, detail, confidence, labeledAt)

Phase 2: API Server Fan-out (requires SAM3 to exist with an endpoint)
  ├── Add fire-and-forget SAM3 trigger in parallel with Pub/Sub enqueue
  └── Handle SAM3 call failure gracefully (log + continue)

Phase 3: Inpaint Service Label Integration (requires Phases 1+2 to be testable)
  ├── Read sceneLabels from Firestore in pipeline.py
  ├── Implement prompt_builder.py with surface-to-prompt mapping
  ├── Implement label-wait requeue logic (500 NACK with attempt guard)
  └── Implement fallback generic prompt when labels never arrive

Phase 4: Model Registry (independent of Phase 1-3, parallel track possible)
  ├── Create app/services/model_registry.py with protocol + loader
  ├── Refactor flux_inpaint.py into app/services/models/flux_fill.py
  ├── Add MODEL_VARIANT env var support
  └── Document Cloud Run traffic split procedure

Phase 5: Inference Optimization (requires Phase 4 model registry structure)
  ├── Evaluate: GGUF path vs torchao int8wo path (benchmark on T4)
  ├── If torchao: migrate from GGUF loader to torchao quantization config
  ├── Apply torch.compile to transformer only (not full pipeline)
  ├── Update Dockerfile — model bake step changes with torchao
  ├── Update readiness probe — account for compile warmup time
  └── Benchmark: steps reduction (28→20) vs quality tradeoff

Phase 6: Mask Cleanup (independent, can parallel-track with any phase)
  └── Add morphological erosion/dilation step in image.py before resize
```

**Critical dependencies:**
- Phase 3 (label integration) cannot be fully tested without Phase 1 (SAM3 writing labels) and Phase 2 (trigger mechanism)
- Phase 5 (torch.compile) requires deciding on GGUF vs torchao first — this is a breaking change to model loading; do not attempt torch.compile while still using GGUF
- Phase 4 (model registry) should precede Phase 5 so optimization work has a clean structure to land in

---

## Recommended Project Structure

```
app/
├── main.py                          # FastAPI app, lifespan, health probes
├── routers/
│   └── inpaint.py                   # Pub/Sub push handler (existing)
├── services/
│   ├── firebase.py                  # Firebase Storage + Firestore client (existing)
│   ├── pipeline.py                  # Orchestration: download → label check → inpaint → upload
│   ├── prompt_builder.py            # NEW: scene labels → inpaint prompt
│   ├── model_registry.py            # NEW: MODEL_VARIANT env var → model loader
│   └── models/
│       ├── __init__.py
│       ├── flux_fill.py             # NEW: FLUX.1-Fill-dev model (refactored from flux_inpaint.py)
│       └── sdxl_inpaint.py          # NEW (Phase 4+): SDXL alternative if evaluated
└── utils/
    └── image.py                     # Image/mask decode, resize, encode, mask cleanup (existing + cleanup)
```

---

## Scalability Considerations

| Concern | Current (single T4) | If throughput grows |
|---------|---------------------|---------------------|
| Label contention | One write per session, no contention | N/A — Firestore handles well |
| Pub/Sub requeue storms | Unlikely; SAM3 is fast (2-5s), backoff handles timing | Add Cloud Tasks if precise delay control needed |
| Model cold start | 60-90s FLUX load; readiness probe gates this | Pre-warming via min-instances=1 in Cloud Run |
| torch.compile warmup | 30-60s on first inference; subsequent calls fast | Must fire a warmup inference after model load |
| Multiple model variants | Cannot co-load on T4 (16GB VRAM) | Separate Cloud Run services per model |

---

## Anti-Patterns to Avoid

### Anti-Pattern 1: Synchronous SAM3 Wait in API Server

**What people do:** `await sam3_client.label(image)` before enqueuing Pub/Sub, blocking the API response.

**Why it's wrong:** Adds 2-5s to every user request. SAM3 GPU contention degrades this further. If SAM3 is slow or down, user-facing latency explodes.

**Do this instead:** Fire-and-forget trigger at the same time as Pub/Sub enqueue. Let the inpaint worker handle the race condition with graceful requeue.

### Anti-Pattern 2: Polling Loop in Inpaint Worker

**What people do:** `while not labels_ready: time.sleep(5)` inside the inpaint handler thread, waiting for SAM3 to write.

**Why it's wrong:** Holds the Cloud Run instance occupied with a sleeping thread. Consumes GPU instance time that costs money. Blocks the event loop from servicing health probes.

**Do this instead:** Return HTTP 500 immediately when labels are absent. Pub/Sub handles the retry with backoff. The instance is freed to serve other requests (or health probes) while waiting.

### Anti-Pattern 3: torch.compile on Full Pipeline

**What people do:** `pipe = torch.compile(pipe)` — compiling the entire diffusion pipeline including text encoders and VAE.

**Why it's wrong:** Text encoders run once per inference call and are not the bottleneck. Compiling them adds compilation time with negligible speedup. VAE decode is fast. The DiT transformer is 90%+ of inference time.

**Do this instead:** `pipe.transformer = torch.compile(pipe.transformer, ...)` — compile only the transformer. Verify with `dynamic=True` to avoid recompilation on variable image sizes.

### Anti-Pattern 4: Storing Labels in Pub/Sub Message Payload

**What people do:** Include `sceneLabels` JSON in the Pub/Sub message data field so the inpaint worker receives them directly.

**Why it's wrong:** Requires SAM3 to complete before the message is published, negating the async benefit. Alternatively, requires a separate delayed message publish from SAM3 (complex, new infrastructure). Pub/Sub message size limit is 10MB but architectural coupling is the real cost.

**Do this instead:** Firestore is the shared state store — both SAM3 and the inpaint worker already have Firebase access. Labels written to Firestore are available on the worker's existing session doc read with zero protocol change.

### Anti-Pattern 5: GGUF + torch.compile Together

**What people do:** Keep existing GGUF quantization loader (`from_single_file` with `GGUFQuantizationConfig`) and add `torch.compile` on top.

**Why it's wrong:** GGUF quantization via `from_single_file` produces non-standard weight tensors that are incompatible with torch.compile graph tracing. Graph breaks occur at quantized weight accesses, producing no speedup or errors.

**Do this instead:** If optimization with torch.compile is the goal, migrate to torchao `int8wo` or `int4wo` quantization which is natively designed to compose with torch.compile. Benchmark torchao int8wo vs GGUF Q8 on the T4 first — if memory fits, torchao + compile wins on throughput.

---

## Integration Points

### External Services

| Service | Integration Pattern | Notes |
|---------|---------------------|-------|
| SAM3 Service | HTTP POST from API server (fire-and-forget, 2s timeout) | SAM3 must expose a stable endpoint; failure must be silent to inpaint path |
| Google Pub/Sub | Push subscription to `/inpaint` endpoint | Existing; backoff policy configurable on subscription (10s min recommended for label wait) |
| Firebase Firestore | Read session doc (labels + status), write updates | Existing; add `sceneLabels` field write from SAM3 side |
| Firebase Storage | Download image/mask, upload results | Existing; no changes needed |
| Hugging Face Hub | Model weights at Docker build time | torchao path changes download command (full model vs GGUF file) |

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| `pipeline.py` ↔ `prompt_builder.py` | Direct function call | `build_inpaint_prompt(scene_labels)` → string |
| `pipeline.py` ↔ `model_registry.py` | Direct function call at startup | Model loaded once; `get_active_model()` returns cached instance |
| `pipeline.py` ↔ `firebase.py` | Direct function call | Existing; add `get_scene_labels(user_id, session_id)` helper |
| `inpaint.py` router ↔ `pipeline.py` | Direct function call via `asyncio.to_thread` | Existing pattern; label-not-ready raises exception → caller returns 500 |

---

## Sources

- [torch.compile + Diffusers guide (PyTorch Blog, 2025)](https://pytorch.org/blog/torch-compile-and-diffusers-a-hands-on-guide-to-peak-performance/) — MEDIUM confidence (H100 benchmarks, T4 data extrapolated)
- [torchao quantization in Diffusers (HuggingFace docs)](https://huggingface.co/docs/diffusers/en/quantization/torchao) — HIGH confidence (official docs)
- [Choose Cloud Tasks or Pub/Sub (Google Cloud docs)](https://docs.cloud.google.com/tasks/docs/comp-pub-sub) — HIGH confidence (official docs)
- [Pub/Sub retry policy (Google Cloud docs)](https://cloud.google.com/pubsub/docs/subscription-retry-policy) — HIGH confidence (official docs)
- [SAM3 paper: Segment Anything with Concepts (arXiv 2511.16719)](https://arxiv.org/html/2511.16719v1) — HIGH confidence (primary source)
- [SAM3 GitHub: facebookresearch/sam3](https://github.com/facebookresearch/sam3) — HIGH confidence (official repo)
- [Firestore data structure best practices (Firebase docs)](https://firebase.google.com/docs/firestore/manage-data/structure-data) — HIGH confidence (official docs)
- [SDXL vs FLUX speed comparison (Stable Diffusion Art)](https://stable-diffusion-art.com/sdxl-vs-flux/) — LOW confidence (community benchmark, 4090 hardware, not T4)
- [diffusers-torchao end-to-end recipes (sayakpaul/diffusers-torchao)](https://github.com/sayakpaul/diffusers-torchao) — MEDIUM confidence (community maintained, widely referenced)

---

*Architecture research for: Pawfect Edit Inpaint — async scene labeling + optimized inference pipeline*
*Researched: 2026-02-21*
