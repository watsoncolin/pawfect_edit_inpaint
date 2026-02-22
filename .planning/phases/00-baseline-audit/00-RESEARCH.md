# Phase 0: Baseline Audit - Research

**Researched:** 2026-02-21
**Domain:** FLUX.1-Fill-dev inference diagnostics on Cloud Run GPU — quantization comparison, guidance scale calibration, torch.compile viability, audit infrastructure
**Confidence:** HIGH (critical GPU identity finding verified; pipeline code read directly; official docs consulted)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Test methodology:**
- Build a single "audit mode" deployment with a dedicated endpoint that cycles through all parameter variations
- Each variation produces an output image — no automated metrics, visual side-by-side comparison only
- Test on production Cloud Run GPU hardware for real timing numbers
- Guidance scale grid: test values 2, 4, 10, 20, 30 (covers full range researchers disagreed on)
- Quantization: compare Q4_0 vs Q5_K_M vs Q8_0 (availability permitting)
- torch.compile: verify compilation actually runs via torch._dynamo.explain(), not assumed

**Reference sessions:**
- Test on session D1A406F5 only (dog on grass, leash handle removal)
- Expected correct output: matching grass with fallen leaves where hand/leash handle was — nothing else
- Single session is sufficient for the diagnostic audit

**Quality thresholds:**
- Both hallucinated objects AND color/texture mismatch are failures
- Subtle seams at mask boundary are acceptable — doesn't need to be pixel-perfect invisible
- User eyeballs all outputs and picks the best-looking overall — no formal scoring

**Audit deliverable:**
- All test output images uploaded to Firebase Storage at a known path
- Markdown report with signed URLs to each image + timing data for each variation
- User reviews report and manually approves which settings to use in Phase 1
- No auto-deployment — audit is purely diagnostic

### Claude's Discretion
- Audit endpoint API design and payload structure
- How to organize test images in Firebase Storage
- Report format and structure details
- Whether to test step count variations alongside guidance scale, or keep that for Phase 1

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| BASE-01 | Establish baseline metrics — run current pipeline on reference session (D1A406F5) and record inference time, output quality | Existing `pipeline.py` already times inference with `time.time()`. Baseline = run current Q4_0 config (28 steps, guidance 10) on D1A406F5 and record elapsed seconds + output image. |
| BASE-02 | Compare Q4_0 vs Q5_K_M vs Q8_0 quantization on reference session for quality and speed | Q5_K_M is NOT available in YarvixPA repo — closest is Q5_K_S (8.29 GB). Q4_0 = 6.81 GB, Q8_0 = 12.7 GB. All three load via `FluxTransformer2DModel.from_single_file()` with `GGUFQuantizationConfig`. Loading is the expensive part — audit runs one quant per invocation (can't hot-swap). |
| BASE-03 | Run guidance scale calibration grid (test values 2, 4, 10, 20, 30) on reference session | `guidance_scale` is a simple parameter to `FluxFillPipeline.__call__()`. Model card default = 30, diffusers API default = 3.5. Community disagrees — critical to measure empirically. Steps and quant can be held constant during this test. |
| BASE-04 | Verify torch.compile viability on T4 (sm_75) — confirm compilation actually runs, not silent fallback | **CRITICAL FINDING: Cloud Run does not offer T4 GPU.** Production hardware is NVIDIA L4 (sm_89, Ada Lovelace). This changes viability significantly — L4 fully supports Flash Attention 2 (requires sm_80+), FP8, and Triton. torch._dynamo.explain() should be called on a representative FLUX transformer forward pass to count graph breaks and confirm Triton compilation actually fires. |
</phase_requirements>

---

## Summary

Phase 0 is a pure diagnostic phase: run the existing production pipeline against a known bad session (D1A406F5) across a grid of parameters, collect output images and timing data, and produce a human-readable report. No model changes are deployed to production as a result.

**The single most important pre-research finding:** the project's REQUIREMENTS.md and STATE.md reference "T4 (sm_75)" but Cloud Run GPU services only offer the **NVIDIA L4 (sm_89, Ada Lovelace)** and RTX PRO 6000. T4 is not available on Cloud Run at all. This changes the entire constraint surface: Flash Attention 2 (requires sm_80+) is viable on L4, FP8 quantization (listed as "out of scope, requires sm_89+") is viable on L4, and torch.compile / Triton work without the workarounds that T4 required. The audit must run on this actual hardware, so BASE-04 should verify L4-specific capabilities, not T4.

There is also a real discrepancy in the codebase: the `Dockerfile` bakes Q8_0 weights at build time, but `flux_inpaint.py` loads Q4_0 from a remote HuggingFace URL. These two things conflict and must be reconciled before the audit can run reliably. The audit endpoint needs to load each quantization from cached local weights (or from HF cache) rather than fetching at runtime.

The guidance scale conflict is genuine and unresolved: the FLUX.1-Fill-dev model card example uses `guidance_scale=30`; the diffusers API default is `3.5`; community reporting for fill/background tasks suggests lower values (2–5) produce more naturalistic backgrounds. The calibration grid (2, 4, 10, 20, 30) directly addresses this. There is no authoritative source that distinguishes background-fill optimal guidance from object-generation optimal guidance — empirical testing is the right call.

**Primary recommendation:** Build the audit as a single new FastAPI endpoint (`/audit`) that accepts a session reference, iterates through all parameter variations sequentially (one model load per quantization group, reuse pipeline for guidance/steps variations within that group), uploads output images to Firebase Storage under `audit/{run_id}/`, and returns a structured JSON report with signed URLs and timing data. Write the Markdown report as a separate artifact from the JSON.

---

## Standard Stack

### Core (already in codebase)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| diffusers | >=0.31.0 | FluxFillPipeline, FluxTransformer2DModel, GGUFQuantizationConfig | Official HuggingFace inference library for FLUX |
| torch | >=2.1.0 (cu124) | Tensor ops, cuda events for timing, dynamo | Required by diffusers |
| firebase-admin | >=6.6.0 | Download session assets, upload audit outputs, generate signed URLs | Already used in production |
| fastapi | >=0.110.0 | Audit endpoint host | Already used in production |
| Pillow | >=10.0.0 | Image handling | Already used |

### Supporting (audit-specific additions)
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| google-cloud-storage | bundled with firebase-admin | `blob.generate_signed_url()` for report URLs | Required: firebase_admin storage client is a google.cloud.storage.Bucket; signed URL generation uses service account credentials already in FIREBASE_CREDENTIALS env var |
| datetime | stdlib | Signed URL expiration | trivial |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Signed URLs (expiring) | Public download URLs | Signed URLs are appropriate for private audit outputs; public URLs require bucket policy changes |
| Sequential quant loading | Parallel model loads | Parallel is impossible on single GPU — must load one model at a time |

**No new dependencies required for the audit endpoint.** `google-cloud-storage` is a transitive dependency of `firebase-admin`.

---

## Architecture Patterns

### Recommended Audit Structure

```
app/
├── routers/
│   ├── inpaint.py          # existing (don't touch)
│   └── audit.py            # NEW: /audit endpoint
├── services/
│   ├── flux_inpaint.py     # existing (minimal changes for param overrides)
│   ├── pipeline.py         # existing (don't touch)
│   ├── firebase.py         # existing (add generate_signed_url helper)
│   └── audit_runner.py     # NEW: orchestrates the test matrix
└── main.py                 # add audit router import
```

### Pattern 1: Audit Endpoint as a Trigger

**What:** POST `/audit` receives `{user_id, session_id, run_id}`, runs the full matrix sequentially in a background thread (same pattern as existing inpaint), returns `202 Accepted` immediately with the `run_id`. A second GET `/audit/{run_id}/status` returns progress and final report path.

**When to use:** When audit takes 15-60 minutes — a synchronous endpoint would time out. Matches the existing pattern of `asyncio.to_thread(run_inpaint, ...)`.

**Alternative:** Simple synchronous endpoint with a long timeout (`timeout=3600`) — acceptable for a one-off diagnostic tool where the user will wait and watch logs.

**Recommendation (Claude's discretion):** Use synchronous long-timeout for simplicity. This is a one-time audit tool, not production API. Set Cloud Run `--timeout=3600` (already at 300s — may need increase for full matrix).

### Pattern 2: Parameter Matrix Execution

**What:** Outer loop over quantization variants (requires model reload each time). Inner loop over guidance scale values (reuses loaded pipeline, just passes different `guidance_scale` to `pipe()`). Steps variation (if tested) is another inner loop.

```python
# Source: architecture derived from existing flux_inpaint.py pattern
QUANT_VARIANTS = [
    ("Q4_0", "https://huggingface.co/YarvixPA/FLUX.1-Fill-dev-GGUF/blob/main/flux1-fill-dev-Q4_0.gguf"),
    ("Q5_K_S", "https://huggingface.co/YarvixPA/FLUX.1-Fill-dev-GGUF/blob/main/flux1-fill-dev-Q5_K_S.gguf"),
    ("Q8_0", "https://huggingface.co/YarvixPA/FLUX.1-Fill-dev-GGUF/blob/main/flux1-fill-dev-Q8_0.gguf"),
]
GUIDANCE_SCALE_GRID = [2, 4, 10, 20, 30]

for quant_name, gguf_url in QUANT_VARIANTS:
    pipe = load_flux_fill_gguf(gguf_url)  # expensive: ~2-5 min
    for gs in GUIDANCE_SCALE_GRID:
        start = time.time()
        result = pipe(prompt=PROMPT, image=image, mask_image=mask,
                      num_inference_steps=NUM_STEPS, guidance_scale=gs)
        elapsed = time.time() - start
        record_result(quant_name, gs, elapsed, result.images[0])
    del pipe
    torch.cuda.empty_cache()
```

**Critical:** `del pipe` + `torch.cuda.empty_cache()` between quantization variants to free VRAM before loading next model. L4 has 24 GB VRAM; Q8_0 transformer is 12.7 GB, leaving margin for pipeline components but not for two transformers simultaneously.

### Pattern 3: Timing with CUDA Events (Accurate GPU Timing)

**What:** `time.time()` measures wall-clock including CPU scheduling. For GPU inference, use CUDA events for accurate kernel-level timing.

```python
# Source: PyTorch docs - torch_compile_tutorial
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
start_event.record()
result = pipe(...)
end_event.record()
torch.cuda.synchronize()
elapsed_ms = start_event.elapsed_time(end_event)
```

**When to use:** For the timing columns in the audit report. Wall-clock time is acceptable given the user wants "real timing numbers" — CUDA events give more precise GPU kernel time. Use both: CUDA events for GPU time, `time.time()` for wall-clock.

### Pattern 4: GGUF Quantization Loading (Verified from diffusers docs)

```python
# Source: https://huggingface.co/docs/diffusers/en/quantization/gguf
from diffusers import FluxFillPipeline, FluxTransformer2DModel, GGUFQuantizationConfig

gguf_path = "https://huggingface.co/YarvixPA/FLUX.1-Fill-dev-GGUF/blob/main/flux1-fill-dev-Q8_0.gguf"
transformer = FluxTransformer2DModel.from_single_file(
    gguf_path,
    quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
    torch_dtype=torch.bfloat16,
)
pipe = FluxFillPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Fill-dev",
    transformer=transformer,
    torch_dtype=torch.bfloat16,
)
pipe.enable_model_cpu_offload()
```

**Note:** Pipeline components (VAE, CLIP, T5 encoder, scheduler) are loaded once from `from_pretrained` cache. Only the transformer changes between quantization variants. Advanced optimization: load pipeline components separately once, swap transformers only. Out of scope for audit but worth noting for Phase 1.

### Pattern 5: torch._dynamo.explain() for Compile Viability

```python
# Source: PyTorch docs - torch.compiler_dynamo_overview
import torch._dynamo as dynamo

# Wrap a representative forward pass as a function
def forward_fn():
    return pipe.transformer(hidden_states, ...)  # representative call

explanation = dynamo.explain(forward_fn)()
print(f"Graph count: {explanation.graph_count}")
print(f"Graph break count: {explanation.graph_break_count}")
for reason in explanation.break_reasons:
    print(f"Break reason: {reason}")
```

**Important nuance:** `torch._dynamo.explain()` tests graph tracing, not actual Triton compilation. A zero-break-count result means compilation is possible; it does NOT mean it actually ran. To confirm Triton compilation fires, set `TORCH_LOGS="output_code"` and look for generated Triton kernel source in logs.

**For the audit:** Run `dynamo.explain()` on a single pipe forward pass. Log the break count and reasons. Also attempt `torch.compile(pipe.transformer, backend="inductor")` on a test forward pass and confirm no errors. Time the compiled vs. uncompiled forward pass.

### Pattern 6: Firebase Storage Signed URLs

```python
# Source: google-cloud-storage docs (via firebase-admin)
import datetime

def generate_signed_url(blob_path: str, expiry_hours: int = 72) -> str:
    bucket = storage.bucket()
    blob = bucket.blob(blob_path)
    url = blob.generate_signed_url(
        version="v4",
        expiration=datetime.timedelta(hours=expiry_hours),
        method="GET",
    )
    return url
```

**Prerequisite:** Signed URL generation requires the service account credentials to have the `iam.serviceAccounts.signBlob` permission. The FIREBASE_CREDENTIALS service account used in production almost certainly has this (it has Firebase Admin roles). If not, calls will fail with a permission error — this should be tested early.

### Anti-Patterns to Avoid

- **Loading a new pipeline for every guidance scale value:** Pipeline components (VAE, T5, CLIP) reload unnecessarily. Reuse the pipeline within a quantization group.
- **Using `time.time()` alone for GPU timing:** Does not account for async kernel scheduling. Always call `torch.cuda.synchronize()` before stopping the timer.
- **Assuming Q5_K_M exists in YarvixPA repo:** It does not. Available 5-bit option is Q5_K_S. The CONTEXT.md says "availability permitting" — Q5_K_S is the substitute.
- **Hot-swapping transformer modules in a live pipeline:** Memory fragmentation risk. `del pipe; torch.cuda.empty_cache()` then construct fresh.
- **Running audit on the production inpaint deployment without isolation:** The audit endpoint changes global model state. Run on a dedicated audit deployment or add a lock/ready-check guard.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Quantized model loading | Custom GGUF parser | `FluxTransformer2DModel.from_single_file()` with `GGUFQuantizationConfig` | Already supports Q4_0, Q5_K_S, Q8_0 and all k-quants |
| Signed URL generation | Custom GCS auth flow | `blob.generate_signed_url(version="v4")` | Handles V4 signature, expiry, auth from existing service account |
| GPU timing | `datetime.now()` diffs | `torch.cuda.Event(enable_timing=True)` + `torch.cuda.synchronize()` | Accounts for async GPU scheduling |
| Compile graph analysis | Manual bytecode inspection | `torch._dynamo.explain()` | Built-in Dynamo introspection tool |
| Image grid report | Custom HTML builder | Markdown with image links (signed URLs) | User explicitly chose Markdown report format |

---

## Common Pitfalls

### Pitfall 1: GPU Identity Mismatch (CRITICAL)
**What goes wrong:** All planning documents reference "T4 (sm_75)" but Cloud Run GPU is L4 (sm_89). Tests calibrated around T4 constraints (no Flash Attention 2, no FP8) may be overly conservative.
**Why it happens:** The REQUIREMENTS.md was written assuming T4 (possibly from Colab testing or older GCP knowledge), but Cloud Run only offers L4.
**How to avoid:** Confirm actual GPU with `torch.cuda.get_device_name(0)` and `torch.cuda.get_device_capability()` at the start of the audit. Log this in the report. If L4 confirmed, note that FP8 exclusion and Flash Attention 2 restrictions in requirements are based on incorrect hardware assumptions.
**Warning signs:** `nvidia-smi` output or CUDA logs showing "NVIDIA L4" — expected if running on Cloud Run.

### Pitfall 2: Dockerfile/Code Quantization Mismatch
**What goes wrong:** `Dockerfile` bakes `flux1-fill-dev-Q8_0.gguf` at build time into the image cache. `flux_inpaint.py` loads `flux1-fill-dev-Q4_0.gguf` from a remote URL at runtime (line 15: `GGUF_URL`). These conflict — the Q8_0 is cached but never used; Q4_0 is fetched from the internet on every cold start.
**Why it happens:** Dockerfile was updated (comment even says "Download GGUF Q8 transformer") but `flux_inpaint.py` was not updated to use the cached path.
**How to avoid:** The audit runner should load GGUF files from the HuggingFace cache (which is populated by the Dockerfile's bake step), not from remote URLs. Use `hf_hub_download("YarvixPA/FLUX.1-Fill-dev-GGUF", filename="flux1-fill-dev-Q8_0.gguf")` to get the local cache path, then pass the local path to `from_single_file()`. For Q4_0 and Q5_K_S, these must be downloaded — the audit deployment needs to fetch them or they must be baked into the audit image.
**Warning signs:** Long model load times on cold start (network download) vs. fast load (local cache hit).

### Pitfall 3: Cloud Run Request Timeout During Long Audit
**What goes wrong:** Full audit matrix (3 quants × 5 guidance values = 15 inferences + 3 model loads) may take 45-90 minutes. Cloud Run default timeout is 300 seconds (5 minutes). Even with `--timeout=3600`, a single HTTP request cannot exceed Cloud Run's 60-minute max.
**Why it happens:** Audit is designed as a single run producing all outputs, but Cloud Run is designed for short-lived request handlers.
**How to avoid:** Two options: (a) Use Cloud Run Jobs instead of Services for the audit — Jobs support arbitrary runtime with no HTTP timeout. (b) Use an async pattern: POST `/audit` enqueues work and returns immediately; use Cloud Run's existing Pub/Sub pattern to process. Option (a) is simpler for a one-off diagnostic.
**Warning signs:** `DeadlineExceeded` error in Cloud Run logs before audit completes.

### Pitfall 4: VRAM OOM Between Model Loads
**What goes wrong:** After inference with Q4_0, PyTorch may not fully release VRAM before loading Q8_0 (12.7 GB). With 24 GB VRAM on L4, Q8_0 transformer alone uses >50% — leftover Q4_0 allocations cause OOM.
**Why it happens:** Python garbage collection doesn't immediately free GPU memory. `del pipe` alone is insufficient.
**How to avoid:** After each quantization group: `del transformer; del pipe; gc.collect(); torch.cuda.empty_cache()`. Import `gc` explicitly.
**Warning signs:** CUDA OOM error when loading the second or third quantization variant.

### Pitfall 5: Signed URL Permission Failure
**What goes wrong:** `blob.generate_signed_url()` raises `google.auth.exceptions.TransportError` or a 403 error about `iam.serviceAccounts.signBlob`.
**Why it happens:** The service account used for Firebase Admin (from FIREBASE_CREDENTIALS) may not have the `Service Account Token Creator` IAM role required for V4 signed URL generation.
**How to avoid:** Test signed URL generation early in the audit (before running the matrix). If it fails, fall back to making the audit output paths public-read blobs (simpler for a one-off diagnostic tool) or use `blob.public_url` after setting `blob.make_public()`.
**Warning signs:** Permission error on first `generate_signed_url()` call.

### Pitfall 6: guidance_scale=30 Produces Artifacts
**What goes wrong:** The model card example uses `guidance_scale=30`, which is the value the existing code uses (10, actually). Community reports suggest high guidance scale with short prompts causes over-saturation, unnatural colors, or hallucinated texture patterns — which may be contributing to the observed hallucinations.
**Why it happens:** FLUX.1-Fill-dev was trained with guidance distillation — the optimal guidance range for this specific model is empirically unclear. The current production value of 10 is between the two camps (official: 30, community fill-mode: 2-5).
**How to avoid:** This is exactly what BASE-03 tests. Run the full grid including 2, 4, 10, 20, 30. No special avoidance needed — the pitfall is assuming guidance 30 is correct before running the test.

---

## Code Examples

### Baseline Timing Pattern (already in codebase, extend with CUDA events)
```python
# Source: app/services/pipeline.py (existing) + torch CUDA event extension
import time
import torch

# Wall-clock timing (existing pattern)
start = time.time()
result = inpaint(image, mask)
elapsed = time.time() - start

# CUDA event timing (accurate GPU time)
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
start_event.record()
result = pipe(prompt=PROMPT, image=image, mask_image=mask,
              num_inference_steps=NUM_STEPS, guidance_scale=gs)
end_event.record()
torch.cuda.synchronize()
gpu_elapsed_ms = start_event.elapsed_time(end_event)
```

### Loading Cached GGUF from HuggingFace Hub
```python
# Source: huggingface_hub pattern (hf_hub_download returns local path)
from huggingface_hub import hf_hub_download

# Returns local cache path (fast if already cached, downloads if not)
local_path = hf_hub_download(
    repo_id="YarvixPA/FLUX.1-Fill-dev-GGUF",
    filename="flux1-fill-dev-Q8_0.gguf",
)
transformer = FluxTransformer2DModel.from_single_file(
    local_path,
    quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
    torch_dtype=torch.bfloat16,
)
```

### Confirm GPU Identity at Audit Start
```python
# Source: torch.cuda API
import torch

gpu_name = torch.cuda.get_device_name(0)
major, minor = torch.cuda.get_device_capability(0)
print(f"GPU: {gpu_name}, compute capability: {major}.{minor} (sm_{major}{minor})")
# Expected on Cloud Run: "NVIDIA L4", 8.9 (sm_89)
```

### VRAM Cleanup Between Quantization Variants
```python
# Source: PyTorch best practices
import gc

del transformer
del pipe
gc.collect()
torch.cuda.empty_cache()
free_mem = torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)
print(f"VRAM freed. Reserved: {torch.cuda.memory_reserved(0)/1e9:.1f} GB")
```

### torch._dynamo.explain() Compile Check
```python
# Source: PyTorch dynamo docs (torch.compiler_dynamo_overview)
import torch._dynamo as dynamo

def test_forward():
    with torch.no_grad():
        # Run a minimal forward pass through the transformer
        return pipe.transformer(hidden_states, ...)

dynamo.reset()
explanation = dynamo.explain(test_forward)()
print(f"Graphs: {explanation.graph_count}, Breaks: {explanation.graph_break_count}")
if explanation.graph_break_count == 0:
    print("torch.compile viable — no graph breaks detected")
else:
    for i, reason in enumerate(explanation.break_reasons):
        print(f"Break {i}: {reason}")
```

### Firebase Storage Signed URL
```python
# Source: google-cloud-storage docs
import datetime
from firebase_admin import storage

def upload_and_sign(path: str, data: bytes, content_type: str, expiry_hours: int = 72) -> str:
    bucket = storage.bucket()
    blob = bucket.blob(path)
    blob.upload_from_string(data, content_type=content_type)
    url = blob.generate_signed_url(
        version="v4",
        expiration=datetime.timedelta(hours=expiry_hours),
        method="GET",
    )
    return url
```

### Audit Report Structure (Markdown)
```markdown
# Baseline Audit Report — {run_id}
**Session:** D1A406F5 | **Date:** {date} | **GPU:** {gpu_name} (sm_{major}{minor})

## GPU Info
- Device: {gpu_name}
- Compute Capability: {major}.{minor}
- VRAM: {total_vram} GB

## Results Matrix

| Quant | Guidance | Steps | Wall-clock (s) | GPU time (ms) | Image |
|-------|----------|-------|---------------|--------------|-------|
| Q4_0  | 10       | 28    | 45.2          | 44100        | [view](signed_url) |
| Q4_0  | 2        | 28    | 45.1          | 43950        | [view](signed_url) |
...

## torch.compile Viability
- Graph breaks detected: {n}
- Break reasons: {reasons}
- Recommendation: {proceed/skip}

## Recommended Settings for Phase 1
[User fills this in after reviewing images]
```

---

## State of the Art

| Old Assumption | Current Reality | Impact |
|----------------|-----------------|--------|
| T4 (sm_75) on Cloud Run | L4 (sm_89) on Cloud Run — T4 not offered | Flash Attention 2 viable; FP8 viable; GGUF CUDA kernels work (require >sm_70) |
| guidance_scale=10 (current) | Official default: 30; community fill-mode: 2-5; diffusers default: 3.5 | Current value is in a "no-man's-land" — neither camp endorses 10 |
| Q4_0 loaded from URL at runtime | Dockerfile caches Q8_0 at build, but code loads Q4_0 from network | Code/image mismatch must be fixed for reproducible timing |
| Q5_K_M assumed available | Only Q5_K_S available in YarvixPA repo (Q5_K_M does not exist there) | Swap Q5_K_M → Q5_K_S in test matrix |
| torch.compile assumed risky on T4 | On L4 (sm_89): Triton supports ≥sm_70, inductor fully supported, Flash Attention 2 supported | torch.compile is low-risk to test on L4 |

**Deprecated/outdated assumptions:**
- "FP8 out of scope, requires sm_89+" — L4 IS sm_89, so FP8 is no longer out of scope if quality/speed warrants it. Audit should note this even if Phase 1 doesn't use it.
- "Flash Attention requires sm_80+" confirmed true — and L4 at sm_89 clears this bar.

---

## Open Questions

1. **Is Q8_0 actually faster than Q4_0 on L4?**
   - What we know: Q8_0 is 12.7 GB vs Q4_0 at 6.81 GB. Larger quant = fewer dequantization ops but more memory bandwidth. On T4 (limited bandwidth) Q8_0 might be slower. On L4 (higher bandwidth) Q8_0 might be faster.
   - What's unclear: L4-specific benchmarks for FLUX GGUF inference not found.
   - Recommendation: This is exactly what BASE-02 measures. No assumption; measure it.

2. **Does the current prompt contribute to hallucinations independently of guidance scale?**
   - What we know: Current prompt is `"empty ground, nothing here, just the natural ground surface continuing seamlessly"`. Community reports suggest negative-style prompts ("nothing here") can confuse FLUX.
   - What's unclear: Whether prompt matters more or less than guidance scale for background fill.
   - Recommendation: Hold prompt constant during the audit (per CONTEXT.md). But flag for Phase 3 (QUAL-01 through QUAL-04) — the prompt itself may need replacement regardless of guidance scale findings.

3. **Can the audit run as a Cloud Run Service or does it need a Cloud Run Job?**
   - What we know: Full matrix (3 quants × 5 guidance + model loads) = estimated 60-90 minutes. Cloud Run Service max timeout = 3600s (60 min). Cloud Run Job has no HTTP timeout.
   - What's unclear: Whether 3 quants × 5 guidance values fits within 60 minutes on L4.
   - Recommendation: Design the audit endpoint to be called with extended timeout. If a single guidance-scale variation takes ~45s (current baseline estimate) then 15 inferences = ~11 minutes GPU time + model load overhead (3 × ~5-10 min) = ~26-41 minutes total. Should fit within 60 minutes. But add a `max_variations` parameter to allow partial runs.

4. **Should step count variations be included in Phase 0 or Phase 1?**
   - What we know: CONTEXT.md says this is at Claude's discretion. Current step count is 28 (production) vs. model card recommendation of 50.
   - What's unclear: How much step count variation (e.g., 20 vs 28 vs 50) matters relative to guidance scale.
   - Recommendation: Include a small step variation (20, 28, 50) on the best-performing quantization at the best guidance scale as a bonus test AFTER the main matrix. This avoids bloating the primary matrix while answering a key Phase 1 question.

---

## Sources

### Primary (HIGH confidence)
- Project source code read directly: `app/services/flux_inpaint.py`, `app/services/pipeline.py`, `app/services/firebase.py`, `app/routers/inpaint.py`, `app/utils/image.py`, `Dockerfile`, `pyproject.toml`, `cloudbuild.yaml`
- https://huggingface.co/docs/diffusers/en/quantization/gguf — GGUF loading pattern with `FluxTransformer2DModel.from_single_file()` and `GGUFQuantizationConfig`; confirmed Q4_0, Q5_K, Q8_0 supported
- https://huggingface.co/docs/diffusers/en/api/pipelines/flux — FluxFillPipeline API; `guidance_scale` default = 3.5, `num_inference_steps` default = 50
- https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev — Model card; official example uses `guidance_scale=30, num_inference_steps=50`
- https://huggingface.co/YarvixPA/FLUX.1-Fill-dev-GGUF — Confirmed available files: Q4_0 (6.81 GB), Q5_K_S (8.29 GB), Q8_0 (12.7 GB). **Q5_K_M does not exist in this repo.**
- https://docs.cloud.google.com/run/docs/configuring/services/gpu — Cloud Run offers NVIDIA L4 (24 GB VRAM) and RTX PRO 6000. **T4 is not available on Cloud Run.**
- https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html — torch compile tutorial; CUDA event timing pattern

### Secondary (MEDIUM confidence)
- https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/ — L4 confirmed as sm_89 (Ada Lovelace)
- WebSearch results: Triton minimum compute capability = 7.0; T4 (sm_75) and L4 (sm_89) both qualify for torch.compile + Triton
- WebSearch + github issue: Flash Attention 2 requires sm_80+; L4 at sm_89 is compatible; T4 at sm_75 is not
- https://huggingface.co/city96/FLUX.1-dev-gguf/discussions/15 — Visual quality ranking: Q8_0 > Q6_K > Q5_K_S > Q4_K_S > Q3_K_S. Q5_K_S is "best choice for people with low VRAM"; Q4_0 has "noticeable quality degradation" vs Q8_0

### Tertiary (LOW confidence)
- Community WebSearch: guidance scale 2-5 preferred for background-fill inpainting; guidance 30 may cause over-fitting to prompt and hallucinations. Not verified with official source — this is exactly why the calibration grid must run.

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — library versions verified from pyproject.toml and diffusers docs
- Architecture patterns: HIGH — code patterns verified from official diffusers GGUF docs and existing codebase
- GPU identity: HIGH — verified directly from Cloud Run documentation (L4, not T4)
- GGUF availability: HIGH — verified directly from YarvixPA HuggingFace model page
- Guidance scale analysis: MEDIUM — model card and diffusers defaults verified; community guidance-for-fill advice is LOW confidence
- torch.compile on L4: MEDIUM — Triton sm_70+ requirement verified; specific FLUX compile behavior on L4 not benchmarked

**Research date:** 2026-02-21
**Valid until:** 2026-03-23 (30 days — FLUX tooling stable but diffusers updates frequently)
