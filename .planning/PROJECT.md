# Pawfect Edit Inpaint — Speed & Quality Improvement

## What This Is

The inpainting service for Pawfect Edit, a pet photography app. It removes distracting objects (leash handles, hands, toys, clutter) from pet photos so the pet is the star. The service receives jobs via Pub/Sub, runs FLUX.1-Fill-dev inference on a T4 GPU, and uploads results to Firebase Storage. This project improves both inference speed (currently 60-120s, target <30s) and output quality (model hallucinates wrong objects instead of filling with matching background).

## Core Value

Inpainted areas must seamlessly match the surrounding background — users should not be able to tell something was removed.

## Requirements

### Validated

- ✓ Pub/Sub push-based job processing — existing
- ✓ Firebase Storage for image I/O — existing
- ✓ Firestore session state management (pending → processing → completed/failed) — existing
- ✓ FLUX.1-Fill-dev inpainting with Q4 GGUF quantization — existing
- ✓ Mask-based compositing to preserve unmasked pixels — existing
- ✓ Preview thumbnail generation — existing
- ✓ Graceful handling of deleted sessions — existing
- ✓ Max retry limit (5 attempts) for transient errors — existing
- ✓ Health and readiness probes — existing
- ✓ Model CPU offload for T4 memory constraints — existing

### Active

- [ ] Reduce end-to-end inpaint time from 60-120s to under 30s on T4 GPU
- [ ] Eliminate hallucinated objects in inpainted areas (wrong content generation)
- [ ] Context-aware prompting — dynamically describe the background surface to guide inpainting
- [ ] Integrate SAM3 scene labeling to provide background context before inpainting
- [ ] API server triggers SAM3 labeling in parallel with Pub/Sub enqueue
- [ ] Inpaint service checks for scene labels, requeues if not yet available
- [ ] Evaluate alternative inpainting models that may offer better speed/quality on T4
- [ ] Optimize inference parameters (steps, guidance scale) for background fill use case
- [ ] Clean up mask artifacts (scattered white specks) before inference

### Out of Scope

- GPU upgrade (T4 is the budget constraint) — optimize within T4
- Real-time/streaming inpainting — async batch processing is fine
- On-device inpainting improvements (small areas handled on iOS) — this project is server-side only
- SAM2 on-device mask generation changes — mask creation stays on device
- Pet segmentation or detection — pets stay in the photo

## Context

**Current architecture flow:**
iOS app → user selects area → SAM2 on-device creates mask → large areas trigger server inpaint → API server → Pub/Sub queue → inpaint service (this codebase)

**Proposed flow addition:**
API server receives inpaint request → triggers SAM3 scene labeling (separate service) in parallel → enqueues Pub/Sub message → inpaint service picks up message → checks if SAM3 labels are available → if yes, builds context-aware prompt and inpaints → if no, requeues message for short delay

**Key quality problem (session D1A406F5-651A-45CD-A76E-BA4D16B944D7):**
Photo of dog on patchy grass with fallen leaves. Mask covers hand holding retractable leash handle (bottom-left). Current generic prompt "empty ground, nothing here" causes model to hallucinate wrong content instead of filling with matching grass/leaves texture. The prompt needs to describe the actual surrounding surface.

**SAM3 service:**
Already exists as separate service for automatic leash detection. Currently returns binary masks only. Needs to be extended to also return scene/surface labels that the inpaint service can use for prompt construction.

**Infrastructure:**
- GPU: NVIDIA T4 (16GB VRAM) on Cloud Run
- Model: FLUX.1-Fill-dev with Q4 GGUF quantized transformer
- Current params: 28 inference steps, guidance_scale=10
- Model loading: baked into Docker image at build time

## Constraints

- **GPU**: T4 16GB VRAM — cannot upgrade, must optimize within this budget
- **Latency target**: End-to-end under 30 seconds (from current 60-120s)
- **Architecture**: Must remain async Pub/Sub worker pattern
- **SAM3 dependency**: Scene labeling is async — inpaint service must handle labels not being ready yet
- **Compatibility**: iOS app interface unchanged — changes are server-side only

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Use SAM3 for scene labeling | Already in infrastructure for leash detection, can extend | — Pending |
| Open to alternative inpainting models | Speed/quality tradeoff on T4 may favor smaller models | — Pending |
| Context-aware prompting over generic prompt | Current "empty ground" causes hallucinations | — Pending |
| Async label checking with requeue | Avoids blocking inpaint on SAM3 latency | — Pending |
| Accept 2-3s for scene classification | Better results justify small delay if total stays <30s | — Pending |

---
*Last updated: 2026-02-21 after initialization*
