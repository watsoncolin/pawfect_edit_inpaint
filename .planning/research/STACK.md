# Stack Research

**Domain:** GPU-accelerated inpainting service optimization (NVIDIA T4, 16GB VRAM)
**Researched:** 2026-02-21
**Confidence:** MEDIUM-HIGH (most claims verified via official docs or multiple sources; T4-specific benchmarks are estimated from cross-source triangulation)

---

## Executive Summary

The existing FLUX.1-Fill-dev + Q4 GGUF stack is the right inpainting model for quality — do not replace it. The 60-120s inference time is caused by three compounding problems: (1) too many inference steps (28 when 20 is the community sweet spot), (2) suboptimal guidance scale (10 when 30 is the official recommendation), and (3) CPU offload overhead that serializes component loading. The hallucination problem is caused entirely by the generic prompt — Florence-2-base (0.23B, fast) can generate context-aware surface descriptions in ~1-2s and eliminate the hallucination without a model change. SAM3 (Nov 2025) adds open-vocabulary text-prompted segmentation and scene understanding, making it a viable scene labeling source for background surface classification. torch.compile with regional compilation is compatible with T4 (SM75 / compute capability 7.5) and can provide meaningful speedups without changing the model.

**Bottom line:** Keep FLUX.1-Fill-dev. Fix the parameters. Add Florence-2 for prompt generation. Apply torch.compile. Explore Q5_K_M GGUF for better quality at comparable speed.

---

## Recommended Stack

### Core Inference Model

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| FLUX.1-Fill-dev | current (Nov 2024) | Primary inpainting model | Purpose-built for inpainting/outpainting; trained with guidance distillation for better efficiency; state-of-the-art quality for seamless background fill; already in production |
| GGUF Q5_K_M quantization | via diffusers GGUF loader | Quantized transformer weights | Q5_K_M fits in 16GB VRAM, delivers near-FP16 quality, recommended as the sweet spot for 16GB cards; current Q4_0 trades too much quality for minimal VRAM savings |
| diffusers FluxFillPipeline | >=0.32.0 | Pipeline orchestration | Official FLUX Fill support; integrates GGUF loading, CPU offload, VAE tiling, and torch.compile hooks |

### Inference Parameter Changes (no code refactor — change constants)

| Parameter | Current | Recommended | Why |
|-----------|---------|-------------|-----|
| `num_inference_steps` | 28 | 20 | Community consensus: FLUX.1-dev is tuned to ~20 steps; diminishing returns beyond 20-30; expected to cut inference time by ~30% |
| `guidance_scale` | 10 | 30 | Black Forest Labs' own example code uses `guidance_scale=30` for Fill models; higher values increase prompt adherence which reduces hallucination; current value of 10 is a text-to-image default, not the Fill recommendation |
| GGUF variant | Q4_0 | Q5_K_M | Q5 delivers noticeably better texture detail; speed difference between Q4 and Q5 is minimal on 16GB VRAM (both VRAM-bound not compute-bound); the quality improvement is worthwhile for background texture matching |

### Scene Labeling / Context-Aware Prompting

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| Florence-2-base | 0.23B params, MIT license | Background surface captioning from image | 0.23B model runs fast (~1-2s on T4); supports `<DETAILED_CAPTION>` and `<DENSE_REGION_CAPTION>` tasks; MIT license (no commercial restrictions); explicitly verified to deploy on NVIDIA T4; avoids the overhead of larger VLMs like LLaVA or BLIP2 |
| SAM3 (Meta Segment Anything 3) | Nov 2025 release | Open-vocabulary segmentation for background region isolation | Released Nov 2025; adds text-prompted segmentation ("grass", "wooden floor", "concrete") on top of SAM2's video tracking; already in infrastructure for leash detection; extending to return surface labels is architecturally clean; 30ms inference on H200 (T4 will be slower, estimate 200-500ms) |

**Prompt construction strategy:**
1. Florence-2 runs on the image region surrounding the mask (crop the ~2x mask bounding box from the original image)
2. Florence-2 `<DETAILED_CAPTION>` returns a natural language description of the surface/background
3. Extract surface keywords (grass, concrete, wood, leaves) from the caption
4. Build prompt: `"[surface description], seamless continuation, no objects"`
5. This replaces the current static `"empty ground, nothing here"` prompt

### Speed Optimization

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| torch.compile (regional) | PyTorch >=2.1 | Compile FLUX transformer blocks for faster execution | T4 is SM75 / compute capability 7.5 — officially supported by torch.compile; regional compilation targets only the DiT transformer blocks (which consume ~96% of compute time); delivers same speedup as full compile with 8x lower compile latency; compile happens once at startup so per-request overhead is zero |
| `enable_model_cpu_offload()` | current (already in use) | Keep transformer on GPU, offload VAE/text encoders when idle | Already deployed; correct choice for T4 (sequential offload causes 3x+ slowdown); model offload moves one whole component at a time vs sequential which offloads leaf-level parameters — much faster |
| `enable_vae_slicing()` | diffusers >=0.20 | Reduce peak VRAM during VAE decode | Minimal speed penalty; prevents OOM during decode of larger images; should be added alongside existing offload |
| `enable_vae_tiling()` | diffusers >=0.20 | Process high-res images in tiles | Needed only for images >2048px; low penalty for typical pet photos; add as safety measure |

**torch.compile implementation:**
```python
# After pipeline loads, before first inference:
pipe.transformer = torch.compile(
    pipe.transformer,
    mode="reduce-overhead",  # Good for repeated inference (same shapes)
    fullgraph=False,          # Required — FLUX has dynamic graph elements
)
```
For regional compile (preferred — lower compile time, same runtime speedup):
```python
for block in pipe.transformer.transformer_blocks:
    torch.compile(block, mode="reduce-overhead", fullgraph=False)
```

### Mask Preprocessing

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| OpenCV (cv2) | >=4.8 | Morphological mask cleanup | Remove scattered white specks via morphological opening (erosion then dilation); PIL alone lacks efficient morphological ops; OpenCV is the standard tool for this; adds <50ms overhead |
| scipy.ndimage (alternative) | >=1.11 | Morphological cleanup if OpenCV is not wanted | Pure Python alternative with `binary_opening`; slower than OpenCV but no additional system dependency |

**Mask cleanup recipe:**
```python
import cv2
import numpy as np

def clean_mask(mask_pil):
    mask_np = np.array(mask_pil.convert("L"))
    # Remove specks smaller than 5x5 pixels (opening)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(mask_np, cv2.MORPH_OPEN, kernel)
    # Slightly dilate to ensure full coverage of target region
    dilated = cv2.dilate(cleaned, kernel, iterations=1)
    return Image.fromarray(dilated)
```

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| transformers | >=4.47.0 (already in use) | Florence-2 model loading and inference | Required for Florence-2; already a project dependency |
| accelerate | >=0.26.0 (already in use) | Model offloading and device management | Already in use; ensure latest for group offload support |
| opencv-python-headless | >=4.8.0 | Mask morphological cleanup | Prefer headless (no GUI deps) for server deployment; add to pyproject.toml |

---

## Alternatives Considered

### Alternative Inpainting Models

| Model | Speed on T4 | Quality | Why Not Recommended |
|-------|------------|---------|---------------------|
| SD 1.5 Inpaint | ~5-10s at 512px | LOW — poor texture coherence, weak prompt adherence, visible seams | Quality is far below FLUX for background fill; would solve speed but fail the quality requirement; produces blurry, low-detail fills |
| SDXL Inpaint | ~15-30s at 1024px | MEDIUM — better than SD 1.5, notably worse than FLUX for seamless fill | Borderline on speed target; quality gap is too large vs FLUX for the seamless background use case; research confirms FLUX Fill is purpose-built and superior |
| Kandinsky 3 Inpaint | ~20-40s on T4 | MEDIUM | Limited community adoption, less tested for background fill use case; no compelling advantage over SDXL let alone FLUX |
| TurboFill (CVPR 2025, Adobe) | 4 steps, very fast | MEDIUM-HIGH — outperforms BrushNet but not purpose-built Fill models | Research paper only, no production-ready diffusers integration as of Feb 2026; builds on DMD2 which is SDXL-class, not FLUX-class |
| FLUX.1-Fill-pro | Fast (hosted API) | HIGH | Requires Black Forest Labs API — eliminates T4 GPU advantage and introduces per-call cost; incompatible with project's own-GPU constraint |

**Verdict:** Stay with FLUX.1-Fill-dev. The quality advantage is decisive for seamless background fill. Speed is achievable through parameter tuning and torch.compile without sacrificing quality.

### Alternative Quantization Schemes

| Scheme | T4 Support | Quality | Speed | Why Not Recommended (or deferred) |
|--------|-----------|---------|-------|-----------------------------------|
| Q4_0 GGUF (current) | YES | MEDIUM — noticeable texture artifacts | Baseline | Current choice; acceptable but Q5 is better |
| Q5_K_M GGUF | YES | HIGH — near-FP16 | ~same as Q4 (VRAM-bound, not compute-bound) | **Recommended upgrade** — better quality at same speed |
| Q8_0 GGUF | MAYBE — tight on 16GB | VERY HIGH | Same or slightly slower | May not fit in 16GB with full pipeline; Q5 is the better fit |
| torchao INT8 weight-only | YES | HIGH | Faster than GGUF (native PyTorch, no decompress step) | Promising but less tested with FLUX Fill pipeline; GGUF has proven diffusers integration; defer to future milestone |
| torchao FP8 | NO — requires compute capability >=8.9 | HIGHEST | 3x faster | T4 is compute capability 7.5 (Turing); FP8 is not available; confirmed by multiple sources |
| bitsandbytes INT8 | YES | HIGH | Slower than GGUF on T4 | Works but bitsandbytes INT8 inference is slower than GGUF for transformer blocks on Turing architecture |
| TensorRT | YES | HIGH | 2-4x faster than PyTorch | High compilation complexity; engine must be rebuilt for each input shape; dynamic shapes in FLUX make TensorRT integration non-trivial; defer unless other optimizations are insufficient |

### Alternative Scene Labeling Approaches

| Approach | Inference Time | Quality | Why Not Recommended |
|----------|---------------|---------|---------------------|
| Hardcoded prompt library | 0ms | LOW — same hallucination problem | Doesn't adapt to actual scene content; defeats the purpose |
| BLIP2 captioning | ~3-5s on T4 | MEDIUM | Larger model than Florence-2; longer inference; less structured output |
| LLaVA scene description | ~10-20s on T4 | HIGH | Too slow; model is 7B+; incompatible with <30s total budget |
| SAM3 semantic labels only | ~200-500ms | MEDIUM | SAM3 returns segmentation masks with concept labels; useful for region isolation but Florence-2 gives richer surface descriptions; use together: SAM3 to isolate background region, Florence-2 to describe it |
| GPT-4o Vision API | ~1-3s | VERY HIGH | External API call; latency variability; cost per request; adds external dependency |

**Verdict:** Florence-2-base for surface captioning (fast, fits T4, MIT license). SAM3 for identifying background region boundaries if needed. Use them as a pipeline: SAM3 isolates the background patch, Florence-2 describes it.

---

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| `enable_sequential_cpu_offload()` | Causes 3x+ inference slowdown by offloading leaf-level parameters individually; confirmed in diffusers GitHub issues | `enable_model_cpu_offload()` (already in use — keep it) |
| torchao FP8 quantization | Not supported on T4 (compute capability 7.5 < required 8.9); will fail at runtime | Q5_K_M GGUF or torchao INT8 weight-only |
| Replacing FLUX with SD 1.5/SDXL | Quality regression too severe for the seamless background fill use case; texture coherence and detail are far worse | Keep FLUX.1-Fill-dev, optimize parameters |
| Generic static prompt "empty ground, nothing here" | Root cause of hallucinations; model fills with whatever it imagines the surface to be | Florence-2-generated surface description from surrounding pixels |
| guidance_scale=10 for FLUX Fill | This is a text-to-image default; official FLUX Fill example uses 30; lower values reduce prompt adherence, increasing hallucination risk | guidance_scale=30 |
| num_inference_steps=50 | BFL's default for maximum quality; community confirms 20 steps is the practical sweet spot for FLUX Dev; current 28 is in the right direction but still above optimal for speed | 20 steps — estimated to reduce inference time ~30% |
| TensorRT (now) | High integration complexity with dynamic-shape FLUX models; not worth the effort until simpler optimizations are exhausted | torch.compile (same GPU, much easier integration) |
| SAM3 as a blocking dependency on the inpaint path | If SAM3/Florence-2 labeling isn't available yet, inpaint should proceed with a fallback prompt, not block | Async label generation with requeue on miss, fallback to region-crop captioning on Florence-2 alone |

---

## Stack Patterns by Scenario

**If Florence-2 labels are not yet available when inpaint worker picks up the job:**
- Crop the 2x mask bounding box region from the original image
- Run Florence-2 directly in the inpaint service on that crop (adds ~1-2s but avoids requeue)
- This is the fallback — SAM3-extended labels are the preferred path when available

**If VRAM is tight during torch.compile warmup:**
- Apply regional compile only to transformer blocks (not the full pipeline)
- Compile happens at service startup; first inference is slow due to tracing, subsequent calls are fast
- Ensure `mode="reduce-overhead"` for repeated fixed-shape inference (most pet photos are similar sizes)

**If Q5_K_M GGUF doesn't fit in 16GB with model_cpu_offload:**
- Fall back to Q4_K_M (better than Q4_0 for quality)
- Alternatively, enable VAE tiling + VAE slicing to reduce peak VRAM during decode, which may allow Q5 to fit

---

## Estimated Inference Time Budget on T4 (16GB)

These are estimates based on cross-source triangulation — not direct T4 benchmarks. Actual numbers require profiling.

| Stage | Current | After Optimization | Notes |
|-------|---------|-------------------|-------|
| Model load (startup, not per-request) | ~60s | ~60s (no change) | Baked into Docker image |
| Florence-2 scene captioning | 0s (not yet built) | ~1-2s | 0.23B model, short inference |
| Mask preprocessing (morphological) | 0s (not yet built) | <0.1s | OpenCV ops are near-instant |
| FLUX Fill inference (28 steps, guidance=10) | 60-120s | — | Current baseline |
| FLUX Fill inference (20 steps, guidance=30, Q5) | — | ~35-50s | Estimated; 30% step reduction + compile warmup |
| FLUX Fill inference (20 steps, +torch.compile) | — | ~25-40s | Estimated; compile adds 10-20% speedup after warmup |
| VAE decode + compositing | included above | included above | — |
| **Total end-to-end** | **60-120s** | **~28-44s** | Optimistic end depends on T4 specific behavior |

**Confidence: LOW** on exact timing estimates. T4 benchmark data for FLUX Fill specifically is not available in public sources. Profiling is mandatory before declaring success.

**Key risk:** The <30s target may require torch.compile + 20 steps + Q5 all together. If torch.compile doesn't provide the expected speedup on T4's Turing architecture, an alternative path is to reduce to 15 steps (quality may suffer) or revisit TensorRT.

---

## Installation Changes

```bash
# Upgrade diffusers to ensure FluxFillPipeline and GGUF Q5 support
uv add "diffusers>=0.32.0"

# Add OpenCV headless for mask morphological cleanup
uv add opencv-python-headless

# Florence-2 uses transformers (already present) — no new dep needed
# torch is already present — torch.compile is built-in since 2.0

# Q5_K_M GGUF model URL (replace current Q4_0 URL in flux_inpaint.py)
# https://huggingface.co/YarvixPA/FLUX.1-Fill-dev-GGUF/blob/main/flux1-fill-dev-Q5_K_M.gguf
# (Verify this file exists on HuggingFace — search YarvixPA or city96 repos for Fill Q5)
```

---

## Version Compatibility

| Package | Version Needed | Compatibility Notes |
|---------|---------------|---------------------|
| diffusers | >=0.32.0 | FluxFillPipeline with GGUF + VAE slicing/tiling support; check release notes for exact FLUX Fill GGUF support introduction |
| torch | >=2.1.0 (already pinned) | torch.compile stability; 2.1+ required for `reduce-overhead` mode |
| transformers | >=4.47.0 (already pinned) | Florence-2 support added in 4.44+; current pinned version is sufficient |
| accelerate | >=0.26.0 (already pinned) | enable_model_cpu_offload; group offload hooks added in 0.30+ if needed |
| opencv-python-headless | >=4.8.0 | No minimum hard constraint; 4.8 is current stable; avoid opencv-python (pulls GUI deps) |

---

## Sources

- `black-forest-labs/FLUX.1-Fill-dev` HuggingFace model card — official guidance_scale=30, num_inference_steps=50 defaults — MEDIUM confidence (verified via WebFetch)
- HuggingFace Diffusers FLUX documentation (`/docs/diffusers/main/en/api/pipelines/flux`) — FluxFillPipeline API, memory optimizations — HIGH confidence (official docs)
- PyTorch blog "torch.compile and Diffusers: A Hands-On Guide to Peak Performance" (July 2025) — regional compile strategy, 3.3x speedup with FP8+compile — MEDIUM confidence (WebSearch verified)
- torchao FP8 T4 incompatibility — compute capability 7.5 < 8.9 requirement — HIGH confidence (multiple sources confirm)
- Florence-2-base HuggingFace model card (`microsoft/Florence-2-base`) — 0.23B params, T4 support, MIT license, captioning tasks — HIGH confidence (verified via WebFetch)
- SAM3 Meta release blog (Nov 2025) and arxiv paper 2511.16719 — text-prompted segmentation, 30ms on H200, concept understanding — MEDIUM confidence (official Meta release + arxiv)
- Community consensus on FLUX steps: 20-30 steps optimal for Dev variant — MEDIUM confidence (multiple community sources agree)
- Q5_K_M vs Q4_0 quality comparison — MEDIUM confidence (multiple GGUF comparison articles agree)
- `enable_sequential_cpu_offload` 3x slowdown — MEDIUM confidence (diffusers GitHub issue #2266)
- OpenCV morphological operations for mask cleanup — HIGH confidence (official OpenCV docs)
- TurboFill (CVPR 2025, arxiv 2504.00996) — 4-step inpainting, outperforms BrushNet — MEDIUM confidence (arxiv + CVPR proceedings)

---

*Stack research for: Pawfect Edit Inpaint — speed and quality optimization on T4 GPU*
*Researched: 2026-02-21*
