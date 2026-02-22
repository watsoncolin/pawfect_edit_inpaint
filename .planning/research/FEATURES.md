# Feature Research

**Domain:** AI Inpainting Service — Background Fill / Object Removal for Pet Photography
**Researched:** 2026-02-21
**Confidence:** MEDIUM (WebSearch-sourced findings verified across multiple sources; no Context7 library docs applicable to this domain research)

---

## Feature Landscape

### Table Stakes (Users Expect These)

Features users assume exist in any high-quality inpainting service. Missing these = output feels broken or amateur.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Mask-based compositing (preserve non-masked pixels) | Without this, inpainting touches pixels that weren't selected — catastrophic quality regression | LOW | Already implemented. Must never regress. Hard compositing: copy original pixels back outside mask post-inference. |
| Context-aware prompt for background fill | Generic prompts ("empty ground") cause hallucination of wrong objects. Users expect seamless removal, not object replacement. | MEDIUM | The core quality fix for session D1A406F5. Describes the actual surface/material under the mask (e.g., "patchy grass with fallen leaves"). |
| Mask artifact cleanup before inference | Scattered mask specks cause the model to inpaint tiny disconnected islands, creating visible noise | MEDIUM | Morphological operations (erosion then dilation = open; dilation then erosion = close) to remove isolated white pixels. Use `cv2.morphologyEx` with a small kernel (3-5px). |
| Mask dilation (expand mask slightly beyond object edge) | Hard mask edges produce visible seams where inpainted region meets original image | LOW | Google Vertex AI recommends 0.01 dilation (percentage of image width). For FLUX Fill, 8-16px dilation typical. Feathering/Gaussian blur on mask edge further softens seam. |
| Tuned inference parameters for background fill | Default 28 steps / guidance=10 is calibrated for general generation, not background-matching fill | MEDIUM | Background fill needs lower guidance scale (2-5) to allow context-driven generation rather than prompt-driven. Steps can be reduced to 15-20 for speed without visible quality loss for fill tasks. See: current 28 steps / guidance=10 is likely over-parameterized for pure background fill. |
| Graceful handling of mask-only or prompt-only inference | Google's Vertex AI recommendation: for background removal, omit the prompt and let the model infer from context. Some tasks benefit from prompt; some benefit from omitting it | LOW | Implement fallback: if no scene labels available, use empty/minimal prompt rather than "empty ground". Empty prompt lets FLUX Fill use surrounding pixels as context. |
| Seam blending / post-process harmonization | Even good inpainting can have color temperature or brightness mismatches at the boundary | MEDIUM | Basic: Gaussian-blur the mask boundary for soft compositing. Advanced: img2img refinement pass at low strength (0.2-0.35 denoising) over the boundary region only. |

### Differentiators (Competitive Advantage)

Features that materially improve output quality or reliability beyond baseline expectations.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Scene-label-driven prompt construction (SAM3 integration) | Converts "what is the background?" into a precise material/surface description that guides fill generation. Eliminates hallucinated objects | MEDIUM | Already planned. SAM3 returns class labels (grass, asphalt, leaves, tile, etc.). Prompt template: "seamless {material}, {texture detail}, matching surrounding background". Confidence: MEDIUM — SAM3 label → text mapping is project-specific; need to define label vocabulary. |
| VLM-based background description as fallback | When SAM3 labels are unavailable or low-confidence, use BLIP-2 / LLaVA to caption the region surrounding the mask and extract material descriptors | HIGH | Run VLM on an inpainted crop of the surrounding area (not the masked region). Extract material nouns. BLIP-2 is lightweight enough for T4. Adds ~2-4s. Use only as fallback to avoid latency on happy path. |
| Harmony score-guided selective re-inpainting | After initial inpaint, measure boundary harmony (color, texture match at mask edge) and selectively re-inpaint only low-harmony regions | HIGH | Research: "Harmony score-guided inpainting: Iterative refinement" (2025). Requires a harmony metric (SSIM or perceptual similarity at boundary). Adds significant latency — only viable if initial pass fails quality threshold. Defer unless quality still insufficient after other improvements. |
| Inpainting Harmony Score (IHS) as quality gate | Automatically detect failed inpaints without human review. Re-queue or flag jobs with IHS below threshold | HIGH | Research: Re-Inpainting Self-Consistency Evaluation (2024, CIKM). Re-inpaint a second time and measure consistency between pass 1 and pass 2. If divergent, flag as failed. Adds full second inference cycle — prohibitive for <30s target unless fast path optimized first. |
| MAE-based object hallucination suppression (ASUKA) | Fine-tuned Masked Auto-Encoder prior replaces text conditioning to constrain generation to plausible reconstructions rather than invented content | HIGH | Research: ASUKA (CVPR 2025). Effectively post-processes a frozen FLUX Fill with a lightweight alignment module. Addresses root cause of hallucination. Significant engineering effort to integrate with existing GGUF pipeline. Defer to Phase 2+ after simpler prompt/parameter fixes are validated. |
| Inference step reduction via Turbo LoRA | Turbo LoRA distillation can reduce FLUX Fill from 28 steps to 8 steps while maintaining acceptable quality | HIGH | Community evidence: FLUX Turbo LoRA works with Fill pipeline. 8 steps may sacrifice quality for background textures. Target 15 steps as a middle ground. Requires testing — confidence LOW on exact step count for acceptable fill quality. |
| Mask quality assessment before inference | Detect and reject/correct masks that are too large (>30% of image), too sparse, or contain isolated artifacts before spending GPU time | MEDIUM | Simple heuristics: pixel count ratio, connected component analysis. Prevents wasted inference on unfixable inputs. Fast to implement. |

### Anti-Features (Commonly Requested, Often Problematic)

Features that seem appealing but introduce costs outweighing benefits in this context.

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| High mask blur (>16px) to soften edges | Sounds like it should improve blending | At high values, the blurred mask has no true 1-boundary region — model sees near-zero mask and generates unchanged content, or the blurred region is too wide and destroys the inpainting anchor. SDXL diffusers forced binarization breaks blur entirely. | Use hard mask with slight dilation (8-12px). Apply soft compositing as a post-process by Gaussian-blending the boundary after inference, not before. |
| Real-time streaming inpainting | Seems like faster UX | T4 inference cannot achieve sub-5s for FLUX Fill at any reasonable quality. Streaming partial denoised frames shows unstable intermediate artifacts, confusing users. Async is the right model. | Keep async Pub/Sub. Display skeleton/progress indicator in app during processing. |
| High guidance scale (>10) for "following the prompt better" | Intuitive that higher guidance = better adherence | For background fill, high guidance forces the model to generate exactly the prompt text, overriding contextual information from surrounding pixels. Grass becomes too uniform, textures too synthetic. Research shows quality improvement plateaus at guidance ~4.0 for contextual fill. | Use guidance 2-5 for background fill. Reserve high guidance (7-10) only for "add new object" tasks, not "remove object" tasks. |
| Multi-pass full re-inference for quality improvement | More passes = better quality (true for some use cases) | Each FLUX Fill inference on T4 costs 20-60s. Two full passes = 40-120s, already over budget. Harmony-score-guided partial re-inpainting on a small boundary crop is far cheaper. | Targeted boundary refinement: img2img at 0.2-0.35 strength over a 20% expansion of the mask boundary only. Fraction of full inference cost. |
| Generating multiple candidate outputs and picking the best | Sampling multiple results and selecting highest quality sounds robust | Multiplies inference cost directly (4 candidates = 4x cost). T4 budget cannot absorb this. Google Vertex AI supports 1-4 samples — valid for their hardware, not for ours. | Use a deterministic seed + low temperature (guidance scale tuned for consistency). A single well-guided pass is more reliable than sampling randomly. |
| On-device inpainting for large masks | Reduces server load | iOS device GPUs cannot handle FLUX Fill. Small masks (<15% area) already handled on-device; large masks exist specifically because they exceed on-device capacity. | Keep current split: small masks → on-device, large masks → server FLUX Fill. Do not blur this boundary. |

---

## Feature Dependencies

```
[Scene-label-driven prompt (SAM3)]
    └──requires──> [SAM3 label → text vocabulary mapping]
    └──requires──> [SAM3 service extension to return labels]
    └──enhances──> [Context-aware prompt for background fill]

[Context-aware prompt for background fill]
    └──requires──> [At least one of: SAM3 labels OR VLM caption OR empty-prompt fallback]

[VLM-based background description]
    └──enhances──> [Context-aware prompt for background fill]
    └──conflicts with latency target──> [<30s end-to-end]
    └──use only as──> [Fallback when SAM3 unavailable]

[Mask artifact cleanup]
    └──must precede──> [Inference parameter tuning]
    └──must precede──> [Mask dilation]
    (Cleanup before dilation prevents dilating artifacts)

[Mask dilation]
    └──must precede──> [Inference]
    └──enhances──> [Seam blending]

[Seam blending / boundary harmonization]
    └──requires──> [Mask-based compositing] (to know which boundary to blend)
    └──enhances──> [Inpainting Harmony Score]

[Inference step reduction (Turbo LoRA)]
    └──conflicts with──> [MAE/ASUKA hallucination suppression]
    (Both modify the inference pipeline; combining requires careful integration)

[Inpainting Harmony Score quality gate]
    └──requires──> [Seam blending] (baseline quality must be good enough to measure)
    └──enables──> [Harmony score-guided selective re-inpainting]

[Mask quality assessment before inference]
    └──must precede──> [All inference features]
    (Prevents wasted GPU time on malformed inputs)
```

### Dependency Notes

- **Scene-label-driven prompt requires SAM3 label vocabulary**: The label-to-text mapping (e.g., `grass` → `"patchy grass with natural variation"`) must be defined as a lookup table or template system before this feature works meaningfully.
- **Mask artifact cleanup must precede mask dilation**: Dilating before cleanup expands artifacts into larger blobs; cleanup first, then dilate the clean mask.
- **VLM fallback conflicts with latency target**: BLIP-2 inference adds 2-4s. Only deploy as fallback to avoid adding latency on the happy path (SAM3 labels present).
- **Turbo LoRA conflicts with ASUKA**: Both modify FLUX Fill's inference pipeline. ASUKA wraps the frozen model; Turbo LoRA modifies the model weights. Integration complexity is high.
- **Harmony score re-inpainting requires baseline quality**: If initial inpainting is bad everywhere, re-inpainting specific regions is still expensive. Get baseline quality right first via prompt + parameter tuning.

---

## MVP Definition

### Launch With (v1 — this milestone)

Minimum feature set to achieve acceptable background fill quality within T4 latency constraints.

- [ ] **Mask artifact cleanup** — morphological open operation to remove scattered specks before inference. Cheap, high impact.
- [ ] **Mask dilation (8-12px)** — expand mask to capture edge fringing. Prevents visible seam lines.
- [ ] **Context-aware prompt from SAM3 labels** — consume scene labels already planned from SAM3 extension. Core quality fix for hallucination problem.
- [ ] **Empty-prompt fallback** — when SAM3 labels not available, send empty/no prompt instead of "empty ground". Lets FLUX Fill use surrounding pixel context. Zero-cost improvement over current approach.
- [ ] **Guidance scale reduction for fill tasks** — change from guidance=10 to guidance=2-4 for background fill mode. One-line change, significant hallucination reduction.
- [ ] **Inference step reduction** — test 20 steps vs 28 steps for background fill. Potentially saves 8-15s per job. Validate quality parity before deploying.
- [ ] **Soft boundary compositing** — Gaussian-blend the boundary between inpainted region and original at post-process time. Low-cost, visually significant.

### Add After Validation (v1.x)

Features to add once core quality issues are resolved.

- [ ] **Mask quality assessment / rejection heuristics** — reject or flag masks with >40% image coverage or <50 connected pixels before queuing. Add after confirming which mask pathologies actually reach the service.
- [ ] **VLM-based background description fallback** — BLIP-2 caption of surrounding region when SAM3 labels absent. Add only if empty-prompt fallback proves insufficient.
- [ ] **Boundary img2img refinement pass** — 0.2-0.35 denoising strength over mask boundary region only. Add if seam artifacts persist after soft compositing.

### Future Consideration (v2+)

Defer until quality baseline established and latency budget confirmed.

- [ ] **Inpainting Harmony Score quality gate** — automated pass/fail scoring via re-inpainting consistency. Adds full inference cycle; only viable if step reduction creates headroom.
- [ ] **Harmony score-guided selective re-inpainting** — targeted re-generation of low-harmony boundary subregions. Complex; needs harmony score infrastructure first.
- [ ] **ASUKA / MAE hallucination suppression** — if prompt + guidance tuning still produces hallucinations on complex scenes. High integration cost against GGUF pipeline.
- [ ] **Turbo LoRA step distillation** — 8-step inference. Needs careful quality validation for background fill specifically; community evidence is for general generation, not fill-specific.

---

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| Context-aware prompt from SAM3 labels | HIGH | MEDIUM | P1 |
| Empty-prompt fallback (no prompt) | HIGH | LOW | P1 |
| Guidance scale reduction (10 → 2-4) | HIGH | LOW | P1 |
| Mask artifact cleanup (morphological) | HIGH | LOW | P1 |
| Mask dilation (8-12px) | HIGH | LOW | P1 |
| Inference step reduction (28 → 20) | HIGH | LOW | P1 |
| Soft boundary compositing | MEDIUM | LOW | P1 |
| Mask quality assessment / rejection | MEDIUM | MEDIUM | P2 |
| VLM fallback description | MEDIUM | HIGH | P2 |
| Boundary img2img refinement pass | MEDIUM | MEDIUM | P2 |
| Inpainting Harmony Score gate | LOW | HIGH | P3 |
| Harmony-guided selective re-inpainting | LOW | HIGH | P3 |
| ASUKA hallucination suppression | MEDIUM | HIGH | P3 |
| Turbo LoRA step distillation | HIGH | HIGH | P3 |

**Priority key:**
- P1: Must have for this milestone
- P2: Should have, add when possible
- P3: Future milestone

---

## Competitor Feature Analysis

| Feature | Adobe Firefly (Photoshop) | Google Vertex AI / Imagen 3 | LaMa / lama-cleaner (OSS) | Our Approach |
|---------|--------------------------|------------------------------|---------------------------|--------------|
| Background-aware fill | Empty prompt → context-from-pixels; optional text for overrides | Empty prompt recommended for removal; MASK_MODE_BACKGROUND automatic segmentation | No text conditioning; purely patch/frequency-based fill | SAM3 labels → structured prompt; empty prompt fallback |
| Mask preprocessing | Not exposed to user; internal dilation | Explicit `maskDilation` parameter (0.01 recommended) | Internal; user controls brush only | Programmatic dilation + artifact cleanup before inference |
| Quality validation | Not exposed; regenerate manually | Not exposed; sample up to 4 candidates | Not applicable | IHS (future); manual retry currently |
| Step/speed tuning | Not user-exposed | `baseSteps` parameter (12 default, up to 75) | Near-instant (non-diffusion) | Tunable; target 20 steps |
| Scene understanding | Implicit in model training | MASK_MODE_SEMANTIC with class IDs | None | Explicit via SAM3 label integration |
| Multi-pass refinement | img2img refinement recommended post-inpaint | Not exposed | Not applicable | Boundary img2img pass (v1.x) |

---

## Sources

- Google Vertex AI Imagen object removal documentation: https://docs.cloud.google.com/vertex-ai/generative-ai/docs/image/edit-remove-objects — MEDIUM confidence (official docs)
- ASUKA: "Towards Enhanced Image Inpainting: Mitigating Unwanted Object Insertion and Preserving Color Consistency" (CVPR 2025): https://arxiv.org/html/2312.04831v3 — MEDIUM confidence (peer-reviewed, verified via WebFetch)
- Re-Inpainting Self-Consistency Evaluation (2024, CIKM): https://arxiv.org/abs/2405.16263 — MEDIUM confidence (peer-reviewed)
- Harmony score-guided inpainting: https://www.sciencedirect.com/science/article/abs/pii/S092523122501673X — MEDIUM confidence (journal, 2025)
- Scene Graph Driven Text-Prompt Generation for Image Inpainting (CVPR 2023W): https://openaccess.thecvf.com/content/CVPR2023W/GCV/papers/Shukla_Scene_Graph_Driven_Text-Prompt_Generation_for_Image_Inpainting_CVPRW_2023_paper.pdf — MEDIUM confidence
- FLUX Fill inference parameters: https://huggingface.co/docs/diffusers/api/pipelines/flux — MEDIUM confidence (official Hugging Face docs)
- AUTOMATIC1111 mask blur discussion: https://github.com/AUTOMATIC1111/stable-diffusion-webui/discussions/1526 — LOW confidence (community forum)
- SDXL mask blur binarization bug: https://github.com/huggingface/diffusers/issues/10690 — MEDIUM confidence (official repo issue)
- Stability AI / community inpainting guides on guidance scale threshold at 4.0: WebSearch multiple sources — LOW confidence
- Flux Turbo LoRA compatibility with Fill pipeline: community blog posts — LOW confidence (needs validation)
- IMFine (CVPR 2025) multi-view refinement: https://arxiv.org/abs/2503.04501 — MEDIUM confidence (peer-reviewed)
- LaMa / lama-cleaner: https://github.com/advimman/lama — HIGH confidence (official repo)
- VideoPainter / SAM2 scene context for inpainting: https://github.com/TencentARC/VideoPainter — MEDIUM confidence

---

*Feature research for: AI Inpainting Service — Background Fill / Object Removal*
*Researched: 2026-02-21*
