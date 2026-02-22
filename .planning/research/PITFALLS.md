# Pitfalls Research

**Domain:** Diffusion inpainting optimization — T4 GPU, FLUX.1-Fill-dev, async SAM3 integration, Pub/Sub job queue
**Researched:** 2026-02-21
**Confidence:** MEDIUM (WebSearch + official docs; no Context7 coverage for diffusion-specific patterns)

---

## Critical Pitfalls

### Pitfall 1: Guidance Scale Mismatch Between Fill-Dev and Standard FLUX

**What goes wrong:**
The official FLUX.1-Fill-dev model card shows `guidance_scale=30` in its example code. Developers assume this is the production-recommended value and lock it in. At guidance=30, Fill-dev over-constrains output to the text prompt, producing flat, plastic-looking fills that ignore surrounding texture. The project currently uses `guidance_scale=10` — already high for FLUX.1-dev (whose sweet spot is 1.5-5) but correct for Fill-dev's distillation-modified behavior. Changing models without recalibrating guidance is the common failure: guidance that works well for Fill-dev causes artifacts or prompt ignoring on any alternative model.

**Why it happens:**
FLUX.1-Fill-dev uses a different guidance distillation than FLUX.1-dev. The guidance scale is not comparable between model variants. Community workflows copy the model card example value (30) without understanding it is for a specific pipeline configuration. When switching to SDXL Lightning or LCM-based alternatives, guidance behaves completely differently (typical range 1-3).

**How to avoid:**
- Before testing any alternative model, establish a fresh guidance calibration pass (try 1, 3, 5, 7 as anchors)
- Never transfer guidance_scale directly when changing model architecture
- Document the guidance range separately for each candidate model during evaluation
- For background fill specifically, lower guidance (3-7 range) allows the model to derive texture from context rather than forcing prompt content

**Warning signs:**
- Generated fill looks plastic or "AI-generated" rather than textural
- Fill matches the prompt exactly but doesn't match the surrounding image
- Fill is high-contrast with sharp internal details in an area that should be uniform (grass, dirt)
- Different seeds produce wildly different results (guidance too high causes mode collapse)

**Phase to address:**
Model evaluation phase — establish per-model guidance calibration before any quality comparison. Do not compare models at identical guidance values.

---

### Pitfall 2: Pub/Sub Requeue Loop Without Termination Condition

**What goes wrong:**
The proposed flow — inpaint service requeues if SAM3 labels aren't available yet — creates an unbounded retry loop if SAM3 fails silently or takes longer than expected. The existing codebase already fixed one infinite retry issue (session deletion during inpaint, commit 499a2b9). A second retry path around SAM3 readiness duplicates the risk. Pub/Sub's dead-letter topic enforcement is approximate — the service may deliver more times than configured before forwarding. Without an explicit attempt counter embedded in the message payload, the service cannot distinguish "SAM3 not ready yet (normal)" from "SAM3 permanently failed (stop retrying)."

**Why it happens:**
The requeue pattern feels safe because Pub/Sub has retry limits, but Pub/Sub's attempt counting resets in certain conditions and only applies when IAM permissions for the dead-letter topic are correctly configured. Developers assume Pub/Sub will stop retrying; it may not in all cases. The existing max-retry limit (5 attempts, commit 72fe58f) applies to the worker-level retry logic, not to SAM3-specific requeues — these are different code paths.

**How to avoid:**
- Embed an attempt counter in the Pub/Sub message payload (not relying on Pub/Sub's native counting alone)
- Distinguish retry types in the message: transient worker errors vs. SAM3 wait vs. permanent failures
- Set a hard wall-clock deadline for SAM3 readiness (e.g., if labels not available within 15 seconds of job enqueue, proceed without them rather than requeue)
- Configure a dead-letter topic for SAM3-wait requeues with max_delivery_attempts=10 and verify IAM permissions are correct before deployment
- Add exponential backoff on requeue delay (not constant delay) to avoid thundering herd when SAM3 service recovers after downtime

**Warning signs:**
- Same job_id appearing more than twice in inpaint service logs within a short window
- Pub/Sub subscription undelivered message count growing continuously
- SAM3 service logs show no activity but inpaint service continues requeuing
- Cloud Run billing anomaly: worker instances processing far more messages than jobs submitted

**Phase to address:**
SAM3 integration phase — design the requeue termination contract before writing any requeue code. The termination condition is a first-class requirement, not an afterthought.

---

### Pitfall 3: Prompt Over-Specification Prevents Context Derivation

**What goes wrong:**
The current generic prompt ("empty ground, nothing here") under-specifies and causes hallucinations. The intuitive fix is to be more specific — generating verbose prompts from SAM3 labels like "patchy green and yellow grass with scattered brown fallen oak leaves, moist soil visible between grass blades, natural outdoor ground texture." This over-specified prompt forces the model to generate exactly that content rather than blending from surrounding context. The model becomes prompt-driven instead of context-driven, and the fill no longer matches the specific area being inpainted.

**Why it happens:**
Inpainting models for background fill work best when the prompt names the surface category and lets the model derive the rest from the surrounding unmasked pixels. The model's self-attention over unmasked regions provides texture, lighting, and color matching that no text prompt can match. Over-specification with SAM3 labels substitutes text for what context should provide.

**How to avoid:**
- Limit SAM3-derived prompt contribution to surface category terms only: "grass", "pavement", "wood floor", "sand" — not material descriptions
- Never describe color, wetness, lighting conditions, or leaf content in the prompt — those come from the image context
- The prompt should complete the sentence "fill this area with ___" where the blank is 1-3 words
- Test prompt specificity with a grid: single word → 2-word phrase → 5-word phrase → observe quality cliff
- Negative prompts (if used) should suppress the removed object category, not describe the background

**Warning signs:**
- Fill looks realistic in isolation but has wrong color tone relative to surrounding image
- Fill has content that matches the prompt but not the photo (e.g., darker grass when surrounding grass is bright)
- Two adjacent inpaints with different SAM3 labels produce inconsistent-looking results in the same image

**Phase to address:**
Prompt engineering phase — run the specificity grid experiment before integrating SAM3 labels. Establish what the minimum useful label is (category noun vs. adjective-noun phrase) before committing to the label schema.

---

### Pitfall 4: torch.compile Silently Failing or Falling Back on T4 (sm_75)

**What goes wrong:**
T4 GPUs use the Turing architecture (CUDA compute capability sm_75). Flash Attention, the primary beneficiary of torch.compile in transformer models, requires Ampere (sm_80) or newer. Attempting `torch.compile` on FLUX.1-Fill-dev's transformer on T4 either silently falls back to eager mode (no speedup, no error) or raises a Triton-related ImportError. The common failure mode is observing no inference speedup and concluding torch.compile "doesn't help" — without confirming whether compilation actually ran or fell back.

Additionally, T4 does not support `bfloat16` natively (a known PyTorch limitation for sm_75). If compilation interacts with dtype handling, it can produce silent numerical errors rather than crashes.

**How to avoid:**
- Verify compilation actually ran: `torch._dynamo.explain(model, ...)` or check `torch._dynamo.config.suppress_errors`
- Do not set `suppress_errors=True` in production — it hides failures
- Use `mode="reduce-overhead"` rather than `mode="max-autotune"` on T4; max-autotune requires Triton kernels that may not be available for sm_75
- Profile with and without compile using identical inputs and step counts; only accept results that show measurable wall-clock improvement
- If torch.compile is not viable, evaluate xformers memory-efficient attention (separate from Flash Attention, has broader GPU support) as an alternative

**Warning signs:**
- Inference time identical with and without `torch.compile` wrapping
- Triton errors in logs that are caught and suppressed
- `UserWarning: torch.compile is not supported` in PyTorch logs
- CUDA OOM errors that appear only after enabling compile (CUDA graphs can increase peak memory)

**Phase to address:**
Inference optimization phase — validate torch.compile actually runs before building any performance baseline around it.

---

### Pitfall 5: Mask Artifact Preprocessing That Biases the Model

**What goes wrong:**
The existing masks have scattered white specks (noted in PROJECT.md as a known quality problem). The intuitive fix is morphological opening (erosion then dilation) to remove small noise. Erosion that is too aggressive removes legitimate mask coverage near the object boundary, leaving thin strips of the removed object unmasked — the model then "sees" the edge of the hand/leash handle and regenerates it. Dilation that is too large expands the mask into unrelated areas and forces the model to regenerate correct-looking background that was never damaged.

Research confirms that mask shape itself can bias generation: if the mask outline clearly resembles the shape of a hand, the model's training on similar mask shapes increases the probability of regenerating hand-like content.

**How to avoid:**
- Use dilation, not erosion, as the primary cleanup operation on sparse-noise masks — remove specks by dilation (expand specks until they merge and then trim with erosion), not by erosion alone
- Target 5-15px dilation for the mask border beyond the object boundary — enough to eliminate edge bias without inpainting healthy image area
- Validate mask preprocessing with a visual diff overlay before running inference
- After preprocessing, verify no isolated white pixels remain using `cv2.connectedComponentsWithStats` and filter components below 50px area
- Do not blur the mask (Gaussian blur on a binary mask creates gray values that most pipelines binarize back to harsh edges anyway)

**Warning signs:**
- Inpainted area shows thin slivers of the original removed object at the mask boundary
- Model regenerates content that resembles the removed object's shape (mask shape bias)
- Dilation creates a visible "halo" of regenerated content around the original object location
- After morphological operations, mask extends into the pet subject area

**Phase to address:**
Mask preprocessing phase — implement and validate preprocessing on the known failing session (D1A406F5) before any model parameter optimization. Bad masks make all other optimizations irrelevant.

---

### Pitfall 6: Model Swap Without Revalidating the Full Pipeline

**What goes wrong:**
Evaluating alternative inpainting models (SD Turbo, Lightning LoRA, smaller FLUX variants) by running them in isolation and comparing output images. The full pipeline includes: GGUF-quantized loading, model CPU offload, mask compositing that preserves unmasked pixels, and preview thumbnail generation. A model that produces good images in isolation may fail in the pipeline because: (1) it requires a different VAE architecture, (2) its output resolution or latent space dimensions differ, (3) it produces NaN values under FP16 that only manifest during compositing, (4) its recommended pipeline class differs from FluxFillPipeline.

The FLUX.1-Fill-dev uses a 4-channel inpaint-specific latent conditioning; models like SD1.5 use 9-channel inpaint latents. These are not interchangeable. Using the wrong pipeline class produces silently wrong outputs, not errors.

**How to avoid:**
- When evaluating alternatives, test through the complete pipeline code path, not via standalone model.generate() calls
- Verify the pipeline class explicitly: FluxFillPipeline is not interchangeable with StableDiffusionInpaintPipeline
- Check latent channel count matches the model card specification before loading
- Run the same test image (session D1A406F5) through each candidate model in the full pipeline before any quality judgment
- Confirm the VAE is architecture-matched (FLUX uses ae.safetensors; SDXL uses sdxl_vae.safetensors — they are not swappable)

**Warning signs:**
- Model evaluation shows great quality in notebook but artifacts appear in the service
- Color shifts at mask boundaries that didn't appear in standalone testing
- NaN in output tensor (check `torch.isnan(output).any()` before compositing)
- FP16 NaN errors specifically (SDXL VAE is known to produce NaN under FP16)

**Phase to address:**
Model evaluation phase — write the evaluation harness to use the production pipeline code path from the start.

---

### Pitfall 7: Q4 GGUF Quantization Quality Loss That Masquerades as a Model Problem

**What goes wrong:**
The current model uses Q4 GGUF quantization. When quality issues appear (hallucinations, wrong content), the investigation focuses on prompt engineering and model parameters — but Q4 quantization itself may be causing or amplifying the hallucinations. Q4 quantizes to 4 bits per weight, losing approximately 75% of precision. Community benchmarks confirm Q4/Q3 hurt quality visibly for FLUX DiT models, while Q8_0 is nearly lossless. Misattributing Q4 quality loss to prompt or parameter problems leads to chasing the wrong fix.

**Why it happens:**
The codebase uses Q4 for memory efficiency on T4 16GB VRAM. This is a reasonable tradeoff, but quality experiments must control for quantization level. Testing Q8_0 on the same failing session would isolate whether quantization is contributing to hallucinations.

**How to avoid:**
- Run the known-failing session (D1A406F5) with Q8_0 GGUF before optimizing prompt or parameters
- If Q8_0 fixes the hallucination, the problem is quantization quality, not prompting — upgrade is justified if VRAM permits
- Q8_0 for FLUX DiT also generates 3x faster due to reduced CPU offload — the memory/speed tradeoff may flip from Q4's favor
- Keep Q4 available as a fallback and document VRAM headroom required for Q8_0 under the current pipeline

**Warning signs:**
- Hallucinations persist even with well-crafted prompts and correct mask
- Output improves dramatically with higher inference steps but degrades at 28 steps — quantization amplifies step sensitivity
- Generated textures look "smeared" or lack the fine grain of surrounding image

**Phase to address:**
Baseline quality audit — before any optimization work, run Q4 vs Q8_0 on the failing session. This is a 1-hour experiment that scopes all subsequent work.

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Hardcoded inference steps (28) | Predictable latency | Cannot tune per-image complexity; under-steps simple fills, over-steps complex ones | Never for production — parameterize steps |
| SAM3 label passed raw to prompt | Simple integration | SAM3 label vocabulary changes break prompt quality silently | Acceptable in MVP if label set is frozen and small |
| Requeue with fixed 5-second delay | Simple to implement | Thundering herd when SAM3 recovers after outage; all delayed jobs arrive simultaneously | Never — use jitter (+/- 50% of delay) from the start |
| Single mask preprocessing function | Simple code | Cannot tune per-mask characteristics (sparse specks vs. thick borders) | Acceptable in MVP; flag for parameterization |
| Q4 GGUF without quality baseline | Smaller model, lower VRAM | Cannot distinguish quantization quality loss from model/prompt quality loss | Acceptable temporarily if Q8_0 baseline is established first |

---

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| SAM3 scene labels | Trusting SAM3 label vocabulary to match inpainting prompt vocabulary without verification | Map SAM3 label categories to prompt terms explicitly; validate on known sessions |
| Pub/Sub requeue for SAM3 wait | Using Pub/Sub native retry counting as the sole termination mechanism | Embed attempt counter in message payload; set wall-clock deadline in message metadata |
| Pub/Sub dead-letter topic | Assuming DLT is active without verifying IAM permissions | Test DLT routing with a synthetic message that always NACKs before production deployment |
| Firebase Storage image I/O | Loading full-resolution image into GPU memory for every inference | Resize to inference resolution before GPU transfer; composite at original resolution after |
| GGUF model loading | Assuming GGUF device placement matches pipeline device_map | Specify device explicitly in GGUF loader; the existing CPU offload fix (commit 8f6a386) solves one instance but the pattern can recur on model swap |

---

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Model CPU offload with large images | Inference takes 90-120s instead of expected 30s because activations are constantly paged to CPU | Profile with `torch.cuda.memory_summary()` during inference; identify offload frequency | Any time image resolution increases or batch size > 1 |
| CUDA graph memory pinning with torch.compile | OOM errors or reduced effective VRAM for inference | Measure peak VRAM with and without compile; disable CUDA graphs if headroom < 2GB | T4 16GB with Q8_0 model (borderline headroom) |
| Thundering herd on SAM3 recovery | All requeued jobs arrive simultaneously when SAM3 comes back online, saturating inpaint worker | Add ±50% jitter to all requeue delays | Any SAM3 outage > 30 seconds affecting multiple concurrent jobs |
| Mask preprocessing on full-resolution image | CPU bottleneck in morphological operations on 4K images before GPU transfer | Apply morphological ops at inference resolution (typically 1024px), not original resolution | Images larger than 2048px on one side |
| Pub/Sub ack deadline timeout during long inference | Message redelivered mid-inference, causing duplicate processing | Extend ack deadline before starting inference (modifyAckDeadline); set to 120s minimum | Any inference taking > 60s (current baseline) |

---

## "Looks Done But Isn't" Checklist

- [ ] **SAM3 requeue termination:** Verify a message that never gets SAM3 labels eventually reaches the dead-letter topic (not just loops forever) — test by disabling SAM3 during a job submission
- [ ] **Mask preprocessing:** After morphological cleanup, verify no single-pixel specks remain and no erosion has exposed the object boundary — visual diff overlay check
- [ ] **Guidance scale calibration:** After any model change, confirm guidance was re-calibrated specifically for that model — do not assume transfer from FLUX.1-Fill-dev
- [ ] **Pipeline class match:** Confirm the pipeline class (FluxFillPipeline vs StableDiffusionInpaintPipeline) matches the loaded model's expected conditioning format — mismatches produce silent wrong outputs
- [ ] **Q8_0 baseline:** Confirm a Q8_0 quality baseline run exists for the known-failing session before attributing quality problems to prompt or parameters
- [ ] **Dead-letter IAM:** Verify Pub/Sub service account has publisher role on the dead-letter topic — without this, delivery attempt counts are not tracked

---

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Guidance scale mismatch after model swap | LOW | Re-run parameter sweep (guidance x steps grid) on known sessions; takes < 4 hours |
| Pub/Sub infinite loop discovered in production | MEDIUM | Pause subscription immediately; drain queue with jq filter to identify looping message IDs; manually NACK beyond threshold; deploy termination fix; resume |
| Over-specified prompt producing wrong fills | LOW | Reduce prompt to surface noun only; re-run same sessions for comparison; 1-2 hours |
| torch.compile silently not running | LOW | Add explicit compilation verification; switch to xformers if T4 limitations confirmed |
| Mask erosion exposed object boundary | MEDIUM | Rebuild mask preprocessing with dilation-first approach; re-validate on test sessions |
| Q4 quality loss misdiagnosed as prompt failure | MEDIUM | Run Q8_0 on failing sessions; if fixed, evaluate VRAM headroom and upgrade quantization level |

---

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| Guidance scale mismatch on model swap | Model evaluation — first task | Calibration grid run on session D1A406F5 before any quality comparison |
| Pub/Sub requeue infinite loop | SAM3 integration design — before coding | Integration test: submit job with SAM3 disabled, verify message reaches dead-letter topic |
| Prompt over-specification | Prompt engineering — specificity experiment first | A/B test: 1-word vs. 5-word prompt on identical sessions; choose based on quality, not intuition |
| torch.compile silent fallback on T4 | Inference optimization — validation step before benchmarking | `torch._dynamo.explain()` output shows successful compilation, not eager fallback |
| Mask preprocessing bias | Mask cleanup — visual validation step | Overlay diff: pre-processing mask vs. post-processing mask; check for boundary erosion |
| Model swap without full pipeline test | Model evaluation — use production code path | Run candidate model through complete service code path, not standalone notebook |
| Q4 quality loss masquerading as prompt failure | Baseline quality audit — before optimization begins | Q4 vs. Q8_0 comparison on session D1A406F5; document delta |

---

## Sources

- FLUX.1-Fill-dev model card (official, Hugging Face): https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev — guidance_scale=30 documented; known limitations (color shifts, edge artifacts, prompt following variability) confirmed HIGH confidence
- Google Cloud Pub/Sub dead-letter topics official docs: https://docs.cloud.google.com/pubsub/docs/dead-letter-topics — max_delivery_attempts range 5-100, approximate counting, IAM requirement confirmed HIGH confidence
- AUTOMATIC1111/ComfyUI community: https://github.com/comfyanonymous/ComfyUI/issues/6765 — bad quality from FLUX Fill without sufficient steps, MEDIUM confidence
- Stable Diffusion Art FLUX Fill guide: https://stable-diffusion-art.com/flux1-fill-inpaint/ — denoising strength and steps recommendations, MEDIUM confidence
- Aligned Stable Inpainting (2025 paper): https://arxiv.org/html/2601.15368 — mask dilation 5-20px to eliminate residual object shape bias, MEDIUM confidence
- GGUF quantization guide (2025): https://apatero.com/blog/gguf-quantized-models-complete-guide-2025 — Q4 visible quality loss for FLUX, Q8_0 optimal balance, MEDIUM confidence (WebSearch)
- PyTorch forums T4/Turing torch.compile: https://discuss.pytorch.org/t/pytorch2-0-error-on-t4/177133 — Flash Attention requires sm_80+; bfloat16 not supported on T4, MEDIUM confidence
- Cloud Pub/Sub infinite loop pattern: https://medium.com/@kedarnath93/gcp-cloud-pub-sub-repeatedly-triggers-the-cloud-run-services-48f547c177da — Cloud Run + Pub/Sub repeat trigger causes, MEDIUM confidence
- Inpainting prompt guide (getimg.ai): https://getimg.ai/guides/guide-to-writing-prompts-for-inpainting — prompt specificity and context derivation best practices, LOW confidence (single source)
- Mask Consistency Regularization paper: https://arxiv.org/html/2509.10259v1 — mask shape bias causes model to regenerate masked object, MEDIUM confidence
- GCP Pub/Sub subscription retry policy: https://docs.cloud.google.com/pubsub/docs/subscription-retry-policy — exponential backoff configuration, HIGH confidence

---
*Pitfalls research for: inpainting optimization — T4 GPU / FLUX.1-Fill-dev / async SAM3 / Pub/Sub*
*Researched: 2026-02-21*
