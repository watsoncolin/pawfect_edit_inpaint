# Project Research Summary

**Project:** Pawfect Edit Inpaint — Speed and Quality Optimization
**Domain:** Async GPU inference service — AI background fill / object removal for pet photography
**Researched:** 2026-02-21
**Confidence:** MEDIUM

## Executive Summary

Pawfect Edit's inpainting service uses FLUX.1-Fill-dev (GGUF Q4 quantized) on a T4 GPU with a Pub/Sub async queue. Two distinct problems compound each other: (1) hallucinations — the model generates wrong content in the removed area — caused entirely by a generic static prompt and mismatched guidance scale; (2) slow inference — 60-120s per job — caused by too many steps (28 vs optimal 20), wrong guidance value, and CPU offload overhead. Both problems are fixable without replacing the model. Keep FLUX.1-Fill-dev. The quality advantage over SD 1.5 or SDXL is decisive for seamless background fill, and no alternative model can match it at comparable complexity.

The recommended approach combines four independent improvements: fix inference parameters (20 steps, guidance=30 for FLUX Fill's distilled guidance behavior — or 2-5 for pure background fill context derivation — requires calibration experiment to resolve conflicting findings), add Florence-2-base for context-aware prompt generation from surrounding pixels (~1-2s on T4, MIT license), apply morphological mask cleanup to eliminate speck artifacts before inference, and evaluate torch.compile on the transformer only (with a verification step since T4's sm_75 architecture may silently fall back to eager mode). A critical pre-work step — running Q8_0 GGUF on the known-failing session D1A406F5 before any other optimization — will determine whether Q4 quantization is contributing to hallucinations and scoping all subsequent work correctly.

The main architectural addition is SAM3 scene labeling integrated as a fire-and-forget pre-computation step: when an inpaint job is requested, the API server triggers SAM3 in parallel with the Pub/Sub enqueue. SAM3 writes surface labels to Firestore; the inpaint worker reads them before inference. If labels haven't arrived, the worker requeues via HTTP 500 NACK (up to 3 attempts, 90s window), then falls back to Florence-2 direct captioning on the surrounding crop. The primary pitfall of this pattern — an unbounded requeue loop — must be addressed with explicit termination conditions before any requeue code is written.

## Key Findings

### Recommended Stack

The existing model and infrastructure are sound; the problem is configuration and pre/post-processing gaps. FLUX.1-Fill-dev with Q5_K_M GGUF (upgrade from current Q4_0) should remain the core. Florence-2-base (0.23B) is the right tool for context-aware prompt generation: faster than BLIP-2, MIT licensed, T4-verified, and purpose-built for captioning tasks. SAM3 (Nov 2025) adds text-prompted segmentation to the existing SAM infrastructure, making background region isolation architecturally clean.

**Core technologies:**
- FLUX.1-Fill-dev + Q5_K_M GGUF: Primary inpainting model — keep it; Q5 delivers near-FP16 quality at comparable speed to Q4 on VRAM-bound T4
- Florence-2-base: Context-aware surface captioning — 0.23B model, 1-2s on T4, eliminates generic prompt hallucination problem
- SAM3: Scene labeling for background surface classification — already in infra, extend to return surface labels to Firestore
- torch.compile (regional, transformer-only): Speed optimization — must verify actual compilation runs on T4 sm_75 before relying on it
- OpenCV (headless): Morphological mask cleanup — erosion/dilation to remove speck artifacts pre-inference
- torchao int8wo: Required if torch.compile is adopted (GGUF quantization is incompatible with torch.compile graph tracing)

**Critical version requirements:**
- diffusers >= 0.32.0 for FluxFillPipeline GGUF + VAE slicing/tiling support
- torch >= 2.1.0 for torch.compile reduce-overhead mode
- transformers >= 4.47.0 for Florence-2 support (already pinned)

**Estimated time budget after all optimizations:** ~28-44s end-to-end (from 60-120s baseline). Confidence is LOW on exact numbers — T4-specific FLUX Fill benchmarks are not publicly available. Profiling is mandatory before declaring the <30s target met.

### Expected Features

**Must have (table stakes — this milestone):**
- Mask artifact cleanup (morphological open) — remove scattered specks before inference; cheap, high impact
- Mask dilation (8-12px) — expand mask to capture edge fringing; prevents visible seam lines
- Context-aware prompt from SAM3 labels — consume surface labels from SAM3 extension; core hallucination fix
- Empty-prompt fallback — when labels unavailable, send no prompt rather than "empty ground"; lets FLUX use surrounding pixels
- Guidance scale calibration — current value of 10 is wrong for Fill-dev; must run calibration experiment
- Inference step reduction (28 → 20) — validate quality parity at 20 steps before deploying
- Soft boundary compositing — Gaussian-blend boundary post-inference; low cost, visually significant

**Should have (add after v1 validation):**
- Mask quality assessment / rejection heuristics — reject masks >40% coverage or <50 connected pixels before wasting GPU time
- Florence-2 fallback when SAM3 labels absent — direct caption on surrounding crop; adds ~1-2s but avoids requeue
- Boundary img2img refinement pass — 0.2-0.35 denoising over mask boundary region only; add if seams persist

**Defer to v2+:**
- Inpainting Harmony Score quality gate — requires full second inference cycle; only viable after step reduction creates headroom
- Harmony score-guided selective re-inpainting — complex; needs IHS infrastructure first
- ASUKA/MAE hallucination suppression — high engineering cost against GGUF pipeline; only if prompt+guidance tuning fails
- Turbo LoRA step distillation (8 steps) — community evidence for general generation, not fill-specific; defer pending validation

**Anti-features (avoid despite intuitive appeal):**
- High mask blur (>16px) before inference — binarization breaks soft edges; use hard mask with post-inference soft compositing instead
- Multi-pass full re-inference — each pass costs 20-60s; targeted boundary refinement is the right tool
- Multiple candidate outputs — multiplies inference cost directly; not feasible on T4 budget
- Real-time streaming inpainting — T4 cannot achieve sub-5s; keep async Pub/Sub with progress indicator

### Architecture Approach

The architecture is an async Pub/Sub pipeline with a new fire-and-forget pre-computation branch. When an inpaint job arrives, the API server triggers SAM3 labeling in parallel (no await, 2s timeout, failure is silent and logged) while simultaneously enqueuing the Pub/Sub message. SAM3 writes `sceneLabels` to the existing Firestore session document. The inpaint worker reads labels on pickup; if absent, it requeues via HTTP 500 NACK for up to 3 attempts within 90s, then falls back to Florence-2 direct captioning. Pub/Sub exponential backoff naturally provides the wait interval (10-30s at attempt 2-3), so no custom delay infrastructure is needed. Labels stored in Firestore (not Pub/Sub payload) leverages existing read patterns with zero extra Firestore reads.

**Major components:**
1. SAM3 Service (extend) — run surface concept prompts, classify dominant background surface, write `sceneLabels` to Firestore
2. API Server (extend) — fire-and-forget SAM3 trigger in parallel with Pub/Sub enqueue; no await, 2s timeout
3. Inpaint Service (extend) — read labels, build context-aware prompt, apply mask cleanup, run optimized FLUX Fill inference
4. `prompt_builder.py` (new) — centralized, unit-testable label-to-prompt mapping; keeps prompt logic out of pipeline
5. `model_registry.py` (new) — MODEL_VARIANT env var selects model at startup; enables Cloud Run traffic splitting for A/B testing
6. Florence-2 integration (new) — fallback captioner when SAM3 labels unavailable; runs on surrounding crop, not masked region

**Build order (enforced by dependencies):**
1. SAM3 extension (labels must exist before integration can be tested)
2. API server fan-out (requires SAM3 endpoint)
3. Inpaint service label integration (requires phases 1+2 to be testable end-to-end)
4. Model registry (parallel-trackable, prerequisite for clean inference optimization)
5. Inference optimization / torch.compile (requires model registry structure; GGUF → torchao migration is a prerequisite for torch.compile)
6. Mask cleanup (independent; can parallel-track with any phase)

### Critical Pitfalls

1. **Pub/Sub requeue infinite loop** — SAM3 label wait creates a second retry path that bypasses the existing 5-attempt limit. Must embed explicit wall-clock deadline and attempt counter in message payload; never rely solely on Pub/Sub native counting. Test by submitting a job with SAM3 disabled and verifying it reaches the dead-letter topic.

2. **torch.compile silently falling back on T4 sm_75** — Flash Attention requires sm_80+; T4 (sm_75, Turing) may produce no speedup or suppressed Triton errors without failing loudly. Always verify compilation ran via `torch._dynamo.explain()`; do not build a performance baseline around compile until verified.

3. **Q4 quality loss misattributed to prompt/parameters** — Run Q8_0 GGUF on session D1A406F5 before any optimization work. If Q8_0 fixes the hallucination, the problem is quantization, not prompting. This 1-hour experiment scopes everything else.

4. **Prompt over-specification preventing context derivation** — Verbose SAM3-driven prompts ("patchy green and yellow grass with fallen oak leaves...") force the model to generate exactly the text, overriding self-attention from surrounding pixels. Prompts should name the surface category only (1-3 words). Run a specificity grid experiment (1-word vs. 5-word) before committing to the label schema.

5. **Guidance scale mismatch** — guidance=30 (model card example) produces flat, plastic fills; guidance=10 (current) is too high for context derivation. The correct range for background fill is 2-5; guidance=30 is for prompt-driven generation, not background fill. Never transfer guidance values across model architectures.

6. **Mask preprocessing erosion exposing object boundaries** — Overly aggressive erosion removes mask coverage near the boundary, leaving object-edge pixels unmasked that the model then regenerates. Use dilation-first approach; validate with visual diff overlay before inference.

## Implications for Roadmap

Based on the dependency graph from architecture research and the pitfall-to-phase mapping from pitfalls research, the following phase structure is recommended:

### Phase 0: Baseline Audit
**Rationale:** Three open questions scope all subsequent work. Answering them first prevents building optimizations on a misdiagnosed foundation. This is a research/validation phase, not a feature phase.
**Delivers:** Confirmed diagnosis of hallucination root cause; guidance scale range for Fill-dev; torch.compile viability verdict on T4
**Experiments:**
- Q4 vs Q8_0 GGUF on session D1A406F5 (isolates quantization contribution to hallucinations)
- Guidance scale calibration grid (1, 3, 5, 7, 10, 30) for background fill mode
- torch.compile smoke test on T4 with `torch._dynamo.explain()` to confirm actual compilation
**Avoids:** Chasing prompt/parameter fixes that are actually quantization quality loss; building performance assumptions on compile that doesn't run

### Phase 1: Quick Wins — Parameters and Mask Preprocessing
**Rationale:** These are one-line changes or simple additions with high impact and no external dependencies. Ship them before any architectural work to establish a better baseline.
**Delivers:** Measurable inference time reduction; elimination of mask speck artifacts; improved baseline fill quality
**Addresses:** Inference step reduction (28→20); guidance scale correction; mask artifact cleanup; mask dilation (8-12px); soft boundary compositing; empty-prompt fallback
**Avoids:** Guidance scale mismatch pitfall; mask preprocessing erosion bias pitfall
**Stack:** OpenCV headless (new dep); parameter constant changes only

### Phase 2: SAM3 Scene Labeling Integration
**Rationale:** This is the core hallucination fix and the highest-dependency new feature. Must build in order: SAM3 extension → API server fan-out → inpaint service integration. Cannot test end-to-end until all three are done.
**Delivers:** Context-aware background fill prompts from scene labels; fire-and-forget pre-computation pattern; graceful label-wait requeue with termination conditions
**Addresses:** Context-aware prompt from SAM3 labels; empty-prompt fallback refinement; label-wait requeue logic
**Implements:** SAM3 service extension; API server parallel fan-out; `prompt_builder.py`; Firestore `sceneLabels` field
**Avoids:** Pub/Sub requeue infinite loop (termination contract must be first-class requirement, not afterthought); prompt over-specification
**Research flag:** Needs deeper planning — SAM3 label vocabulary must be defined before integration; label-to-prompt mapping is project-specific and has no prior art to copy

### Phase 3: Model Registry and Structure
**Rationale:** Clean code structure prerequisite for inference optimization. Enables Cloud Run traffic splitting for A/B evaluation. Low risk, parallel-trackable with Phase 2.
**Delivers:** MODEL_VARIANT env var selection; `model_registry.py` with InpaintModel protocol; refactored `flux_fill.py` module; Cloud Run traffic split capability
**Uses:** Existing Python Protocol pattern; Cloud Run revision traffic splitting (no new infra)
**Avoids:** Model swap without full pipeline test pitfall (registry enforces consistent pipeline class selection)

### Phase 4: Inference Optimization
**Rationale:** Most technically uncertain phase. Requires Phase 3 structure. GGUF → torchao migration is a prerequisite for torch.compile. Benchmark results will determine whether torch.compile provides meaningful speedup on T4 before committing to the migration cost.
**Delivers:** Quantified inference time improvement; decision on GGUF vs torchao path; torch.compile integration (if viable) or xformers alternative
**Uses:** torchao int8wo + torch.compile (if T4 verification passes); VAE slicing/tiling; diffusers >= 0.32.0
**Avoids:** torch.compile silent fallback pitfall; GGUF + torch.compile incompatibility anti-pattern
**Research flag:** Needs deeper planning — torchao + torch.compile on T4 sm_75 is less documented than H100 benchmarks; T4-specific FLUX Fill benchmarks don't exist publicly; profiling results will determine actual path

### Phase 5: Florence-2 Fallback and Polish
**Rationale:** Adds resilience when SAM3 labels are unavailable, validates v1.x feature set, and introduces mask quality gates. Lower priority than the core hallucination and speed fixes.
**Delivers:** Florence-2 direct captioning fallback; mask quality assessment / rejection heuristics; boundary img2img refinement (if seams persist)
**Addresses:** VLM fallback for missing labels; mask pathology rejection before GPU time is wasted
**Avoids:** Adding Florence-2 to the happy path (latency concern); VLM over-use on fast path

### Phase Ordering Rationale

- Phase 0 before everything: the Q4 vs Q8_0 audit and guidance calibration determine the correct targets for all subsequent phases; without them, optimization work may chase the wrong problem.
- Phase 1 before Phase 2: parameter and mask fixes are independent and fast to ship; they establish a better baseline for measuring the quality improvement from SAM3 labels.
- Phase 2 before Phase 4: SAM3 integration proves the architecture pattern; inference optimization changes the model loading path and should land into stable, tested code.
- Phase 3 parallel with Phase 2: model registry is structurally independent; can be developed in parallel and merged before Phase 4.
- Phase 4 last for inference: GGUF → torchao migration is a breaking change to model loading; it should land after the labeling pipeline is stable and tested.

### Research Flags

Phases likely needing `/gsd:research-phase` during planning:
- **Phase 2 (SAM3 Integration):** SAM3 label vocabulary and label-to-prompt mapping are project-specific. Need to define the surface category set, confidence thresholds, and fallback behavior. No reference implementation exists.
- **Phase 4 (Inference Optimization):** torch.compile behavior on T4 sm_75 is underdocumented. The GGUF → torchao migration path needs validation against the exact model checkpoint. Profiling results from Phase 0 will drive decisions here.

Phases with standard patterns (skip research-phase):
- **Phase 1 (Parameters + Mask Preprocessing):** Well-documented; OpenCV morphological ops and diffusers parameter changes are standard. Results from Phase 0 calibration audit provide the target values.
- **Phase 3 (Model Registry):** Standard Python Protocol pattern + Cloud Run traffic splitting; both are well-documented with official guides.
- **Phase 5 (Florence-2 + Polish):** Florence-2 integration follows standard transformers patterns; mask quality heuristics are simple OpenCV operations.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | MEDIUM-HIGH | Core model choice (keep FLUX.1-Fill-dev) is HIGH confidence. Specific speedup estimates on T4 are LOW confidence — no public T4 + FLUX Fill benchmarks exist; estimates are cross-source triangulation |
| Features | MEDIUM | Must-have features are well-scoped. Guidance scale target is MEDIUM — conflicting signals between model card (30) and community fill-mode best practices (2-5); calibration experiment required |
| Architecture | MEDIUM-HIGH | Existing components (Pub/Sub, Firestore, Firebase Storage) are HIGH confidence. New SAM3 integration pattern is MEDIUM — fire-and-forget design is solid but requeue termination needs careful implementation |
| Pitfalls | MEDIUM | Pub/Sub infinite loop and torch.compile fallback pitfalls are HIGH confidence (specific evidence from codebase and PyTorch forums). Guidance scale mismatch is MEDIUM confidence (model card vs community sources conflict) |

**Overall confidence:** MEDIUM

### Gaps to Address

- **Guidance scale target for FLUX.1-Fill-dev background fill:** Model card says 30; community fill-mode guidance says 2-5. These are not reconcilable without an experiment. Phase 0 calibration grid resolves this before Phase 1 ships.
- **torch.compile viability on T4 sm_75:** No public benchmarks for FLUX Fill + torch.compile on T4. Phase 0 smoke test with `torch._dynamo.explain()` will confirm or reject this optimization path before Phase 4 planning.
- **Q5_K_M GGUF availability for FLUX.1-Fill-dev:** STACK.md notes the HuggingFace URL needs verification. Confirm the file exists in the YarvixPA or city96 repo before committing to Q5 upgrade.
- **SAM3 label vocabulary coverage:** The surface concept set (grass, concrete, wood, sand, gravel, leaves, tile, water) covers common outdoor pet photo backgrounds. Indoor scenes (carpet, hardwood) and unusual surfaces are gaps — define the fallback behavior for uncovered categories in Phase 2.
- **torchao int8wo VRAM footprint on T4:** ARCHITECTURE.md notes this as "manageable" but lacks specific numbers. Benchmark at Docker build time before committing to torchao migration in Phase 4.
- **Pub/Sub ack deadline during long inference:** With 60-120s current inference times, the default 60s ack deadline causes message redelivery mid-inference. Must extend ack deadline (modifyAckDeadline to 120s+) before starting inference — this is an existing risk that should be verified and fixed in Phase 1 or Phase 2.

## Sources

### Primary (HIGH confidence)
- HuggingFace Diffusers FLUX documentation — FluxFillPipeline API, memory optimizations, VAE slicing/tiling
- Google Cloud Pub/Sub docs — retry policy, dead-letter topics, max_delivery_attempts, IAM requirements
- SAM3 GitHub (facebookresearch/sam3) and arXiv 2511.16719 — text-prompted segmentation capabilities
- Florence-2-base HuggingFace model card (microsoft/Florence-2-base) — 0.23B params, T4 support, MIT license
- Firebase Firestore docs — data structure best practices, document size constraints
- OpenCV morphological operations docs — mask preprocessing patterns
- torchao quantization in Diffusers (HuggingFace docs) — int8wo + torch.compile compatibility

### Secondary (MEDIUM confidence)
- FLUX.1-Fill-dev model card — guidance_scale=30 example; known limitations (color shifts, edge artifacts)
- PyTorch blog "torch.compile and Diffusers" (July 2025) — regional compile strategy, speedup benchmarks (H100, extrapolated to T4)
- Aligned Stable Inpainting (2025 paper) — mask dilation 5-20px to eliminate object shape bias
- Community consensus on FLUX steps: 20-30 steps optimal for Dev variant
- Q5_K_M vs Q4_0 quality comparison — multiple GGUF comparison articles
- ASUKA (CVPR 2025) — MAE-based hallucination suppression
- Harmony score-guided inpainting (2025 journal) — IHS quality gating approach
- PyTorch forums T4/Turing torch.compile — Flash Attention sm_80 requirement, bfloat16 T4 limitations

### Tertiary (LOW confidence)
- Community guidance scale threshold at 4.0 for contextual fill — needs project-specific validation via calibration experiment
- Flux Turbo LoRA compatibility with Fill pipeline — community blog posts, not formally validated
- SDXL vs FLUX speed comparison — 4090 hardware benchmarks, not T4

---
*Research completed: 2026-02-21*
*Ready for roadmap: yes*
