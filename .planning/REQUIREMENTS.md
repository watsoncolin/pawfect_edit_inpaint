# Requirements: Pawfect Edit Inpaint — Speed & Quality Improvement

**Defined:** 2026-02-21
**Core Value:** Inpainted areas must seamlessly match the surrounding background — users should not be able to tell something was removed.

## v1 Requirements

Requirements for this milestone. Each maps to roadmap phases.

### Baseline & Validation

- [ ] **BASE-01**: Establish baseline metrics — run current pipeline on reference session (D1A406F5) and record inference time, output quality
- [ ] **BASE-02**: Compare Q4_0 vs Q5_K_S vs Q8_0 quantization on reference session for quality and speed
- [ ] **BASE-03**: Run guidance scale calibration grid (test values 2, 4, 10, 20, 30) on reference session
- [ ] **BASE-04**: Verify torch.compile viability on L4 (sm_89) — confirm compilation actually runs, not silent fallback

### Inference Optimization

- [ ] **INFER-01**: Reduce inference steps from 28 to optimal value (target ~20) validated by baseline testing
- [ ] **INFER-02**: Set guidance scale to empirically-determined optimal value for background fill
- [ ] **INFER-03**: Upgrade quantization from Q4_0 to best-performing variant (Q5_K_S or Q8_0) based on baseline results
- [ ] **INFER-04**: Apply torch.compile regional compilation to DiT transformer blocks (if BASE-04 confirms viability)
- [ ] **INFER-05**: End-to-end inference time under 30 seconds on L4

### Mask Preprocessing

- [ ] **MASK-01**: Remove mask artifacts (scattered isolated white pixels) via morphological operations before inference
- [ ] **MASK-02**: Apply mask dilation (8-16px) to extend mask beyond object edge for cleaner fill

### Prompt & Quality

- [ ] **QUAL-01**: Replace static prompt with context-aware prompt derived from scene analysis
- [ ] **QUAL-02**: Integrate Florence-2-base to generate surface description from image region surrounding the mask (~1-2s overhead)
- [ ] **QUAL-03**: Build prompt construction pipeline: scene labels + Florence-2 description → inpainting prompt
- [ ] **QUAL-04**: Inpainted areas match surrounding background texture and color (no hallucinated objects)

### SAM3 Integration

- [ ] **SAM3-01**: Extend SAM3 service to return surface/scene labels for the area surrounding the inpaint mask
- [ ] **SAM3-02**: API server triggers SAM3 scene labeling in parallel with Pub/Sub enqueue (fan-out)
- [ ] **SAM3-03**: Store scene labels in Firestore session document (`sceneLabels` field)
- [ ] **SAM3-04**: Inpaint service reads scene labels from Firestore and uses them for prompt construction
- [ ] **SAM3-05**: Graceful fallback when SAM3 labels are not available (use Florence-2 only or empty prompt)

## v2 Requirements

### Quality Refinement

- **QUAL-V2-01**: Harmony score quality validation — re-inpaint if output scores below threshold
- **QUAL-V2-02**: Soft boundary compositing (feathered blend at mask edges)
- **QUAL-V2-03**: Boundary img2img refinement pass if seams persist

### Architecture

- **ARCH-V2-01**: Model registry pattern for clean model switching
- **ARCH-V2-02**: Cloud Run traffic splitting for A/B model evaluation
- **ARCH-V2-03**: Requeue with explicit termination contract (wall-clock deadline + embedded counter)

### Advanced Optimization

- **OPT-V2-01**: torchao int8 weight-only quantization (alternative to GGUF path)
- **OPT-V2-02**: ASUKA-style MAE prior integration if hallucinations persist after prompt/guidance fixes

## Out of Scope

| Feature | Reason |
|---------|--------|
| GPU upgrade beyond L4 | Optimize within current L4 hardware |
| Model replacement (SD 1.5, SDXL) | FLUX Fill quality is superior for background fill; optimize within FLUX |
| Real-time/streaming inpainting | Async batch processing is sufficient |
| On-device inpainting changes | Small areas handled on iOS; this is server-side only |
| SAM2 mask generation changes | Mask creation stays on device |
| FP8 quantization | Now viable on L4 (sm_89) — evaluate in future if needed |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| BASE-01 | Phase 0 | Pending |
| BASE-02 | Phase 0 | Pending |
| BASE-03 | Phase 0 | Pending |
| BASE-04 | Phase 0 | Pending |
| INFER-01 | Phase 1 | Pending |
| INFER-02 | Phase 1 | Pending |
| INFER-03 | Phase 1 | Pending |
| INFER-04 | Phase 4 | Pending |
| INFER-05 | Phase 4 | Pending |
| MASK-01 | Phase 1 | Pending |
| MASK-02 | Phase 1 | Pending |
| QUAL-01 | Phase 3 | Pending |
| QUAL-02 | Phase 3 | Pending |
| QUAL-03 | Phase 3 | Pending |
| QUAL-04 | Phase 3 | Pending |
| SAM3-01 | Phase 2 | Pending |
| SAM3-02 | Phase 2 | Pending |
| SAM3-03 | Phase 2 | Pending |
| SAM3-04 | Phase 2 | Pending |
| SAM3-05 | Phase 2 | Pending |

**Coverage:**
- v1 requirements: 20 total
- Mapped to phases: 20
- Unmapped: 0

---
*Requirements defined: 2026-02-21*
*Last updated: 2026-02-21 — traceability filled after roadmap creation*
