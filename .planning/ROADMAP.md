# Roadmap: Pawfect Edit Inpaint — Speed & Quality Improvement

## Overview

This milestone improves the inpaint service on two fronts: eliminating hallucinations (the model generates wrong content instead of matching background) and reducing inference time from 60-120s to under 30s on a T4 GPU. The work begins with a diagnostic audit to confirm the root causes, then ships independent quick wins (parameters, mask cleanup), then builds the SAM3-driven context-aware prompt pipeline, then adds Florence-2 as a resilient fallback and completes the prompt integration, and finally optimizes inference at the model level. Each phase delivers a measurable, verifiable improvement on the reference session (D1A406F5).

## Phases

**Phase Numbering:**
- Integer phases (0, 1, 2, 3, 4): Planned milestone work
- Decimal phases (1.1, 2.1): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 0: Baseline Audit** - Diagnose root causes before building anything
- [ ] **Phase 1: Parameters + Mask Preprocessing** - Ship quick wins with no external dependencies
- [ ] **Phase 2: SAM3 Scene Labeling Integration** - Wire scene labels into Firestore end-to-end
- [ ] **Phase 3: Context-Aware Prompt Pipeline** - Eliminate hallucinations using scene labels and Florence-2
- [ ] **Phase 4: Inference Optimization** - Push end-to-end time under 30s

## Phase Details

### Phase 0: Baseline Audit
**Goal**: Confirm the root cause of hallucinations and establish calibrated targets before any optimization work begins
**Depends on**: Nothing (first phase)
**Requirements**: BASE-01, BASE-02, BASE-03, BASE-04
**Success Criteria** (what must be TRUE):
  1. Baseline inference time and a side-by-side quality comparison of Q4_0 vs Q8_0 output on session D1A406F5 are recorded — the quantization contribution to hallucinations is confirmed or ruled out
  2. A guidance scale calibration grid has been run and the optimal value for background fill mode on FLUX.1-Fill-dev is identified and documented
  3. torch.compile viability on T4 (sm_75) is confirmed or rejected via `torch._dynamo.explain()` — not assumed
  4. A written audit report exists with specific numeric targets for steps, guidance scale, and quantization tier that Phase 1 will use
**Plans**: 2 plans

Plans:
- [x] 00-01-PLAN.md — Implement audit endpoint, parameter matrix runner, Firebase signed URL helper
- [ ] 00-02-PLAN.md — Update Cloud Run timeout + human: deploy, run audit, review report, record approved settings

### Phase 1: Parameters + Mask Preprocessing
**Goal**: Measurably reduce inference time and eliminate mask speck artifacts using parameter changes and mask cleanup — no new external dependencies
**Depends on**: Phase 0
**Requirements**: INFER-01, INFER-02, INFER-03, MASK-01, MASK-02
**Success Criteria** (what must be TRUE):
  1. Inference step count is reduced from 28 to the Phase-0-validated optimal (target ~20) and the change is live in the service
  2. Guidance scale is set to the Phase-0-calibrated value and deployed
  3. Quantization is upgraded from Q4_0 to the best-performing variant identified in Phase 0 (Q5_K_M or Q8_0)
  4. A morphological mask cleanup pass removes scattered isolated white pixels before inference — visible on any mask with speck artifacts
  5. Mask dilation (8-16px) is applied before inference — object edges no longer appear as hard seams in output
**Plans**: TBD

### Phase 2: SAM3 Scene Labeling Integration
**Goal**: Surface/scene labels for the inpaint area are available in Firestore before the inpaint worker runs, with graceful fallback when they are not
**Depends on**: Phase 1
**Requirements**: SAM3-01, SAM3-02, SAM3-03, SAM3-04, SAM3-05
**Success Criteria** (what must be TRUE):
  1. SAM3 service returns surface/scene labels (e.g., "grass", "concrete", "leaves") for the region surrounding the inpaint mask — not just binary mask output
  2. When an inpaint job is requested, the API server fires SAM3 labeling in parallel with Pub/Sub enqueue — the inpaint enqueue is not blocked by SAM3
  3. Scene labels are written to the Firestore session document (`sceneLabels` field) and the inpaint service reads them on pickup
  4. The inpaint service requeues via NACK when labels are not yet available, with an explicit termination contract (attempt counter + wall-clock deadline) that prevents infinite requeue loops
  5. When SAM3 labels remain unavailable after the requeue window, the service falls back gracefully (Florence-2 or empty prompt) without failing the job
**Plans**: TBD

### Phase 3: Context-Aware Prompt Pipeline
**Goal**: The inpaint prompt describes the actual background surface being filled — inpainted areas match surrounding texture and color with no hallucinated objects
**Depends on**: Phase 2
**Requirements**: QUAL-01, QUAL-02, QUAL-03, QUAL-04
**Success Criteria** (what must be TRUE):
  1. The static "empty ground, nothing here" prompt is replaced with a dynamically constructed prompt derived from SAM3 labels and/or Florence-2 scene captioning
  2. Florence-2-base is integrated to generate a surface description from the image region surrounding the mask (~1-2s overhead) — used when SAM3 labels are absent
  3. A `prompt_builder.py` module maps scene labels + Florence-2 captions to a final inpainting prompt — prompt logic is centralized and unit-testable, separate from the inference pipeline
  4. On the reference session (D1A406F5 — dog on grass with fallen leaves), the inpainted area fills with matching grass/leaves texture and no wrong object is hallucinated into the frame
**Plans**: TBD

### Phase 4: Inference Optimization
**Goal**: End-to-end inpaint time is under 30 seconds on T4, with torch.compile applied to the transformer if Phase 0 confirmed it is viable
**Depends on**: Phase 3
**Requirements**: INFER-04, INFER-05
**Success Criteria** (what must be TRUE):
  1. torch.compile regional compilation is applied to the DiT transformer blocks (conditional on Phase 0 viability confirmation) — compilation is verified to actually run via `torch._dynamo.explain()`, not assumed
  2. End-to-end inpaint time on T4 is measured at under 30 seconds on the reference session, including all pre/post-processing steps
  3. If torch.compile is not viable on T4, an alternative optimization (e.g., xformers attention, VAE tiling tuning) is documented and applied in its place
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 0 → 1 → 2 → 3 → 4

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 0. Baseline Audit | 1/2 | In progress | - |
| 1. Parameters + Mask Preprocessing | 0/TBD | Not started | - |
| 2. SAM3 Scene Labeling Integration | 0/TBD | Not started | - |
| 3. Context-Aware Prompt Pipeline | 0/TBD | Not started | - |
| 4. Inference Optimization | 0/TBD | Not started | - |
