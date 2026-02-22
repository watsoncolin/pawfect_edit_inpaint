# Phase 0: Baseline Audit - Context

**Gathered:** 2026-02-21
**Status:** Ready for planning

<domain>
## Phase Boundary

Diagnostic testing on the production T4 to confirm root causes of hallucinations and slow inference. Compares quantization variants, guidance scale values, and torch.compile viability. Produces a report with visual side-by-side comparisons and timing data. Does NOT deploy any changes — Phase 1 consumes the audit results.

</domain>

<decisions>
## Implementation Decisions

### Test methodology
- Build a single "audit mode" deployment with a dedicated endpoint that cycles through all parameter variations
- Each variation produces an output image — no automated metrics, visual side-by-side comparison only
- Test on production Cloud Run T4 hardware for real timing numbers
- Guidance scale grid: test values 2, 4, 10, 20, 30 (covers full range researchers disagreed on)
- Quantization: compare Q4_0 vs Q5_K_M vs Q8_0 (availability permitting)
- torch.compile: verify compilation actually runs via torch._dynamo.explain(), not assumed

### Reference sessions
- Test on session D1A406F5 only (dog on grass, leash handle removal)
- Expected correct output: matching grass with fallen leaves where hand/leash handle was — nothing else
- Single session is sufficient for the diagnostic audit

### Quality thresholds
- Both hallucinated objects AND color/texture mismatch are failures
- Subtle seams at mask boundary are acceptable — doesn't need to be pixel-perfect invisible
- User eyeballs all outputs and picks the best-looking overall — no formal scoring

### Audit deliverable
- All test output images uploaded to Firebase Storage at a known path
- Markdown report with signed URLs to each image + timing data for each variation
- User reviews report and manually approves which settings to use in Phase 1
- No auto-deployment — audit is purely diagnostic

### Claude's Discretion
- Audit endpoint API design and payload structure
- How to organize test images in Firebase Storage
- Report format and structure details
- Whether to test step count variations alongside guidance scale, or keep that for Phase 1

</decisions>

<specifics>
## Specific Ideas

- The audit should make it dead simple to compare outputs — a report where you can see image A vs B vs C with the settings labeled
- Timing data per variation is important to know if Q8_0 is actually faster than Q4_0 as research suggested

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 00-baseline-audit*
*Context gathered: 2026-02-21*
