---
phase: 00-baseline-audit
plan: 01
subsystem: api
tags: [fastapi, diffusers, flux, gguf, firebase, torch, dynamo, cuda]

# Dependency graph
requires: []
provides:
  - POST /audit endpoint accepting {user_id, session_id, run_id}
  - audit_runner.run_audit() — 3-quant x 5-guidance matrix with dual CUDA/wall-clock timing
  - Firebase Storage signed URL helpers (generate_signed_url, upload_and_sign)
  - Markdown audit report uploaded to audit/{run_id}/REPORT.md with signed image URLs
  - torch.compile viability check via dynamo.explain() (graph count + break count + reasons)
affects:
  - 00-baseline-audit (phase execution — run this endpoint to collect data)
  - 01-quality (quant + guidance selection from report informs Phase 1 config)
  - 04-speed (torch.compile break count from report informs Phase 4 go/no-go)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Synchronous FastAPI route handler (def not async) for long-running GPU tasks — FastAPI auto-runs in thread pool"
    - "Dual CUDA timing: wall-clock time.time() + CUDA Events start_evt.elapsed_time(end_evt) per inference"
    - "VRAM cleanup pattern: del transformer; del pipe; gc.collect(); torch.cuda.empty_cache() between quant groups"
    - "hf_hub_download returns local cache path — avoids re-download on repeated audit runs"
    - "dynamo.explain()() pattern for torch.compile graph break analysis without requiring compile() call"

key-files:
  created:
    - app/services/audit_runner.py
    - app/routers/audit.py
  modified:
    - app/services/firebase.py
    - app/main.py

key-decisions:
  - "Q5_K_M variant does not exist in YarvixPA repo — use Q5_K_S (8.29 GB) instead"
  - "Audit route is synchronous (def not async) — appropriate for long-running GPU task, FastAPI handles thread pool"
  - "torch.compile check runs after the 15-inference matrix to avoid polluting timing results"
  - "dynamo.explain()() wraps a 1-step inference (not the bare transformer) — tests the full pipeline call graph"

patterns-established:
  - "Firebase signed URL: generate_signed_url(path, expiry_hours=72) via blob.generate_signed_url(version='v4')"
  - "Audit storage layout: audit/{run_id}/{quant_label}_gs{guidance:.0f}.png + audit/{run_id}/REPORT.md"
  - "Quant loading: hf_hub_download(repo_id=HF_REPO, filename=gguf_filename) -> FluxTransformer2DModel.from_single_file"

requirements-completed: [BASE-01, BASE-02, BASE-03, BASE-04]

# Metrics
duration: 1min
completed: 2026-02-22
---

# Phase 0 Plan 01: Baseline Audit Endpoint Summary

**POST /audit endpoint with 3-quant x 5-guidance matrix runner, CUDA event timing, VRAM cleanup, dynamo.explain() compile check, and signed-URL Markdown report uploaded to Firebase Storage**

## Performance

- **Duration:** 1 min
- **Started:** 2026-02-22T02:53:04Z
- **Completed:** 2026-02-22T02:54:26Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- Implemented `run_audit()` in audit_runner.py: outer loop over Q4_0/Q5_K_S/Q8_0, inner loop over guidance=[2,4,10,20,30], 15 total inferences
- Dual timing per inference: wall-clock (time.time) + CUDA events (start_evt.elapsed_time(end_evt) after cuda.synchronize)
- VRAM freed between quantization groups via del + gc.collect + cuda.empty_cache
- torch.compile viability assessed via dynamo.explain()() on Q4_0 pipeline after the matrix (graph_count, graph_break_count, break_reasons capped at 5)
- Markdown report with GPU identity, timing table, signed image links, and compile section uploaded to audit/{run_id}/REPORT.md
- Added generate_signed_url() and upload_and_sign() helpers to firebase.py
- POST /audit endpoint registered via audit router in main.py; synchronous def handler for GPU task

## Task Commits

Each task was committed atomically:

1. **Task 1: Add generate_signed_url/upload_and_sign to firebase.py; register audit router in main.py** - `98b4430` (feat)
2. **Task 2: Implement audit_runner.py and POST /audit endpoint** - `590d86c` (feat)

**Plan metadata:** (docs commit to follow)

## Files Created/Modified
- `app/services/audit_runner.py` - Core audit engine: GPU identity, session asset download, quant x guidance matrix, dual timing, VRAM cleanup, dynamo.explain(), report generation
- `app/routers/audit.py` - FastAPI router exposing POST /audit, auto-generates run_id if not provided
- `app/services/firebase.py` - Added import datetime, generate_signed_url(), upload_and_sign()
- `app/main.py` - Imported and registered audit_router after inpaint_router

## Decisions Made
- Q5_K_M does not exist in the YarvixPA/FLUX.1-Fill-dev-GGUF repo; Q5_K_S (8.29 GB) is used instead — consistent with the research doc
- Route handler is synchronous (`def`, not `async def`) — FastAPI automatically runs it in a thread pool, correct for long-running GPU work
- torch.compile check is isolated after the 15-inference matrix to avoid cache effects or JIT warmup polluting timing measurements
- dynamo.explain()() wraps the full pipeline call (1 step) rather than the bare transformer module, giving a more realistic break count for the actual inference graph

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None — no external service configuration required beyond existing Firebase credentials already in the environment.

## Next Phase Readiness

The audit endpoint is ready to invoke. To collect the Phase 0 data set:

```bash
curl --max-time 7200 -X POST https://<service-url>/audit \
  -H "Content-Type: application/json" \
  -d '{"user_id": "<uid>", "session_id": "D1A406F5"}'
```

The returned `report_url` signed link contains the Markdown report with the timing table and torch.compile assessment needed to fill in the "Recommended Settings for Phase 1" section.

Blockers remain as documented in STATE.md:
- Q4 vs Q8 hallucination contribution unconfirmed until audit runs
- Optimal guidance scale for fill mode unresolved (official 30 vs community 2-5)
- torch.compile viability on L4 sm_89 unconfirmed until audit runs

---
*Phase: 00-baseline-audit*
*Completed: 2026-02-22*

## Self-Check: PASSED

- app/services/audit_runner.py: FOUND
- app/routers/audit.py: FOUND
- app/services/firebase.py: FOUND
- app/main.py: FOUND
- .planning/phases/00-baseline-audit/00-01-SUMMARY.md: FOUND
- Task 1 commit 98b4430: FOUND
- Task 2 commit 590d86c: FOUND
