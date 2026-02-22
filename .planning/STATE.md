# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-21)

**Core value:** Inpainted areas must seamlessly match the surrounding background — users should not be able to tell something was removed.
**Current focus:** Phase 0 — Baseline Audit

## Current Position

Phase: 0 of 4 (Baseline Audit)
Plan: 1 of TBD in current phase (00-01 complete)
Status: In progress — audit endpoint built, awaiting execution
Last activity: 2026-02-22 — Plan 00-01 completed

Progress: [█░░░░░░░░░] 5%

## Performance Metrics

**Velocity:**
- Total plans completed: 1
- Average duration: 1 min
- Total execution time: 1 min

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 00-baseline-audit | 1 | 1 min | 1 min |

**Recent Trend:**
- Last 5 plans: 00-01 (1 min)
- Trend: -

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- Pending: Use SAM3 for scene labeling (already in infra, extend to return surface labels)
- Pending: Context-aware prompting over generic prompt (current "empty ground" causes hallucinations)
- Pending: Async label checking with requeue (avoids blocking inpaint on SAM3 latency)
- Pending: Open to alternative inpainting models if speed/quality on T4 warrants it
- **00-01:** Q5_K_M does not exist in YarvixPA repo — Q5_K_S (8.29 GB) used instead
- **00-01:** Audit route is synchronous def (not async def) — FastAPI runs in thread pool, correct for long-running GPU task
- **00-01:** torch.compile check runs after the 15-inference matrix to avoid polluting timing results
- **00-01:** dynamo.explain()() wraps full pipeline call (1 step) not bare transformer — tests realistic call graph

### Pending Todos

None yet.

### Blockers/Concerns

- Phase 0 (critical): Q4 quantization contribution to hallucinations is unconfirmed — must run POST /audit on session D1A406F5 to get Q8_0 comparison
- Phase 0 (critical): Guidance scale target for FLUX.1-Fill-dev background fill is unresolved — model card (30) conflicts with community fill-mode guidance (2-5) — audit will test all 5 values
- Phase 0: torch.compile viability on L4 sm_89 unconfirmed — audit will capture graph_break_count via dynamo.explain()
- Phase 2 (future): SAM3 label vocabulary coverage for unusual indoor surfaces (carpet, hardwood) not defined
- Phase 2 (future): Pub/Sub ack deadline during long inference — default 60s may cause redelivery mid-inference; verify and fix before or during Phase 2

## Session Continuity

Last session: 2026-02-22
Stopped at: 00-01-PLAN.md complete — POST /audit endpoint built and committed; ready to execute audit run
Resume file: None
