# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-21)

**Core value:** Inpainted areas must seamlessly match the surrounding background — users should not be able to tell something was removed.
**Current focus:** Phase 0 — Baseline Audit

## Current Position

Phase: 0 of 4 (Baseline Audit)
Plan: 0 of TBD in current phase
Status: Ready to plan
Last activity: 2026-02-21 — Roadmap created

Progress: [░░░░░░░░░░] 0%

## Performance Metrics

**Velocity:**
- Total plans completed: 0
- Average duration: -
- Total execution time: -

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| - | - | - | - |

**Recent Trend:**
- Last 5 plans: -
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

### Pending Todos

None yet.

### Blockers/Concerns

- Phase 0 (critical): Q4 quantization contribution to hallucinations is unconfirmed — must run Q8_0 on session D1A406F5 before any optimization work
- Phase 0 (critical): Guidance scale target for FLUX.1-Fill-dev background fill is unresolved — model card (30) conflicts with community fill-mode guidance (2-5)
- Phase 0: torch.compile viability on T4 sm_75 unconfirmed — Flash Attention requires sm_80+; may silently fall back
- Phase 2 (future): SAM3 label vocabulary coverage for unusual indoor surfaces (carpet, hardwood) not defined
- Phase 2 (future): Pub/Sub ack deadline during long inference — default 60s may cause redelivery mid-inference; verify and fix before or during Phase 2

## Session Continuity

Last session: 2026-02-21
Stopped at: Roadmap created, STATE.md initialized — ready to plan Phase 0
Resume file: None
