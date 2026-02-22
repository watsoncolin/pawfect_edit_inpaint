# Codebase Structure

**Analysis Date:** 2026-02-21

## Directory Layout

```
pawfect_edit_inpaint/
├── app/                        # FastAPI application source code
│   ├── __init__.py             # Package marker
│   ├── main.py                 # FastAPI app factory + lifespan + health probes
│   ├── routers/                # HTTP endpoint handlers (route modules)
│   │   ├── __init__.py
│   │   └── inpaint.py          # POST /inpaint — Pub/Sub message handler
│   ├── services/               # Business logic + external integrations
│   │   ├── __init__.py
│   │   ├── firebase.py         # Firestore + Storage client (lazy singleton)
│   │   ├── flux_inpaint.py     # FLUX.1-Fill-dev model loading + inference
│   │   └── pipeline.py         # Full inpainting workflow orchestration
│   └── utils/                  # Shared utilities (image processing)
│       ├── __init__.py
│       └── image.py            # Decode, resize, encode, preview generation
├── .github/                    # GitHub configuration
│   └── workflows/
│       └── deploy.yml          # GitHub Actions workflow (currently unused)
├── Dockerfile                  # Container image definition (NVIDIA CUDA + Python 3.10 + uv)
├── cloudbuild.yaml             # Google Cloud Build pipeline (build → push → deploy to Cloud Run)
├── pyproject.toml              # Project metadata + dependencies (torch, diffusers, fastapi, firebase-admin, etc.)
├── README.md                   # Project documentation (architecture, endpoints, environment vars)
└── .gitignore                  # Git ignore patterns

(Not committed: venv/, .uv/, __pycache__/, .pytest_cache/)
```

## Directory Purposes

**app/**
- Purpose: Python FastAPI application package
- Contains: HTTP routes, business logic, integrations, utilities
- Key files: `main.py` (entry point), `routers/` (endpoints), `services/` (logic)

**app/routers/**
- Purpose: HTTP endpoint definitions
- Contains: Request handlers, message parsing, error response mapping
- Key files: `inpaint.py` (Pub/Sub handler)

**app/services/**
- Purpose: Core business logic and external service integrations
- Contains: Pipeline orchestration, Firebase client, FLUX model inference, state management
- Key files: `pipeline.py` (workflow), `firebase.py` (storage + firestore), `flux_inpaint.py` (model)

**app/utils/**
- Purpose: Reusable utility functions (image processing)
- Contains: Image decoding/encoding, resizing, preview generation
- Key files: `image.py` (PIL image operations)

**.github/workflows/**
- Purpose: CI/CD automation (currently not in use; Cloud Build is primary)
- Contains: GitHub Actions workflow definition
- Key files: `deploy.yml` (unused alternative deployment)

## Key File Locations

**Entry Points:**
- `app/main.py`: FastAPI application factory, lifespan context manager (model loading), health + readiness endpoints
- `Dockerfile`: Container image build, model weight pre-caching at build time

**Configuration:**
- `pyproject.toml`: Python version (3.10-3.12), dependencies, build tool config (uv)
- `cloudbuild.yaml`: Build arguments (HF_TOKEN), deployment parameters (GPU/CPU/memory), probe configuration
- `Dockerfile`: CUDA base image, Python 3.10, environment variables (HF_HUB_ENABLE_HF_TRANSFER, CUDA_MODULE_LOADING)

**Core Logic:**
- `app/services/pipeline.py`: Main inpainting workflow (download → resize → inpaint → upload → update)
- `app/services/flux_inpaint.py`: Model loading and inference execution
- `app/routers/inpaint.py`: Pub/Sub message handling and validation

**Integration Points:**
- `app/services/firebase.py`: Firebase Admin SDK initialization and Firestore/Storage operations
- `app/utils/image.py`: Image processing for FLUX compatibility

**Testing:**
- Not present in current codebase (no test directory)

## Naming Conventions

**Files:**
- Python modules: `snake_case.py` (e.g., `firebase.py`, `flux_inpaint.py`, `image.py`)
- Special files: `main.py` (FastAPI app), `__init__.py` (package markers)

**Directories:**
- Feature-based: `routers/`, `services/`, `utils/` — grouped by responsibility
- Nested packages: `app/` top-level, functional subpackages within

**Functions/Classes:**
- Functions: `snake_case` (e.g., `load_model()`, `run_inpaint()`, `encode_jpeg()`)
- Constants: `UPPER_CASE` (e.g., `PROMPT`, `NUM_STEPS`, `GUIDANCE_SCALE`, `COST_PER_INPAINT`)
- Private/Module-level: Leading underscore (e.g., `_pipe`, `_initialized`, `_init()`)

**Variables:**
- Local: `snake_case` (e.g., `user_id`, `session_id`, `image_bytes`)
- Environment variables: `UPPER_CASE_WITH_UNDERSCORES` (e.g., `FIREBASE_CREDENTIALS`, `HF_TOKEN`)

## Where to Add New Code

**New Feature (inpainting variant or new processing step):**
- Primary code: `app/services/pipeline.py` or new service file in `app/services/`
- Import in: `app/routers/inpaint.py` or create new router
- Configuration: Add constants to `flux_inpaint.py` if model-specific (e.g., new `NUM_STEPS` variant), or to pipeline

**New Endpoint:**
- Implementation: Create new file in `app/routers/` (e.g., `app/routers/health.py`)
- Registration: Import and include router in `app/main.py` with `app.include_router(router)`

**New Utility Function (image processing):**
- Implementation: Add to `app/utils/image.py` or create new utility file
- Import: From calling module with `from app.utils.image import function_name`

**New External Integration:**
- Implementation: Create new service file in `app/services/` (e.g., `app/services/analytics.py`)
- Lazy initialization: Follow singleton pattern in `firebase.py` with module-level `_initialized` flag
- Startup hook: If async init needed, add to lifespan in `main.py`

## Special Directories

**Deleted/Runtime-Generated:**
- `venv/` or `.uv/`: Virtual environment directory (excluded from git via `.gitignore`)
- `__pycache__/`: Python bytecode cache (excluded from git)
- `.pytest_cache/`: Test cache (excluded from git)
- `.next/`: Unused in this Python backend
- `dist/`: Unused in Cloud Run deployment

## Model Weights & Baking

**Location:** Pre-cached in container image during Docker build phase
- GGUF Q4 transformer (~12.7GB): Downloaded at `Dockerfile` line 34 via HuggingFace Hub
- FLUX.1-Fill-dev pipeline components (tokenizer, scheduler): Downloaded at `Dockerfile` line 36

**Why:** Eliminates cold-start download latency (~5+ minutes) when Cloud Run instance starts

**Build argument:** `HF_TOKEN` (GitHub secret) required for gated model access, cleared from final image

## Environment & Deployment

**Local Development:**
```bash
uv sync                                    # Install dependencies
uv run uvicorn app.main:app --reload      # Run with auto-reload
```

**Build & Deploy:**
- Google Cloud Build (`cloudbuild.yaml`) orchestrates: build Docker image → push to GCR → deploy to Cloud Run
- Cloud Run configuration: 1x GPU (NVIDIA L4), 8 CPUs, 32GB RAM, single instance, single concurrent request

---

*Structure analysis: 2026-02-21*
