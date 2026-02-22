# Technology Stack

**Analysis Date:** 2026-02-21

## Languages

**Primary:**
- Python 3.10 - Core application runtime and AI/ML inference
- YAML - Configuration and CI/CD workflows

## Runtime

**Environment:**
- Python 3.10 (specified in `pyproject.toml`)
- CUDA 12.4 (NVIDIA GPU compute)

**Package Manager:**
- uv (fast Python package manager)
- Lockfile: `uv.lock` (managed by uv, ensures reproducible builds)

## Frameworks

**Core:**
- FastAPI 0.110.0+ - HTTP API framework and request routing
- Uvicorn [standard] - ASGI application server

**Machine Learning:**
- torch 2.1.0+ - PyTorch deep learning framework
- diffusers 0.31.0+ - Hugging Face diffusion model library (FLUX.1-Fill-dev)
- transformers 4.47.0+ - Transformer model implementations
- accelerate 0.26.0+ - Distributed training and inference optimization

**Image Processing:**
- Pillow 10.0.0+ - Image encoding/decoding and manipulation

**Quantization:**
- gguf 0.6.0+ - GGUF format support for quantized models
- protobuf 5.0.0+ - Data serialization (required by GGUF)
- sentencepiece 0.2.0+ - Tokenizer dependencies

**Development/Testing:**
- pytest 8.3.0+ - Test runner (dev dependency)
- black 24.10.0+ - Code formatter (dev dependency)
- ruff 0.8.0+ - Linter (dev dependency)

**Data Transfer:**
- hf-transfer 0.1.6+ - Optimized Hugging Face Hub downloads

## Key Dependencies

**Critical:**
- torch - Core ML framework; used for GPU-accelerated inference
- diffusers - Provides FLUX.1-Fill-dev pipeline for image inpainting
- transformers - Model architecture definitions and loading

**Infrastructure:**
- firebase-admin 6.6.0+ - Firebase Authentication, Firestore, and Cloud Storage integration
- python-multipart - Multipart form data parsing (required by FastAPI)

## Configuration

**Environment:**
- Environment variables configured via Google Cloud Secret Manager (referenced in `cloudbuild.yaml`)
- Required at runtime: `FIREBASE_CREDENTIALS` (base64-encoded Firebase service account JSON)
- Required at build time: `HF_TOKEN` (Hugging Face token for gated model access)

**Build:**
- `pyproject.toml` - Python project metadata, dependencies, and tool configuration
- `Dockerfile` - Multi-stage container build with NVIDIA CUDA base
- `cloudbuild.yaml` - Google Cloud Build configuration for CI/CD
- `.github/workflows/deploy.yml` - GitHub Actions workflow triggering Cloud Build

**Code Quality:**
- Black configuration: line-length=100, target-version=py310
- Ruff configuration: line-length=100, target-version=py310

## Platform Requirements

**Development:**
- Python 3.10+ installed locally
- Optional: NVIDIA GPU with CUDA 12.4 (CPU inference possible but slow)
- `FIREBASE_CREDENTIALS` environment variable set for local testing

**Production:**
- Deployed on Google Cloud Run
- GPU: NVIDIA L4 (24GB VRAM)
- CPU: 8 cores
- Memory: 32GB
- Machine type: E2_HIGHCPU_32 (for build)
- Docker container runtime (via Cloud Run)
- Google Cloud Secret Manager for credential storage

**Build:**
- Google Cloud Build service account with permissions to push to Container Registry and deploy to Cloud Run

## PyPI Index Configuration

**Custom Index:**
- pytorch-cu124: https://download.pytorch.org/whl/cu124 (CUDA 12.4 optimized PyTorch wheels)

## Docker Image

**Base Image:**
- `nvidia/cuda:12.4.0-runtime-ubuntu22.04` - NVIDIA CUDA runtime with Ubuntu 22.04

**Build-Time Optimizations:**
- `UV_COMPILE_BYTECODE=1` - Compile Python to bytecode during build
- `UV_LINK_MODE=copy` - Copy dependencies instead of symlinking
- `HF_HUB_ENABLE_HF_TRANSFER=1` - Use optimized Hugging Face Hub transfer library
- `CUDA_MODULE_LOADING=LAZY` - Lazy load CUDA modules (reduce startup time)
- `PYTHONUNBUFFERED=1` - Unbuffered Python stdout for real-time logs

**Model Baking:**
- FLUX.1-Fill-dev GGUF Q8 quantization (~12.7GB) pre-downloaded at build time
- Pipeline components (tokenizer, scheduler) pre-loaded
- Reduces cold start time on Cloud Run

---

*Stack analysis: 2026-02-21*
