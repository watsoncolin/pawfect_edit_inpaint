FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    APP_HOME=/app \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    CUDA_MODULE_LOADING=LAZY \
    HF_HUB_ENABLE_HF_TRANSFER=1

WORKDIR $APP_HOME

# Install Python 3.10
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install dependencies
COPY pyproject.toml uv.lock* ./
RUN uv sync --no-dev --no-install-project

# Model baking: TINY_MODEL=1 downloads ~50MB test model, otherwise full GGUF (~12.7GB)
ARG HF_TOKEN
ARG TINY_MODEL=0
ENV HF_TOKEN=${HF_TOKEN} TINY_MODEL=${TINY_MODEL}
RUN if [ "$TINY_MODEL" = "1" ]; then \
      echo "Baking tiny test model..." && \
      uv run python -c "from diffusers import FluxFillPipeline; FluxFillPipeline.from_pretrained('katuni4ka/tiny-random-flux-fill')"; \
    else \
      echo "Baking full GGUF model..." && \
      uv run python -c "\
from huggingface_hub import hf_hub_download; \
hf_hub_download('YarvixPA/FLUX.1-Fill-dev-GGUF', filename='flux1-fill-dev-Q8_0.gguf'); \
from diffusers import FluxFillPipeline; \
FluxFillPipeline.from_pretrained('black-forest-labs/FLUX.1-Fill-dev', transformer=None)"; \
    fi
# Clear token from image
ENV HF_TOKEN=

# Copy application
COPY app/ app/

EXPOSE 8080

CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
