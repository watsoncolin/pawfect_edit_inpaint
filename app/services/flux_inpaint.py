import logging

import torch
from diffusers import FluxFillPipeline, FluxTransformer2DModel, GGUFQuantizationConfig
from PIL import Image

logger = logging.getLogger(__name__)

_pipe = None

PROMPT = "empty ground, nothing here, just the natural ground surface continuing seamlessly"
NUM_STEPS = 28
GUIDANCE_SCALE = 10

GGUF_URL = "https://huggingface.co/YarvixPA/FLUX.1-Fill-dev-GGUF/blob/main/flux1-fill-dev-Q4_0.gguf"


def load_model():
    """Load FLUX.1-Fill-dev pipeline with Q4 GGUF quantized transformer."""
    global _pipe

    logger.info("Loading GGUF Q4 transformer...")
    transformer = FluxTransformer2DModel.from_single_file(
        GGUF_URL,
        quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
        torch_dtype=torch.bfloat16,
    )

    logger.info("Loading FLUX.1-Fill-dev pipeline with GGUF transformer...")
    _pipe = FluxFillPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Fill-dev",
        transformer=transformer,
        torch_dtype=torch.bfloat16,
    )

    logger.info("Enabling sequential CPU offload...")
    _pipe.enable_sequential_cpu_offload()

    logger.info("FLUX.1-Fill-dev (Q4 GGUF) ready")


def is_ready() -> bool:
    return _pipe is not None


def inpaint(image: Image.Image, mask: Image.Image) -> Image.Image:
    """Run FLUX inpainting on image with mask. Returns the inpainted image."""
    if _pipe is None:
        raise RuntimeError("Model not loaded")

    logger.info(f"Running inpaint: image={image.size}, mask={mask.size}, steps={NUM_STEPS}")
    result = _pipe(
        prompt=PROMPT,
        image=image,
        mask_image=mask,
        num_inference_steps=NUM_STEPS,
        guidance_scale=GUIDANCE_SCALE,
    )

    return result.images[0]
