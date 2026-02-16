import logging

import torch
from diffusers import FluxFillPipeline
from optimum.quanto import qfloat8, quantize
from PIL import Image

logger = logging.getLogger(__name__)

_pipe = None

PROMPT = "seamless natural background matching surrounding area, photorealistic, no leash, no rope, no cord"
NUM_STEPS = 28
GUIDANCE_SCALE = 30


def load_model():
    """Load and quantize FLUX.1-Fill-dev pipeline."""
    global _pipe
    logger.info("Loading FLUX.1-Fill-dev pipeline...")
    _pipe = FluxFillPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Fill-dev",
        torch_dtype=torch.bfloat16,
    )

    logger.info("Quantizing transformer to float8...")
    quantize(_pipe.transformer, weights=qfloat8)

    logger.info("Enabling model CPU offload...")
    _pipe.enable_model_cpu_offload()

    logger.info("FLUX.1-Fill-dev ready")


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
