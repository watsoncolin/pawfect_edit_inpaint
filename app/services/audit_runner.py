import gc
import io
import logging
import time
from datetime import datetime
from typing import Any

import torch
import torch._dynamo as dynamo
from diffusers import FluxTransformer2DModel, GGUFQuantizationConfig
from huggingface_hub import hf_hub_download

from app.services import firebase
from app.services.flux_inpaint import get_pipeline, load_model as reload_startup_model
from app.utils.image import decode_image, decode_mask, resize_for_flux

logger = logging.getLogger(__name__)

REFERENCE_SESSION_PROMPT = "empty ground, nothing here, just the natural ground surface continuing seamlessly"
NUM_STEPS = 28  # held constant for quant and guidance tests

# Quantization variants: (label, HF filename)
# Q5_K_M does not exist in YarvixPA repo — use Q5_K_S (8.29 GB)
QUANT_VARIANTS = [
    ("Q4_0", "flux1-fill-dev-Q4_0.gguf"),
    ("Q5_K_S", "flux1-fill-dev-Q5_K_S.gguf"),
    ("Q8_0", "flux1-fill-dev-Q8_0.gguf"),
]

# Guidance scale grid (covers full disagreement range: official=30, community fill=2-5, current prod=10)
GUIDANCE_SCALE_GRID = [2, 4, 10, 20, 30]

HF_REPO = "YarvixPA/FLUX.1-Fill-dev-GGUF"


def _remove_offload_hooks(pipe):
    """Remove existing CPU offload hooks so they can be re-attached."""
    from accelerate.hooks import remove_hook_from_module
    for component_name in pipe.config:
        component = getattr(pipe, component_name, None)
        if isinstance(component, torch.nn.Module):
            remove_hook_from_module(component, recurse=True)


def _swap_transformer(pipe, gguf_filename: str):
    """Load a new GGUF transformer and swap it into the existing pipeline.

    Frees the old transformer first to keep peak memory low.
    """
    # Remove offload hooks before modifying pipeline
    _remove_offload_hooks(pipe)

    # Free old transformer (skip .to() — may be on meta device from offload)
    if pipe.transformer is not None:
        del pipe.transformer
        gc.collect()
        torch.cuda.empty_cache()

    local_gguf = hf_hub_download(repo_id=HF_REPO, filename=gguf_filename)
    transformer = FluxTransformer2DModel.from_single_file(
        local_gguf,
        quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
        torch_dtype=torch.bfloat16,
    )
    pipe.transformer = transformer
    return pipe


def run_audit(user_id: str, session_id: str, run_id: str) -> dict[str, Any]:
    """
    Run the full baseline audit parameter matrix.

    Outer loop: 3 quantization variants (Q4_0, Q5_K_S, Q8_0)
    Inner loop: 5 guidance scale values (2, 4, 10, 20, 30)
    Total: 15 inferences + 1 torch.compile viability check.

    Reuses the startup pipeline's shared components (text encoders, VAE, scheduler)
    and only swaps the transformer for each quant variant to stay within memory limits.

    All outputs uploaded to Firebase Storage under audit/{run_id}/.
    Returns a dict with run_id, gpu info, results list, compile viability, and report URL.
    """
    # -------------------------------------------------------------------------
    # 1. Confirm GPU identity
    # -------------------------------------------------------------------------
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        sm = f"sm_{major}{minor}"
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    else:
        sm = "N/A"
        total_vram_gb = 0.0
    logger.info(f"GPU: {gpu_name}, compute capability: {sm}, VRAM: {total_vram_gb:.1f} GB")

    # -------------------------------------------------------------------------
    # 2. Get the existing pipeline (reuse text encoders, VAE, scheduler)
    # -------------------------------------------------------------------------
    pipe = get_pipeline()
    if pipe is None:
        raise RuntimeError("Startup pipeline not loaded — cannot run audit")
    logger.info("Reusing startup pipeline components for audit")

    # -------------------------------------------------------------------------
    # 3. Download session assets from Firebase Storage
    # -------------------------------------------------------------------------
    base_path = f"users/{user_id}/sessions/{session_id}"
    logger.info(f"Downloading session assets from {base_path}")
    image_bytes = firebase.download_blob(f"{base_path}/original.jpg")
    mask_bytes = firebase.download_blob(f"{base_path}/mask_auto.png")

    image_orig = decode_image(image_bytes)
    mask_orig = decode_mask(mask_bytes)
    image_resized = resize_for_flux(image_orig)
    mask_resized = resize_for_flux(mask_orig)
    logger.info(f"Session assets ready: image={image_resized.size}, mask={mask_resized.size}")

    # -------------------------------------------------------------------------
    # 4. Run parameter matrix (swap transformer only, reuse everything else)
    # -------------------------------------------------------------------------
    results: list[dict[str, Any]] = []

    for quant_label, gguf_filename in QUANT_VARIANTS:
        logger.info(f"Swapping transformer: {quant_label} ({gguf_filename})")
        _swap_transformer(pipe, gguf_filename)
        pipe.enable_sequential_cpu_offload()
        logger.info(f"Pipeline ready for {quant_label}")

        for gs in GUIDANCE_SCALE_GRID:
            vram_before = torch.cuda.memory_allocated(0) / 1e9
            logger.info(f"Running inference: {quant_label}, guidance_scale={gs}, VRAM allocated: {vram_before:.1f} GB")

            # Dual timing: wall-clock + CUDA events
            start_wall = time.time()
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)
            start_evt.record()

            result_pipe = pipe(
                prompt=REFERENCE_SESSION_PROMPT,
                image=image_resized,
                mask_image=mask_resized,
                num_inference_steps=NUM_STEPS,
                guidance_scale=gs,
            )

            end_evt.record()
            torch.cuda.synchronize()
            wall_s = time.time() - start_wall
            gpu_ms = start_evt.elapsed_time(end_evt)

            vram_after = torch.cuda.memory_allocated(0) / 1e9
            logger.info(f"Inference done: {quant_label} gs={gs}, wall={wall_s:.1f}s, gpu={gpu_ms:.0f}ms, VRAM: {vram_after:.1f} GB")

            # Save image to PNG bytes
            img = result_pipe.images[0]
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            img_bytes = buf.getvalue()

            # Upload to Firebase Storage
            storage_path = f"audit/{run_id}/{quant_label}_gs{gs:.0f}.png"
            firebase.upload_blob(storage_path, img_bytes, "image/png")
            url = firebase.generate_signed_url(storage_path)

            results.append({
                "quant": quant_label,
                "guidance": gs,
                "steps": NUM_STEPS,
                "wall_s": wall_s,
                "gpu_ms": gpu_ms,
                "image_path": storage_path,
                "signed_url": url,
            })

        # VRAM cleanup after each quantization group
        reserved = torch.cuda.memory_reserved(0) / 1e9
        logger.info(f"VRAM after {quant_label}: {reserved:.1f} GB reserved")

    # -------------------------------------------------------------------------
    # 5. torch.compile viability check (reuse pipeline, swap back to Q4_0)
    # -------------------------------------------------------------------------
    logger.info("Running torch.compile viability check with Q4_0...")
    _swap_transformer(pipe, "flux1-fill-dev-Q4_0.gguf")
    pipe.enable_sequential_cpu_offload()

    dynamo.reset()

    def _test_forward():
        with torch.no_grad():
            return pipe(
                prompt=REFERENCE_SESSION_PROMPT,
                image=image_resized,
                mask_image=mask_resized,
                num_inference_steps=1,  # minimal — just to trace
                guidance_scale=4.0,
            )

    explanation = dynamo.explain(_test_forward)()
    compile_graph_count = explanation.graph_count
    compile_break_count = explanation.graph_break_count
    compile_break_reasons = [str(r) for r in explanation.break_reasons[:5]]  # cap at 5
    compile_viable = compile_break_count == 0
    compile_recommendation = (
        "Proceed with torch.compile in Phase 4 — zero graph breaks detected"
        if compile_viable
        else f"torch.compile has {compile_break_count} graph break(s) — review break reasons before Phase 4"
    )
    logger.info(
        f"torch.compile check: graphs={compile_graph_count}, breaks={compile_break_count}, viable={compile_viable}"
    )

    # -------------------------------------------------------------------------
    # 6. Restore Q4_0 transformer (startup default) for normal inpainting
    # -------------------------------------------------------------------------
    _swap_transformer(pipe, "flux1-fill-dev-Q4_0.gguf")
    pipe.enable_sequential_cpu_offload()
    logger.info("Startup pipeline restored with Q4_0 transformer")

    # -------------------------------------------------------------------------
    # 7. Generate Markdown report
    # -------------------------------------------------------------------------
    report_date = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    # Build results table rows
    table_rows = []
    for r in results:
        row = (
            f"| {r['quant']} | {r['guidance']} | {r['steps']} "
            f"| {r['wall_s']:.1f} | {r['gpu_ms']:.0f} "
            f"| [view]({r['signed_url']}) |"
        )
        table_rows.append(row)
    table_body = "\n".join(table_rows)

    break_reasons_str = ", ".join(compile_break_reasons) if compile_break_reasons else "None"

    report_md = f"""# Baseline Audit Report — {run_id}
**Session:** {session_id} | **Date:** {report_date} | **GPU:** {gpu_name} ({sm})

## GPU Info
- Device: {gpu_name}
- Compute Capability: {sm}
- VRAM: {total_vram_gb:.1f} GB

## Results Matrix

| Quant | Guidance | Steps | Wall-clock (s) | GPU time (ms) | Image |
|-------|----------|-------|----------------|---------------|-------|
{table_body}

## torch.compile Viability
- Graph breaks detected: {compile_break_count}
- Graph count: {compile_graph_count}
- Break reasons: {break_reasons_str}
- Recommendation: {compile_recommendation}

## Recommended Settings for Phase 1
*(Review images above and fill in)*
- Best quantization variant:
- Best guidance scale:
- Notes:
"""

    report_path = f"audit/{run_id}/REPORT.md"
    firebase.upload_blob(report_path, report_md.encode("utf-8"), "text/markdown")
    report_signed_url = firebase.generate_signed_url(report_path)
    logger.info(f"Audit report uploaded: {report_path}")

    # -------------------------------------------------------------------------
    # 8. Return result dict
    # -------------------------------------------------------------------------
    return {
        "run_id": run_id,
        "gpu": gpu_name,
        "sm": sm,
        "results": results,
        "compile_viable": compile_viable,
        "report_url": report_signed_url,
        "report_path": report_path,
    }
