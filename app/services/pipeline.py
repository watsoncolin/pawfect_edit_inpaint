import logging
import time

from PIL import Image

from app.services import firebase
from app.services.flux_inpaint import inpaint
from app.utils.image import create_preview, decode_image, decode_mask, encode_jpeg, resize_for_flux

logger = logging.getLogger(__name__)

COST_PER_INPAINT = 0.03


def run_inpaint(user_id: str, session_id: str):
    """Full pipeline: download → inpaint → upload → update Firestore."""
    logger.info(f"Starting inpaint: userId={user_id}, sessionId={session_id}")
    base_path = f"users/{user_id}/sessions/{session_id}"

    try:
        # Mark as processing
        firebase.update_session(user_id, session_id, {"processingStatus": "processing"})

        # Download original image and mask
        image_bytes = firebase.download_blob(f"{base_path}/original.jpg")
        mask_bytes = firebase.download_blob(f"{base_path}/mask_auto.png")

        image = decode_image(image_bytes)
        mask = decode_mask(mask_bytes)
        original_size = image.size

        # Resize to FLUX-compatible dimensions
        image_resized = resize_for_flux(image)
        mask_resized = resize_for_flux(mask)

        # Ensure mask matches image dimensions
        if mask_resized.size != image_resized.size:
            mask_resized = mask_resized.resize(image_resized.size)

        logger.info(f"Original size: {original_size}, FLUX size: {image_resized.size}")

        # Run inference
        start = time.time()
        result = inpaint(image_resized, mask_resized)
        elapsed = time.time() - start
        logger.info(f"Inference completed in {elapsed:.1f}s")

        # Resize result back to original dimensions
        if result.size != original_size:
            result = result.resize(original_size, Image.LANCZOS)

        # Composite: only use FLUX output in the masked area, keep original pixels elsewhere.
        # This prevents color shifts in unmasked regions.
        if mask.size != original_size:
            mask = mask.resize(original_size, Image.LANCZOS)
        result = Image.composite(result, image, mask.convert("L"))

        # Encode and upload
        edited_bytes = encode_jpeg(result, quality=95)
        preview_bytes = create_preview(result, max_size=512, quality=85)

        edited_path = firebase.upload_blob(f"{base_path}/edited.jpg", edited_bytes, "image/jpeg")
        preview_path = firebase.upload_blob(f"{base_path}/preview.jpg", preview_bytes, "image/jpeg")

        # Update Firestore
        firebase.update_session(user_id, session_id, {
            "processingStatus": "completed",
            "editedImagePath": edited_path,
            "previewImagePath": preview_path,
            "cost": COST_PER_INPAINT,
        })

        logger.info(f"Inpaint complete: sessionId={session_id}, time={elapsed:.1f}s")

    except Exception as e:
        logger.exception(f"Error during inpaint: {e}")
        # Mark as failed — ACK the message to prevent infinite retries on permanent errors
        try:
            firebase.update_session(user_id, session_id, {
                "processingStatus": "failed",
                "errorMessage": str(e),
            })
        except Exception:
            logger.exception("Failed to update Firestore with error status")
        raise
