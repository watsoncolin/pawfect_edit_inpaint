import io

from PIL import Image, ImageOps


def decode_image(data: bytes) -> Image.Image:
    """Decode image bytes to PIL Image, applying EXIF orientation."""
    image = Image.open(io.BytesIO(data))
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def decode_mask(data: bytes) -> Image.Image:
    """Decode mask bytes to PIL Image in grayscale."""
    image = Image.open(io.BytesIO(data))
    return image.convert("L")


def resize_for_flux(image: Image.Image, max_size: int = 1024) -> Image.Image:
    """Resize image to FLUX-compatible dimensions (multiples of 8, max max_size px)."""
    w, h = image.size

    # Scale down if needed
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        w = int(w * scale)
        h = int(h * scale)

    # Round to nearest multiple of 8
    w = (w // 8) * 8
    h = (h // 8) * 8

    # Ensure minimum size
    w = max(w, 8)
    h = max(h, 8)

    if (w, h) != image.size:
        return image.resize((w, h), Image.LANCZOS)
    return image


def encode_jpeg(image: Image.Image, quality: int = 95) -> bytes:
    """Encode PIL Image to JPEG bytes."""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def create_preview(image: Image.Image, max_size: int = 512, quality: int = 85) -> bytes:
    """Create a smaller preview JPEG."""
    w, h = image.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        w = int(w * scale)
        h = int(h * scale)
        image = image.resize((w, h), Image.LANCZOS)
    return encode_jpeg(image, quality)
