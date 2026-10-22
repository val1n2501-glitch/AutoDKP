from __future__ import annotations

import io
import warnings

from PIL import Image, ImageFilter, ImageOps, ImageStat, UnidentifiedImageError


class UnsafeImageError(ValueError):
    pass


def assess_image_quality(raw: bytes) -> tuple[str, ...]:
    """Return conservative, user-facing warnings without rejecting the image."""
    with Image.open(io.BytesIO(raw)) as source:
        image = source.convert("RGB")
        warnings_found: list[str] = []
        if image.width < 900 or image.height < 650:
            warnings_found.append("низкое разрешение")
        preview = image.copy()
        preview.thumbnail((640, 640))
        gray = ImageOps.grayscale(preview)
        brightness = ImageStat.Stat(gray).mean[0]
        if brightness < 45:
            warnings_found.append("фото слишком тёмное")
        # FIND_EDGES also highlights the artificial outer image border.  Exclude
        # that border so a blank/blurred photo is not mistaken for a sharp one.
        edges = gray.filter(ImageFilter.FIND_EDGES)
        if edges.width > 8 and edges.height > 8:
            edges = edges.crop((3, 3, edges.width - 3, edges.height - 3))
        if ImageStat.Stat(edges).var[0] < 90:
            warnings_found.append("возможна сильная размытость")
        histogram = gray.histogram()
        glare_ratio = sum(histogram[248:]) / max(1, preview.width * preview.height)
        if glare_ratio > 0.20:
            warnings_found.append("много пересвеченных участков или бликов")
        return tuple(warnings_found)


def sanitize_image(raw: bytes, *, max_bytes: int, max_pixels: int) -> bytes:
    """Decode and re-encode a single-frame JPEG/PNG without source metadata."""
    if len(raw) > max_bytes:
        raise UnsafeImageError("Файл слишком большой.")
    is_jpeg = raw.startswith(b"\xff\xd8\xff")
    is_png = raw.startswith(b"\x89PNG\r\n\x1a\n")
    if not (is_jpeg or is_png):
        raise UnsafeImageError("Пока поддерживаются только JPEG и PNG.")

    previous_limit = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = max_pixels
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as source:
                if source.format not in {"JPEG", "PNG"} or getattr(source, "n_frames", 1) != 1:
                    raise UnsafeImageError("Неподдерживаемый тип изображения.")
                if source.width * source.height > max_pixels:
                    raise UnsafeImageError("Слишком большое разрешение фото.")
                normalized = ImageOps.exif_transpose(source).convert("RGB")
                normalized.load()
                output = io.BytesIO()
                normalized.save(output, format="JPEG", quality=92, optimize=True)
                return output.getvalue()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise UnsafeImageError("Не удалось безопасно прочитать фото.") from error
    finally:
        Image.MAX_IMAGE_PIXELS = previous_limit
