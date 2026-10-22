from __future__ import annotations

import io
import warnings

from PIL import Image, ImageOps, UnidentifiedImageError


class UnsafeImageError(ValueError):
    pass


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
