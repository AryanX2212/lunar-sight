"""Upload validation and safe decoding.

Design notes
------------
* The file extension and the browser-supplied content type are both treated as
  untrusted hints. The real format comes from Pillow reading the magic bytes.
* Decode is a two-step: open() to inspect the header cheaply, then load the
  pixels only once the declared dimensions pass the limits. That stops a
  decompression bomb before it allocates.
* Everything leaves this module as a contiguous uint8 RGB array so downstream
  code never has to branch on channel count or bit depth.
"""
from __future__ import annotations

import io
from typing import Any

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .config import CONFIG
from .errors import PayloadTooLarge, UnsupportedImage, ValidationError

# Pillow refuses images above this pixel count by raising DecompressionBombError.
# We set our own ceiling slightly under Pillow's so we produce a clean API error
# rather than an exception from deep inside the decoder.
Image.MAX_IMAGE_PIXELS = CONFIG.MAX_PIXELS + 1

VALID_MODES = ("naive", "scientific", "clahe", "zerodce")


def validate_upload_size(raw: bytes) -> None:
    if not raw:
        raise ValidationError("The uploaded file is empty. Choose an image file.")
    if len(raw) > CONFIG.MAX_UPLOAD_BYTES:
        raise PayloadTooLarge(
            f"That file is {len(raw) / 1e6:.1f} MB. The limit is "
            f"{CONFIG.MAX_UPLOAD_BYTES / 1e6:.0f} MB.",
            details={"bytes": len(raw), "limit_bytes": CONFIG.MAX_UPLOAD_BYTES},
        )


def _inspect(raw: bytes) -> tuple[Image.Image, str]:
    """Open the header and check the format and dimensions before loading pixels."""
    try:
        img = Image.open(io.BytesIO(raw))
    except UnidentifiedImageError:
        raise UnsupportedImage(
            "That file is not an image we can read. Use PNG, JPEG, TIFF, BMP or WebP."
        ) from None
    except Image.DecompressionBombError:
        raise PayloadTooLarge("That image is too large to process safely.") from None
    except Exception as exc:  # corrupt header, truncated file, etc.
        raise UnsupportedImage(f"The image header could not be read: {exc}") from None

    fmt = (img.format or "").upper()
    if fmt not in CONFIG.ALLOWED_FORMATS:
        raise UnsupportedImage(
            f"{fmt or 'Unknown'} files are not supported. "
            "Use PNG, JPEG, TIFF, BMP or WebP.",
            details={"detected_format": fmt},
        )

    width, height = img.size
    if width < CONFIG.MIN_EDGE or height < CONFIG.MIN_EDGE:
        raise ValidationError(
            f"The image is {width}x{height}. Both sides must be at least "
            f"{CONFIG.MIN_EDGE} pixels.",
            details={"width": width, "height": height},
        )
    if width * height > CONFIG.MAX_PIXELS:
        raise PayloadTooLarge(
            f"The image is {width}x{height} ({width * height / 1e6:.1f} MP). "
            f"The limit is {CONFIG.MAX_PIXELS / 1e6:.0f} MP.",
            details={"width": width, "height": height},
        )
    return img, fmt


def _to_rgb_array(img: Image.Image) -> np.ndarray:
    """Load pixels and normalise to uint8 RGB, honouring EXIF orientation."""
    # exif_transpose also loads the image. Guard it: some files carry malformed
    # EXIF that raises here but decode fine without the rotation.
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass

    if img.mode in ("I;16", "I;16B", "I;16L", "I"):
        # 16-bit scientific TIFFs. Scale by the observed range rather than a
        # fixed 65535, because instrument data often occupies a narrow band.
        arr = np.asarray(img).astype(np.float64)
        lo, hi = float(arr.min()), float(arr.max())
        arr = np.zeros_like(arr) if hi <= lo else (arr - lo) / (hi - lo)
        arr8 = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        return np.repeat(arr8[:, :, None], 3, axis=2)

    if img.mode == "RGBA":
        # Composite onto black: lunar frames are dark, and black is the honest
        # background for a scene shot against space.
        backdrop = Image.new("RGBA", img.size, (0, 0, 0, 255))
        img = Image.alpha_composite(backdrop, img)

    if img.mode != "RGB":
        img = img.convert("RGB")

    arr = np.asarray(img, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise UnsupportedImage("The image could not be converted to RGB.")
    return np.ascontiguousarray(arr)


def _downscale_if_needed(arr: np.ndarray) -> tuple[np.ndarray, bool]:
    h, w = arr.shape[:2]
    longest = max(h, w)
    if longest <= CONFIG.MAX_EDGE:
        return arr, False
    scale = CONFIG.MAX_EDGE / longest
    new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    resized = Image.fromarray(arr).resize(new_size, Image.LANCZOS)
    return np.asarray(resized, dtype=np.uint8), True


def decode_image(raw: bytes, filename: str = "") -> tuple[np.ndarray, dict[str, Any]]:
    """Validate and decode an upload.

    Returns the RGB array plus a provenance record describing what arrived and
    what we did to it before any enhancement ran.
    """
    validate_upload_size(raw)
    img, fmt = _inspect(raw)
    original_size = img.size

    try:
        arr = _to_rgb_array(img)
    except UnsupportedImage:
        raise
    except Image.DecompressionBombError:
        raise PayloadTooLarge("That image is too large to process safely.") from None
    except Exception as exc:
        raise UnsupportedImage(
            f"The image data is damaged and could not be decoded: {exc}"
        ) from None
    finally:
        img.close()

    arr, was_resized = _downscale_if_needed(arr)

    meta = {
        "filename": filename or "upload",
        "detected_format": fmt,
        "source_width": original_size[0],
        "source_height": original_size[1],
        "width": int(arr.shape[1]),
        "height": int(arr.shape[0]),
        "resized_for_processing": was_resized,
        "bytes": len(raw),
    }
    return arr, meta


def validate_options(form: dict[str, Any]) -> dict[str, Any]:
    """Coerce and bound-check the enhancement options from a form submission."""
    mode = str(form.get("mode", CONFIG.DEFAULT_MODE)).strip().lower()
    if mode not in VALID_MODES:
        raise ValidationError(
            f"Unknown enhancement mode '{mode}'.",
            details={"supported_modes": list(VALID_MODES)},
        )

    def number(key: str, default: float, low: float, high: float) -> float:
        raw_value = form.get(key)
        if raw_value in (None, ""):
            return default
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            raise ValidationError(f"'{key}' must be a number.") from None
        if not np.isfinite(value):
            raise ValidationError(f"'{key}' must be a finite number.")
        if not low <= value <= high:
            raise ValidationError(
                f"'{key}' must be between {low} and {high}.",
                details={key: value, "min": low, "max": high},
            )
        return value

    def flag(key: str, default: bool) -> bool:
        raw_value = form.get(key)
        if raw_value in (None, ""):
            return default
        return str(raw_value).strip().lower() in ("1", "true", "yes", "on")

    return {
        "mode": mode,
        "gamma": number("gamma", CONFIG.GAMMA_DEFAULT, 1.0, 5.0),
        "clahe_clip": number("clahe_clip", CONFIG.CLAHE_CLIP_DEFAULT, 0.5, 8.0),
        "clahe_grid": int(number("clahe_grid", CONFIG.CLAHE_GRID_DEFAULT, 2, 32)),
        "denoise": flag("denoise", True),
        "preserve_highlights": flag("preserve_highlights", True),
    }
