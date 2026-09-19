"""Runtime configuration.

Every tunable lives here so the pipeline modules stay free of magic numbers.
Values can be overridden with environment variables prefixed LUNAR_.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Config:
    # --- server ---
    HOST: str = os.environ.get("LUNAR_HOST", "127.0.0.1")
    PORT: int = _int("LUNAR_PORT", 8000)
    DEBUG: bool = os.environ.get("LUNAR_DEBUG", "0") == "1"

    # --- upload limits ---
    MAX_UPLOAD_BYTES: int = _int("LUNAR_MAX_UPLOAD_BYTES", 16 * 1024 * 1024)
    MAX_PIXELS: int = _int("LUNAR_MAX_PIXELS", 40_000_000)
    MAX_EDGE: int = _int("LUNAR_MAX_EDGE", 2400)
    MIN_EDGE: int = _int("LUNAR_MIN_EDGE", 16)

    # Pillow format identifiers we accept. Detected from bytes, never extension.
    # MPO is included because many phone cameras write multi-picture JPEGs that
    # Pillow reports as MPO rather than JPEG.
    ALLOWED_FORMATS: frozenset = field(
        default_factory=lambda: frozenset({"PNG", "JPEG", "MPO", "TIFF", "BMP", "WEBP"})
    )

    # --- results storage ---
    RESULT_TTL_SECONDS: int = _int("LUNAR_RESULT_TTL", 30 * 60)
    MAX_RESULTS: int = _int("LUNAR_MAX_RESULTS", 64)
    SWEEP_INTERVAL_SECONDS: int = _int("LUNAR_SWEEP_INTERVAL", 60)

    # --- enhancement defaults ---
    DEFAULT_MODE: str = "scientific"
    GAMMA_DEFAULT: float = _float("LUNAR_GAMMA", 2.2)
    CLAHE_CLIP_DEFAULT: float = _float("LUNAR_CLAHE_CLIP", 2.5)
    CLAHE_GRID_DEFAULT: int = _int("LUNAR_CLAHE_GRID", 8)

    # --- Zero-DCE ---
    WEIGHTS_PATH: str = os.environ.get(
        "LUNAR_WEIGHTS", os.path.join("models", "zero_dce_weights.npz")
    )
    DCE_ITERATIONS: int = _int("LUNAR_DCE_ITERATIONS", 8)
    DCE_MAX_EDGE: int = _int("LUNAR_DCE_MAX_EDGE", 512)

    # --- reporting ---
    HISTOGRAM_BINS: int = _int("LUNAR_HIST_BINS", 64)
    PREVIEW_MAX_EDGE: int = _int("LUNAR_PREVIEW_MAX_EDGE", 1400)


CONFIG = Config()
