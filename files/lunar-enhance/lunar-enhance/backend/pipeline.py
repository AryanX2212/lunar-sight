"""Orchestration: one upload in, one complete analysis out.

The pipeline is the only place that knows the order of operations, so routes.py
stays a thin HTTP layer and the processing modules stay independently testable.
"""
from __future__ import annotations

import io
import time
from typing import Any

import numpy as np
from PIL import Image

from . import __version__
from .config import CONFIG
from .enhance import enhance
from .errors import ProcessingFailed
from .metrics import (
    compare,
    compute_metrics,
    difference_map,
    difference_summary,
    histogram,
    pixel_telemetry,
)
from .storage import STORE
from .zerodce import WEIGHTS, enhance_zerodce


def encode_png(rgb_u8: np.ndarray) -> bytes:
    """PNG rather than JPEG: this is analysis output, and JPEG artefacts would
    show up in the difference map as change the enhancement did not make."""
    img = Image.fromarray(rgb_u8, mode="RGB")
    longest = max(img.size)
    if longest > CONFIG.PREVIEW_MAX_EDGE:
        scale = CONFIG.PREVIEW_MAX_EDGE / longest
        img = img.resize(
            (max(1, int(img.width * scale)), max(1, int(img.height * scale))),
            Image.LANCZOS,
        )
    buffer = io.BytesIO()
    # compress_level=3 rather than optimize=True: measured on a 1024x768 frame
    # it is ~10x faster and the file is actually smaller, because optimize
    # spends its passes searching filters that do not suit photographic data.
    img.save(buffer, format="PNG", compress_level=3)
    return buffer.getvalue()


def _run_enhancement(rgb_u8: np.ndarray, options: dict[str, Any]):
    if options["mode"] == "zerodce":
        return enhance_zerodce(rgb_u8, **options)
    return enhance(rgb_u8, options)


def process(rgb_u8: np.ndarray, options: dict[str, Any], source_meta: dict[str, Any]) -> dict[str, Any]:
    """Run the full analysis and store it. Returns the JSON-ready payload."""
    timings: dict[str, float] = {}

    started = time.perf_counter()
    try:
        enhanced, report = _run_enhancement(rgb_u8, options)
    except Exception as exc:
        raise ProcessingFailed(
            f"The {options['mode']} enhancement failed: {exc}"
        ) from exc
    timings["enhance_ms"] = round((time.perf_counter() - started) * 1000, 1)

    if enhanced.shape != rgb_u8.shape or enhanced.dtype != np.uint8:
        raise ProcessingFailed(
            "The enhancement returned an image of the wrong shape or type."
        )

    started = time.perf_counter()
    before_metrics = compute_metrics(rgb_u8)
    after_metrics = compute_metrics(enhanced)
    timings["metrics_ms"] = round((time.perf_counter() - started) * 1000, 1)

    started = time.perf_counter()
    diff = difference_map(rgb_u8, enhanced)
    images = {
        "original": encode_png(rgb_u8),
        "enhanced": encode_png(enhanced),
        "difference": encode_png(diff),
    }
    timings["encode_ms"] = round((time.perf_counter() - started) * 1000, 1)
    timings["total_ms"] = round(sum(timings.values()), 1)

    payload: dict[str, Any] = {
        "source": source_meta,
        "options": options,
        "enhancement": report,
        "metrics": {"before": before_metrics, "after": after_metrics},
        "comparison": compare(before_metrics, after_metrics),
        "histogram": {
            "before": histogram(rgb_u8),
            "after": histogram(enhanced),
        },
        "difference": difference_summary(rgb_u8, enhanced),
        "telemetry": pixel_telemetry(rgb_u8, enhanced),
        "timings_ms": timings,
        "engine": {
            "version": __version__,
            "zero_dce": WEIGHTS.status(),
        },
        "image_bytes": {k: len(v) for k, v in images.items()},
    }

    result_id = STORE.put(payload, images)
    payload["id"] = result_id
    payload["images"] = {
        name: f"/api/results/{result_id}/image/{name}" for name in images
    }
    # Store the identical payload the client receives, so a later GET of the
    # result is byte-for-byte what the POST returned.
    STORE.set_payload(result_id, payload)
    return payload
