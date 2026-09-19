"""Image quality measurement, histograms and difference maps.

Honesty rules for this module
-----------------------------
Every metric carries a `kind` field:

  measured  - computed directly from the pixels, exact within float precision.
  estimated - derived from an assumption about the image (for example that a
              flat region's high-frequency content is sensor noise). Useful for
              comparison between before and after, but not an absolute figure.

Nothing here invents a number. If a metric cannot be computed for an image it
is returned as null rather than filled with a plausible default.
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .config import CONFIG

# Rec. 709 luma weights, matching how the eye weighs the channels.
LUMA_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

SHADOW_CLIP_LEVEL = 2      # 0-255; at or below this, shadow detail is gone
HIGHLIGHT_CLIP_LEVEL = 253  # at or above this, highlight detail is gone


def luminance(rgb_u8: np.ndarray) -> np.ndarray:
    """Rec. 709 luma as float32 in 0..255."""
    return rgb_u8.astype(np.float32) @ LUMA_WEIGHTS


def _metric(
    value: float | None,
    kind: str,
    unit: str,
    label: str,
    description: str,
    better: str = "higher",
) -> dict[str, Any]:
    return {
        "value": None if value is None else round(float(value), 4),
        "kind": kind,
        "unit": unit,
        "label": label,
        "description": description,
        "better": better,
    }


def _entropy(gray_u8: np.ndarray) -> float:
    counts = np.bincount(gray_u8.ravel(), minlength=256).astype(np.float64)
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log2(p)).sum())


def _estimate_noise_sigma(gray_u8: np.ndarray) -> float:
    """Immerkaer's fast noise estimator.

    Convolves with a kernel that annihilates locally-linear intensity, so what
    remains is dominated by noise. Estimated: real scene texture also survives
    the kernel, so this reads high on detailed images.
    """
    h, w = gray_u8.shape
    if h < 3 or w < 3:
        return 0.0
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
    response = cv2.filter2D(gray_u8.astype(np.float32), -1, kernel)
    sigma = np.abs(response).sum() / (36.0 * (w - 2) * (h - 2))
    return float(sigma * np.sqrt(0.5 * np.pi))


def _sharpness(gray_u8: np.ndarray) -> float:
    """Variance of the Laplacian: a standard focus/acutance proxy."""
    return float(cv2.Laplacian(gray_u8.astype(np.float32), cv2.CV_32F).var())


def _colorfulness(rgb_u8: np.ndarray) -> float:
    """Hasler & Suesstrunk colourfulness. Near zero for true greyscale frames."""
    r, g, b = (rgb_u8[:, :, i].astype(np.float32) for i in range(3))
    rg = r - g
    yb = 0.5 * (r + g) - b
    return float(
        np.sqrt(rg.std() ** 2 + yb.std() ** 2)
        + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    )


def compute_metrics(rgb_u8: np.ndarray) -> dict[str, dict[str, Any]]:
    """Full metric set for one image."""
    gray = np.clip(luminance(rgb_u8), 0, 255).astype(np.uint8)
    lum = gray.astype(np.float32)
    total = lum.size

    mean = float(lum.mean())
    rms = float(lum.std())
    occupied = int(np.count_nonzero(np.bincount(gray.ravel(), minlength=256)))

    p1, p99 = np.percentile(lum, [1.0, 99.0])
    sigma = _estimate_noise_sigma(gray)

    return {
        "mean_luminance": _metric(
            mean, "measured", "0-255", "Mean brightness",
            "Average Rec. 709 luma across the frame.",
            better="neutral",
        ),
        "rms_contrast": _metric(
            rms, "measured", "0-255", "RMS contrast",
            "Standard deviation of luma. Rises as the tonal spread widens.",
        ),
        "entropy": _metric(
            _entropy(gray), "measured", "bits", "Entropy",
            "Shannon entropy of the luma histogram, out of a maximum 8 bits. "
            "Higher means more distinguishable tonal levels survive.",
        ),
        "dynamic_range_used": _metric(
            100.0 * occupied / 256.0, "measured", "%", "Levels occupied",
            "Share of the 256 available luma levels that contain at least one pixel.",
        ),
        "effective_range": _metric(
            float(p99 - p1), "measured", "0-255", "1-99% span",
            "Distance between the 1st and 99th luma percentiles, ignoring outliers.",
        ),
        "shadow_clipping": _metric(
            100.0 * float(np.count_nonzero(gray <= SHADOW_CLIP_LEVEL)) / total,
            "measured", "%", "Crushed shadows",
            f"Pixels at or below luma {SHADOW_CLIP_LEVEL}, where detail is unrecoverable.",
            better="lower",
        ),
        "highlight_clipping": _metric(
            100.0 * float(np.count_nonzero(gray >= HIGHLIGHT_CLIP_LEVEL)) / total,
            "measured", "%", "Blown highlights",
            f"Pixels at or above luma {HIGHLIGHT_CLIP_LEVEL}, where detail is lost to saturation.",
            better="lower",
        ),
        "sharpness": _metric(
            _sharpness(gray), "measured", "variance", "Acutance",
            "Variance of the Laplacian. Tracks edge definition, but also rises with noise.",
        ),
        "noise_sigma": _metric(
            sigma, "estimated", "0-255", "Noise level",
            "Immerkaer estimate of sensor noise. Scene texture inflates this, so read "
            "it as a comparison between two versions of the same frame, not an absolute.",
            better="lower",
        ),
        "snr_estimate": _metric(
            (mean / sigma) if sigma > 1e-6 else None,
            "estimated", "ratio", "Signal to noise",
            "Mean brightness divided by the estimated noise level. Inherits the "
            "noise estimate's assumptions.",
        ),
        "colorfulness": _metric(
            _colorfulness(rgb_u8), "measured", "index", "Colourfulness",
            "Hasler & Suesstrunk index. Near zero for a true monochrome frame.",
            better="neutral",
        ),
    }


def histogram(rgb_u8: np.ndarray, bins: int | None = None) -> dict[str, Any]:
    """Per-channel and luma histograms, normalised so the peak bin is 1.0."""
    bins = bins or CONFIG.HISTOGRAM_BINS
    edges = np.linspace(0, 256, bins + 1)
    out: dict[str, Any] = {
        "bins": bins,
        "centers": [round(float(c), 2) for c in (edges[:-1] + edges[1:]) / 2.0],
    }
    channels = {
        "red": rgb_u8[:, :, 0],
        "green": rgb_u8[:, :, 1],
        "blue": rgb_u8[:, :, 2],
        "luma": np.clip(luminance(rgb_u8), 0, 255),
    }
    for name, data in channels.items():
        counts, _ = np.histogram(data, bins=edges)
        peak = counts.max()
        out[name] = [
            round(float(c), 5) for c in (counts / peak if peak else counts.astype(float))
        ]
    return out


def difference_map(before_u8: np.ndarray, after_u8: np.ndarray) -> np.ndarray:
    """A colourised map of where the enhancement changed the image, and by how much.

    The magnitude is the per-pixel luma delta. It is mapped through INFERNO so
    that dark purple is untouched and yellow is a large lift. The colourmap is
    a visualisation aid, not data: the numbers behind it are in the summary.
    """
    delta = np.abs(luminance(after_u8) - luminance(before_u8))
    peak = float(delta.max())
    scaled = np.zeros_like(delta) if peak < 1e-6 else delta / peak
    heat = cv2.applyColorMap((scaled * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)


def difference_summary(before_u8: np.ndarray, after_u8: np.ndarray) -> dict[str, Any]:
    delta = luminance(after_u8) - luminance(before_u8)
    absolute = np.abs(delta)
    changed = float(np.count_nonzero(absolute >= 1.0)) / absolute.size
    return {
        "mean_shift": round(float(delta.mean()), 3),
        "mean_absolute_shift": round(float(absolute.mean()), 3),
        "max_absolute_shift": round(float(absolute.max()), 3),
        "pixels_changed_pct": round(100.0 * changed, 2),
        "note": "Luma deltas in 0-255 units. Positive mean shift means the frame was lifted.",
    }


def compare(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Before/after deltas for every metric, with direction of improvement."""
    out: dict[str, Any] = {}
    for key, b in before.items():
        a = after.get(key, {})
        bv, av = b.get("value"), a.get("value")
        delta = None if bv is None or av is None else round(av - bv, 4)
        pct = None
        if delta is not None and bv not in (None, 0):
            pct = round(100.0 * delta / abs(bv), 2)
        # "neutral" metrics (brightness, colourfulness) have no better direction:
        # a frame lifted into grey mush would score well on both.
        # A delta of zero is neither a gain nor a cost; colouring it as a
        # regression would misreport "nothing changed" as "something got worse".
        improved = None
        direction = b.get("better")
        if delta is not None and direction in ("higher", "lower") and abs(delta) > 1e-9:
            improved = delta > 0 if direction == "higher" else delta < 0
        out[key] = {
            "label": b.get("label"),
            "kind": b.get("kind"),
            "unit": b.get("unit"),
            "before": bv,
            "after": av,
            "delta": delta,
            "delta_pct": pct,
            "improved": improved,
        }
    return out


def pixel_telemetry(before_u8: np.ndarray, after_u8: np.ndarray) -> dict[str, Any]:
    """Coarse accounting of how pixels moved between tonal zones."""
    b = np.clip(luminance(before_u8), 0, 255)
    a = np.clip(luminance(after_u8), 0, 255)
    zones = [("shadows", 0, 64), ("midtones", 64, 192), ("highlights", 192, 256)]
    total = b.size
    rows = []
    for name, lo, hi in zones:
        before_n = int(np.count_nonzero((b >= lo) & (b < hi)))
        after_n = int(np.count_nonzero((a >= lo) & (a < hi)))
        rows.append({
            "zone": name,
            "range": f"{lo}-{hi - 1}",
            "before_pct": round(100.0 * before_n / total, 2),
            "after_pct": round(100.0 * after_n / total, 2),
        })
    return {
        "total_pixels": int(total),
        "zones": rows,
        "recovered_from_black": int(
            np.count_nonzero((b <= SHADOW_CLIP_LEVEL) & (a > SHADOW_CLIP_LEVEL))
        ),
        "pushed_to_white": int(
            np.count_nonzero((b < HIGHLIGHT_CLIP_LEVEL) & (a >= HIGHLIGHT_CLIP_LEVEL))
        ),
    }
