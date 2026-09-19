"""The four enhancement modes.

These are deliberately different approaches, not four tunings of one curve:

  naive       Fixed gamma lift. The baseline every other mode is judged against.
              Included so the interface can show what the cheap answer costs:
              it amplifies noise exactly as hard as it amplifies signal.
  scientific  Luminance-only pipeline. Percentile black-point, adaptive gamma
              from the image's own median, bilateral denoise in the dark end,
              then unsharp masking. Chroma is carried through untouched, so
              colour relationships survive.
  clahe       Contrast-limited adaptive histogram equalisation on the L channel
              in LAB. Strongest at pulling out local texture like crater rims.
  zerodce     Learned per-pixel tone curves (see zerodce.py).

All modes take and return uint8 RGB and must be non-destructive: the caller's
array is never modified in place.
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .config import CONFIG


def _to_lab(rgb_u8: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2LAB)


def _from_lab(lab: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def _apply_luma(rgb_u8: np.ndarray, new_luma_f: np.ndarray) -> np.ndarray:
    """Rebuild an RGB image from a modified luminance channel.

    Works in LAB so the a/b chroma channels pass through untouched. The
    alternative, scaling RGB by a luma ratio, drags saturation up with
    brightness and turns grey regolith faintly blue.
    """
    lab = _to_lab(rgb_u8)
    lab[:, :, 0] = np.clip(new_luma_f, 0, 255).astype(np.uint8)
    return _from_lab(lab)


def _black_point(channel_f: np.ndarray, percentile: float = 0.5) -> np.ndarray:
    """Subtract the noise floor, then rescale to reclaim the lost range."""
    floor = float(np.percentile(channel_f, percentile))
    if floor <= 0:
        return channel_f
    lifted = np.clip(channel_f - floor, 0, None)
    span = 255.0 - floor
    return lifted * (255.0 / span) if span > 1e-6 else lifted


def _unsharp(channel_u8: np.ndarray, amount: float = 0.6, radius: int = 3) -> np.ndarray:
    blurred = cv2.GaussianBlur(channel_u8, (0, 0), radius)
    return cv2.addWeighted(channel_u8, 1.0 + amount, blurred, -amount, 0)


def _shadow_weighted_denoise(l_f: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Bilateral denoise in float, blended in proportional to how dark each pixel was.

    Noise is most visible where the signal was weakest. Denoising uniformly
    would soften the well-lit crater rims for no benefit, so the filtered
    result is mixed in only where it is needed.

    This must run in float32, not uint8. In a badly underexposed frame most of
    the image occupies luma 0-2, and an 8-bit filter can only ever return those
    same integers. The quantization then survives into the gamma stage, which
    maps 0 to 0 and 2 to ~90 and tears the shadows into black speckle. Filtering
    in float lets those pixels resolve to fractional values and lift smoothly.
    """
    l_f = l_f.astype(np.float32, copy=False)
    filtered = cv2.bilateralFilter(l_f, d=7, sigmaColor=8.0, sigmaSpace=7)
    weight = np.clip(1.0 - (l_f / 96.0), 0.0, 1.0) * strength
    return l_f * (1 - weight) + filtered * weight


# --------------------------------------------------------------------------
# Modes
# --------------------------------------------------------------------------

def enhance_naive(rgb_u8: np.ndarray, gamma: float, **_: Any) -> tuple[np.ndarray, dict]:
    inv = 1.0 / max(gamma, 1e-6)
    lut = np.array([((i / 255.0) ** inv) * 255 for i in range(256)], dtype=np.uint8)
    out = cv2.LUT(rgb_u8, lut)
    return out, {
        "method": "Fixed gamma lift",
        "gamma": round(gamma, 3),
        "operations": [f"Per-channel gamma {gamma:.2f} via 256-entry lookup table"],
        "caveat": "Applied identically to every pixel, so sensor noise in the "
                  "shadows is amplified by the same factor as real detail.",
    }


def enhance_scientific(
    rgb_u8: np.ndarray,
    gamma: float,
    denoise: bool = True,
    preserve_highlights: bool = True,
    **_: Any,
) -> tuple[np.ndarray, dict]:
    lab = _to_lab(rgb_u8)
    l_f = lab[:, :, 0].astype(np.float32)
    steps: list[str] = []

    if denoise:
        l_f = _shadow_weighted_denoise(l_f)
        steps.append("Shadow-weighted bilateral denoise on luminance, in float")

    l_f = _black_point(l_f, percentile=0.5)
    steps.append("Black point set at the 0.5th percentile, range rescaled")


    # Adaptive gamma: aim the image's median at mid-grey rather than forcing a
    # fixed exponent. A frame at median 8 needs far more lift than one at 40.
    # Everything here is done in terms of the exponent p, where out = in ** p
    # and p < 1 brightens; gamma is just 1/p, reported back for the operator.
    median = float(np.clip(np.median(l_f), 1.0, 250.0))
    target = 128.0
    p_adaptive = float(np.log(target / 255.0) / np.log(median / 255.0))
    p_requested = 1.0 / max(gamma, 1e-6)
    # Blend toward the operator's requested gamma so the control still bites.
    p = float(np.clip(0.65 * p_adaptive + 0.35 * p_requested, 1.0 / 6.0, 1.0))
    effective = 1.0 / p
    normalised = np.clip(l_f / 255.0, 0, 1)
    l_f = np.power(normalised, p) * 255.0
    steps.append(
        f"Adaptive gamma {effective:.2f} (median {median:.1f} steered toward {target:.0f})"
    )

    if preserve_highlights:
        # Soft shoulder: compress the top end instead of letting it clip flat.
        knee = 205.0
        over = l_f > knee
        headroom = 255.0 - knee
        l_f[over] = knee + headroom * np.tanh((l_f[over] - knee) / headroom)
        steps.append("Soft-shoulder highlight rolloff above luma 205")

    l_u8 = np.clip(l_f, 0, 255).astype(np.uint8)
    l_u8 = _unsharp(l_u8, amount=0.5, radius=3)
    steps.append("Unsharp mask, amount 0.5, radius 3px")

    out = _apply_luma(rgb_u8, l_u8.astype(np.float32))
    return out, {
        "method": "Luminance-only scientific pipeline",
        "gamma": round(effective, 3),
        "requested_gamma": round(gamma, 3),
        "operations": steps,
        "caveat": "Tone mapping is non-linear, so pixel values are no longer "
                  "proportional to scene radiance. Do not use the output for "
                  "photometry; use it to see structure.",
    }


def enhance_clahe(
    rgb_u8: np.ndarray,
    clahe_clip: float,
    clahe_grid: int,
    denoise: bool = True,
    **_: Any,
) -> tuple[np.ndarray, dict]:
    lab = _to_lab(rgb_u8)
    l_f = lab[:, :, 0].astype(np.float32)
    steps: list[str] = []

    if denoise:
        l_f = _shadow_weighted_denoise(l_f, strength=0.8)
        steps.append("Shadow-weighted bilateral denoise before equalisation")

    # CLAHE redistributes contrast within tiles; it does not create range. On a
    # frame whose histogram is compressed into the bottom few levels there is
    # almost nothing to redistribute, so normalise the tone scale first.
    l_f = _black_point(l_f, percentile=0.5)
    median = float(np.clip(np.median(l_f), 1.0, 250.0))
    p = float(np.clip(np.log(96.0 / 255.0) / np.log(median / 255.0), 1.0 / 6.0, 1.0))
    l_f = np.power(np.clip(l_f / 255.0, 0, 1), p) * 255.0
    steps.append(f"Adaptive pre-normalisation, gamma {1.0 / p:.2f}, before equalisation")

    grid = int(np.clip(clahe_grid, 2, 32))
    clahe = cv2.createCLAHE(clipLimit=float(clahe_clip), tileGridSize=(grid, grid))
    lab[:, :, 0] = clahe.apply(np.clip(l_f, 0, 255).astype(np.uint8))
    steps.append(f"CLAHE on L, clip limit {clahe_clip:.2f}, {grid}x{grid} tiles")

    return _from_lab(lab), {
        "method": "Contrast-limited adaptive histogram equalisation",
        "clip_limit": round(float(clahe_clip), 2),
        "tile_grid": grid,
        "operations": steps,
        "caveat": "Equalisation is local, so brightness is no longer comparable "
                  "between different regions of the frame. Tile edges can show "
                  "as faint seams at high clip limits.",
    }


MODES = {
    "naive": enhance_naive,
    "scientific": enhance_scientific,
    "clahe": enhance_clahe,
}


def enhance(rgb_u8: np.ndarray, options: dict[str, Any]) -> tuple[np.ndarray, dict]:
    """Dispatch to the requested mode. zerodce is routed in pipeline.py."""
    mode = options.get("mode", CONFIG.DEFAULT_MODE)
    func = MODES.get(mode)
    if func is None:
        raise KeyError(f"enhance() cannot handle mode '{mode}'")
    return func(rgb_u8, **{k: v for k, v in options.items() if k != "mode"})
