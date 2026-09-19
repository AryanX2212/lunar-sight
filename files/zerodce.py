"""Zero-DCE: learned per-pixel tone curves, in pure NumPy.

Zero-DCE (Guo et al., CVPR 2020) does not output an image. It outputs the
parameters of a curve, and the curve is applied iteratively:

    LE(x, a) = x + a * (x^2 - x)

applied n times with a different per-pixel, per-channel map `a` each time. The
map comes from a small 7-layer convolutional network with symmetric skip
concatenation. Because the operation is a monotonic curve in 0..1, the output
cannot invert the ordering of tones: brighter input stays brighter. That is
what makes it safe for imagery you intend to read structurally.

This module runs inference in NumPy so the server has no deep-learning runtime
dependency. Weights are loaded from an .npz produced by tools/convert_weights.py.

If no weights are present, `analytic_curve_enhance` is used instead. It is NOT
a neural network and never claims to be: every payload it produces is tagged
neural=False and the interface labels it accordingly.
"""
from __future__ import annotations

import os
import threading
from typing import Any

import cv2
import numpy as np

from .config import CONFIG

# Layer names expected in the .npz, in execution order.
LAYER_NAMES = ("e_conv1", "e_conv2", "e_conv3", "e_conv4", "e_conv5", "e_conv6", "e_conv7")


# ---------------------------------------------------------------------------
# NumPy convolution
# ---------------------------------------------------------------------------

def _conv2d(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """3x3 same-padding convolution.

    x      (H, W, Cin) float32
    weight (Cout, Cin, 3, 3) float32, PyTorch layout
    """
    kh, kw = weight.shape[2], weight.shape[3]
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(x, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode="reflect")

    # Sliding windows give an (H, W, Cin, kh, kw) view with no copy.
    windows = np.lib.stride_tricks.sliding_window_view(padded, (kh, kw), axis=(0, 1))
    # -> (H, W, Cin, kh, kw); contract against (Cout, Cin, kh, kw)
    out = np.tensordot(windows, weight, axes=([2, 3, 4], [1, 2, 3]))
    return (out + bias.reshape(1, 1, -1)).astype(np.float32)


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0, out=x)


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------

class WeightStore:
    """Loads Zero-DCE weights once and reports honestly on what it found."""

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        self._weights: dict[str, np.ndarray] | None = None
        self._status = "not_loaded"
        self._detail = "Weights have not been checked yet."

    def _load(self) -> None:
        if not os.path.exists(self.path):
            self._status = "missing"
            self._detail = (
                f"No weight file at {self.path}. Run tools/convert_weights.py to "
                "convert a PyTorch Zero-DCE checkpoint."
            )
            return
        try:
            data = np.load(self.path)
            weights: dict[str, np.ndarray] = {}
            for name in LAYER_NAMES:
                wk, bk = f"{name}.weight", f"{name}.bias"
                if wk not in data or bk not in data:
                    raise KeyError(f"missing '{wk}' or '{bk}'")
                weights[wk] = np.ascontiguousarray(data[wk], dtype=np.float32)
                weights[bk] = np.ascontiguousarray(data[bk], dtype=np.float32)

            # Shape contract: 7 layers, 32 filters, final layer 3*n_iter outputs.
            final = weights["e_conv7.weight"]
            if final.shape[0] % 3 != 0:
                raise ValueError(
                    f"final layer has {final.shape[0]} output channels, expected a multiple of 3"
                )
            self._weights = weights
            self._iterations = final.shape[0] // 3
            self._status = "loaded"
            self._detail = (
                f"Loaded from {self.path}; {self._iterations} curve iterations."
            )
        except Exception as exc:
            self._weights = None
            self._status = "invalid"
            self._detail = f"Weight file could not be read: {exc}"

    def get(self) -> dict[str, np.ndarray] | None:
        with self._lock:
            if self._status == "not_loaded":
                self._load()
            return self._weights

    def status(self) -> dict[str, Any]:
        with self._lock:
            if self._status == "not_loaded":
                self._load()
            return {
                "status": self._status,
                "detail": self._detail,
                "path": self.path,
                "neural_available": self._status == "loaded",
            }


WEIGHTS = WeightStore(CONFIG.WEIGHTS_PATH)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _estimate_curves(rgb_f: np.ndarray, w: dict[str, np.ndarray]) -> np.ndarray:
    """Run the DCE-Net. Input (H, W, 3) in 0..1. Returns (H, W, 3*n_iter)."""
    e1 = _relu(_conv2d(rgb_f, w["e_conv1.weight"], w["e_conv1.bias"]))
    e2 = _relu(_conv2d(e1, w["e_conv2.weight"], w["e_conv2.bias"]))
    e3 = _relu(_conv2d(e2, w["e_conv3.weight"], w["e_conv3.bias"]))
    e4 = _relu(_conv2d(e3, w["e_conv4.weight"], w["e_conv4.bias"]))
    # Symmetric skip concatenation, matching the reference implementation.
    e5 = _relu(_conv2d(np.concatenate([e3, e4], axis=2), w["e_conv5.weight"], w["e_conv5.bias"]))
    e6 = _relu(_conv2d(np.concatenate([e2, e5], axis=2), w["e_conv6.weight"], w["e_conv6.bias"]))
    return np.tanh(_conv2d(np.concatenate([e1, e6], axis=2), w["e_conv7.weight"], w["e_conv7.bias"]))


def _apply_curves(rgb_f: np.ndarray, curves: np.ndarray, iterations: int) -> np.ndarray:
    x = rgb_f
    for i in range(iterations):
        a = curves[:, :, i * 3:(i + 1) * 3]
        x = x + a * (np.square(x) - x)
    return np.clip(x, 0.0, 1.0)


def zero_dce_enhance(rgb_u8: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Neural enhancement. Raises RuntimeError if weights are unavailable."""
    weights = WEIGHTS.get()
    if weights is None:
        raise RuntimeError(WEIGHTS.status()["detail"])

    h, w_px = rgb_u8.shape[:2]
    longest = max(h, w_px)
    # Curve maps are smooth and low-frequency, so they can be estimated at
    # reduced resolution and upsampled. This keeps NumPy inference tractable
    # without visibly changing the result.
    scale = min(1.0, CONFIG.DCE_MAX_EDGE / longest)
    if scale < 1.0:
        small = cv2.resize(
            rgb_u8, (max(1, int(w_px * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        small = rgb_u8

    curves = _estimate_curves(small.astype(np.float32) / 255.0, weights)
    if curves.shape[:2] != (h, w_px):
        curves = cv2.resize(curves, (w_px, h), interpolation=cv2.INTER_LINEAR)
        if curves.ndim == 2:  # cv2 drops the last axis when it is 1
            curves = curves[:, :, None]

    iterations = curves.shape[2] // 3
    out = _apply_curves(rgb_u8.astype(np.float32) / 255.0, curves, iterations)
    return (out * 255.0).round().astype(np.uint8), {
        "method": "Zero-DCE (deep curve estimation)",
        "neural": True,
        "iterations": iterations,
        "inference_scale": round(scale, 3),
        "operations": [
            "7-layer DCE-Net with symmetric skip concatenation, NumPy inference",
            f"{iterations} iterations of LE(x,a) = x + a(x^2 - x)",
        ],
        "caveat": "The curve is monotonic per pixel, so tonal ordering is "
                  "preserved, but the mapping is learned and non-linear. The "
                  "output is not radiometrically calibrated.",
    }


# ---------------------------------------------------------------------------
# Fallback
# ---------------------------------------------------------------------------

def analytic_curve_enhance(rgb_u8: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Retinex-style illumination correction. NOT a neural network.

    Estimates an illumination map as the per-pixel channel maximum, smooths it
    with an edge-preserving filter, then divides it out. It uses the same
    iterative curve form as Zero-DCE, but the curve parameter comes from this
    estimate rather than from anything learned.
    """
    rgb_f = rgb_u8.astype(np.float32) / 255.0

    illumination = rgb_f.max(axis=2)
    illumination = cv2.bilateralFilter(illumination, d=9, sigmaColor=0.12, sigmaSpace=9)
    illumination = np.clip(illumination, 0.02, 1.0)

    # a in [-1, 0): the darker the local illumination, the stronger the lift.
    shape = np.clip(1.0 - illumination, 0.0, 1.0)[:, :, None]
    iterations = int(np.clip(CONFIG.DCE_ITERATIONS, 1, 16))

    def run(strength: float, source: np.ndarray, shape_map: np.ndarray) -> np.ndarray:
        x = source
        a = -shape_map * strength
        for _ in range(iterations):
            x = x + a * (np.square(x) - x)
        return np.clip(x, 0.0, 1.0)

    # The curve compounds across iterations, so a fixed strength blows out dark
    # frames. Solve for the strength that lands the median near mid-grey, using
    # a thumbnail so the search costs almost nothing.
    thumb = cv2.resize(rgb_f, (96, 96), interpolation=cv2.INTER_AREA)
    thumb_shape = cv2.resize(shape[:, :, 0], (96, 96), interpolation=cv2.INTER_AREA)[:, :, None]
    target_median = 0.42
    low, high = 0.0, 0.95
    strength = 0.3
    for _ in range(18):  # bisection; the response is monotonic in strength
        strength = 0.5 * (low + high)
        if float(np.median(run(strength, thumb, thumb_shape))) < target_median:
            low = strength
        else:
            high = strength

    x = run(strength, rgb_f, shape)

    return (x * 255.0).round().astype(np.uint8), {
        "method": "Analytic illumination curve (fallback)",
        "neural": False,
        "iterations": iterations,
        "operations": [
            "Illumination map from per-pixel channel maximum, bilaterally smoothed",
            f"{iterations} iterations of LE(x,a) = x + a(x^2 - x) with analytic a",
            f"Curve strength {strength:.3f}, solved to place the median near mid-grey",
        ],
        "caveat": "No Zero-DCE weights were loaded, so this is a hand-written "
                  "approximation, not a trained network. Results differ from "
                  "published Zero-DCE output.",
    }


def enhance_zerodce(rgb_u8: np.ndarray, **_: Any) -> tuple[np.ndarray, dict[str, Any]]:
    """Neural path when weights exist, clearly-labelled analytic path when not."""
    if WEIGHTS.get() is not None:
        try:
            return zero_dce_enhance(rgb_u8)
        except Exception as exc:
            out, report = analytic_curve_enhance(rgb_u8)
            report["fallback_reason"] = f"Neural inference failed: {exc}"
            return out, report
    out, report = analytic_curve_enhance(rgb_u8)
    report["fallback_reason"] = WEIGHTS.status()["detail"]
    return out, report
