"""Generate a synthetic low-light lunar-style test frame.

This is NOT lunar data. It is procedurally generated terrain that shares the
statistical awkwardness of real low-light planetary imagery -- a crushed
histogram, read noise, hot pixels, low-angle shadowing -- so the pipeline can
be exercised without shipping someone else's imagery.

Usage:
    python tools/make_sample.py [--out samples/synthetic_lunar.png] [--seed 7]
"""
from __future__ import annotations

import argparse
import os

import numpy as np
from PIL import Image


def _value_noise(shape: tuple[int, int], cells: int, rng: np.random.Generator) -> np.ndarray:
    """Smooth noise by bicubic-upsampling a coarse random lattice."""
    h, w = shape
    lattice = rng.random((cells + 1, cells + 1)).astype(np.float32)
    img = Image.fromarray((lattice * 255).astype(np.uint8), mode="L")
    return np.asarray(img.resize((w, h), Image.BICUBIC), dtype=np.float32) / 255.0


def fractal_terrain(shape: tuple[int, int], rng: np.random.Generator, octaves: int = 6) -> np.ndarray:
    """Fractional Brownian motion: the standard model for rough terrain."""
    height = np.zeros(shape, dtype=np.float32)
    amplitude, total, cells = 1.0, 0.0, 3
    for _ in range(octaves):
        height += amplitude * _value_noise(shape, cells, rng)
        total += amplitude
        amplitude *= 0.52
        cells *= 2
    return height / total


def add_craters(height: np.ndarray, rng: np.random.Generator, count: int = 42) -> np.ndarray:
    """Bowl depressions with raised rims, sized on a power law like real fields."""
    h, w = height.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    out = height.copy()
    for _ in range(count):
        radius = float(np.clip(rng.pareto(1.7) * 11 + 7, 7, min(h, w) * 0.17))
        cy, cx = rng.uniform(0, h), rng.uniform(0, w)
        dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) / radius
        depth = 0.11 + 0.10 * rng.random()

        bowl = np.where(dist < 1.0, -depth * (1.0 - dist ** 2), 0.0)
        rim = np.where(
            (dist >= 0.82) & (dist < 1.5),
            depth * 0.7 * np.exp(-((dist - 1.0) ** 2) / 0.035),
            0.0,
        )
        out += bowl + rim
    return out


def shade(height: np.ndarray, sun_azimuth: float = 2.5, sun_elevation: float = 0.13) -> np.ndarray:
    """Lambertian shading from a low sun, which is what makes relief readable."""
    gy, gx = np.gradient(height.astype(np.float32))
    scale = 85.0
    nx, ny, nz = -gx * scale, -gy * scale, np.ones_like(height)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz)

    lx = np.cos(sun_azimuth) * np.cos(sun_elevation)
    ly = np.sin(sun_azimuth) * np.cos(sun_elevation)
    lz = np.sin(sun_elevation)

    lit = (nx * lx + ny * ly + nz * lz) / norm
    return np.clip(lit, 0.0, 1.0)


def build(width: int, height_px: int, seed: int, exposure: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shape = (height_px, width)

    terrain = fractal_terrain(shape, rng)
    terrain = add_craters(terrain, rng)
    lit = shade(terrain)

    # Albedo variation: maria are darker than highlands, with fine regolith mottle.
    albedo = 0.55 + 0.45 * _value_noise(shape, 4, rng)
    albedo *= 0.93 + 0.07 * _value_noise(shape, 40, rng)

    radiance = lit * albedo
    # Ambient term: even shadowed floors catch a little scattered light.
    radiance = 0.035 + 0.965 * radiance

    signal = radiance * exposure

    # Sensor model. Photon shot noise scales with the square root of signal;
    # read noise is additive and is what dominates a frame this dark.
    photons = np.clip(signal * 255.0, 0, None)
    shot = rng.normal(0.0, np.sqrt(photons + 1e-6) * 0.55)
    read = rng.normal(0.0, 2.1, shape)
    frame = photons + shot + read

    # A faint colour cast, as a real sensor's channels do not match exactly.
    rgb = np.stack([frame * 1.02, frame * 1.0, frame * 0.96], axis=2)

    # A scattering of hot pixels, which every real detector has.
    hot = rng.random(shape) < 0.00008
    rgb[hot] = rng.uniform(150, 255, (int(hot.sum()), 3))

    return np.clip(rgb, 0, 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=os.path.join("samples", "synthetic_lunar.png"))
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--exposure", type=float, default=0.075,
                        help="Lower means a darker frame. 0.075 is a hard case.")
    args = parser.parse_args()

    frame = build(args.width, args.height, args.seed, args.exposure)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    Image.fromarray(frame).save(args.out)

    luma = frame.astype(np.float32) @ np.array([0.2126, 0.7152, 0.0722])
    print(f"Wrote {args.out}  {args.width}x{args.height}")
    print(f"  mean luma {luma.mean():.1f}/255, median {np.median(luma):.1f}, "
          f"p99 {np.percentile(luma, 99):.1f}")
    print(f"  {100 * np.count_nonzero(luma <= 2) / luma.size:.1f}% of pixels are crushed to black")
    print("  Synthetic data. Not a real lunar observation.")


if __name__ == "__main__":
    main()
