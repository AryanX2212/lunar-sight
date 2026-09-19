# Lunar Low-Light Lab

A local web app for recovering structure from underexposed planetary imagery,
and for measuring whether the recovery actually worked.

Runs entirely on your machine at **http://127.0.0.1:8000/**. Nothing is uploaded
anywhere, and results live in memory only.

---

## Quick start

```bash
pip install -r requirements.txt
python tools/make_sample.py     # optional: a synthetic test frame
python run.py
```

Open http://127.0.0.1:8000/, drop in a frame, pick a method, press **Enhance frame**.

Options:

```bash
python run.py --port 8080
python run.py --debug          # reloader and verbose errors
```

---

## What it does

Load a dark frame and the app runs it through one of four enhancement methods,
then reports, side by side:

- a draggable before/after comparison, the enhanced frame alone, and a **change
  map** showing where the enhancement moved pixels and by how much
- eleven **measurements** taken from the pixels, before and after, with the
  direction of improvement marked
- **luma histograms** for both versions, and a breakdown of how pixels moved
  between shadows, midtones and highlights
- the exact list of **operations applied**, with the method's own caveat

### The four methods

| Method | What it is | When it helps |
|---|---|---|
| **Scientific** | Luminance-only pipeline: shadow-weighted denoise, percentile black point, adaptive gamma, soft highlight shoulder, unsharp mask | The default. Preserves colour relationships and does not blow the highlights |
| **Adaptive equalisation** | Adaptive pre-normalisation, then CLAHE on the L channel | Strongest at local texture like crater rims |
| **Zero-DCE** | Learned per-pixel tone curves, run as a NumPy forward pass | Smooth, natural lift that cannot invert tonal ordering |
| **Fixed gamma** | A plain gamma lift | The baseline. Included so you can see what the cheap answer costs |

They are deliberately different approaches rather than four tunings of one
curve. Fixed gamma is there as a control: it amplifies noise exactly as hard as
it amplifies signal, and the measurements make that visible.

---

## Measured vs estimated

Every metric is tagged, and the interface shows the tag:

- **measured** — computed directly from the pixels. Mean brightness, RMS
  contrast, entropy, levels occupied, 1–99% span, clipping percentages,
  acutance, colourfulness.
- **estimated** — derived from an *assumption* about the image. Noise level uses
  Immerkaer's estimator, which assumes high-frequency content in flat regions is
  sensor noise; real scene texture inflates it. Signal-to-noise inherits that
  assumption.

Read the estimated rows as a before-and-after comparison of the same frame, not
as absolute figures. Nothing is ever filled in with a plausible default: a
metric that cannot be computed comes back as `null`.

**The output is not radiometrically calibrated.** All four methods apply
non-linear tone mapping, so pixel values are no longer proportional to scene
radiance. Use the results to *see* structure, not to do photometry.

---

## Zero-DCE weights

Out of the box there are no weights, and the Zero-DCE mode uses an analytic
illumination-curve approximation instead. This is **not a neural network**, and
it says so everywhere it appears: in `/api/health`, in the payload
(`"neural": false`), in the engine chip in the header, and in the result card.

To use the real thing, convert a PyTorch checkpoint once:

```bash
pip install torch                    # conversion only
python tools/convert_weights.py --checkpoint Epoch99.pth
```

That writes `models/zero_dce_weights.npz`. Restart the server and `/api/health`
will report the weights as loaded. The server itself never imports torch —
inference runs in NumPy — so the runtime stays small.

Checkpoints from the reference implementation (`Li-Chongyi/Zero-DCE`) work
directly; `module.`-prefixed keys from a DataParallel save are handled.

---

## API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/enhance` | multipart: `image` plus options. Returns the full analysis |
| `GET` | `/api/results/<id>` | The stored payload, identical to what the POST returned |
| `GET` | `/api/results/<id>/image/<name>` | `original`, `enhanced` or `difference` as PNG. `?download=1` to attach |
| `DELETE` | `/api/results/<id>` | Drop a result immediately |
| `GET` | `/api/health` | Liveness, engine status, store stats |
| `GET` | `/api/config` | Limits, defaults and the method list |
| `POST` | `/api/maintenance/sweep` | Force expiry collection |

Options on `/api/enhance`: `mode` (`scientific`, `clahe`, `zerodce`, `naive`),
`gamma` (1–5), `clahe_clip` (0.5–8), `clahe_grid` (2–32), `denoise`,
`preserve_highlights`.

Every failure returns the same shape, so the client never parses prose:

```json
{ "error": { "code": "unsupported_image", "message": "...", "details": {} } }
```

Codes: `validation_failed` (400), `unsupported_image` (415),
`payload_too_large` (413), `not_found` (404), `result_expired` (410),
`processing_failed` (500).

```bash
curl -F image=@samples/synthetic_lunar.png -F mode=scientific \
     http://127.0.0.1:8000/api/enhance
```

---

## How uploads are handled

- Format is detected from the file's **magic bytes**, never its extension or the
  browser-supplied content type. A JPEG named `.png` is treated as a JPEG; an
  HTML file named `.png` is rejected.
- The header is inspected before pixels are loaded, so a decompression bomb is
  refused before it allocates.
- 16-bit TIFFs are scaled by their observed range, not a fixed 65535, because
  instrument data often occupies a narrow band.
- RGBA is composited onto black; EXIF orientation is honoured.
- Frames longer than 2400px on the long edge are downscaled, and the result
  says so.

Limits: 16 MB, 40 MP, both sides at least 16px. All configurable in
`backend/config.py` or via `LUNAR_*` environment variables.

---

## Cleanup

Results hold decoded PNG bytes, so nothing is kept indefinitely. Three
independent mechanisms bound memory: a 30-minute TTL, a 64-result capacity cap
that evicts oldest-first, and a background sweeper that reclaims expired entries
even when no requests arrive. Everything is cleared on shutdown.

Storage is process-local and deliberately not persistent — this is a local
analysis tool, and your imagery should not outlive the session.

---

## Tests

```bash
python -m unittest discover -s tests -v
```

52 tests using only the standard library, covering validation and the
decode-by-content rule, all four enhancement modes, metric correctness against
known values, the storage lifecycle including the background sweeper, and the
full HTTP surface with its error contract. `pytest` will collect them unchanged
if you prefer it.

Several tests are regressions for bugs found during the build — notably that
`scientific` mode must not crush shadows further (the float-denoise fix) and
that the Zero-DCE curve must never invert tonal ordering.

---

## Security

The app binds to `127.0.0.1` and has **no authentication**. Do not expose it on
`0.0.0.0` on a shared network without putting something in front of it.

It renders user-supplied images and serves them back, so responses carry a
strict CSP with no `unsafe-inline`, plus `nosniff`, `DENY` framing and
`no-referrer`. There is no inline script anywhere in the frontend.

---

## Layout

```
run.py                     entry point, binds 127.0.0.1:8000
backend/
  config.py                every tunable, overridable via LUNAR_* env vars
  errors.py                the single error contract
  validation.py            upload validation and safe decoding
  metrics.py               measurements, histograms, difference maps
  enhance.py               naive, scientific and CLAHE modes
  zerodce.py               NumPy DCE-Net, weight loading, analytic fallback
  storage.py               TTL store with eviction and background sweeper
  pipeline.py              orchestration
  routes.py                HTTP surface
  security.py              response hardening
  app.py                   application factory and error handling
frontend/
  templates/index.html
  static/css/app.css
  static/js/app.js         no framework, no build step
tools/
  make_sample.py           synthetic lunar-style test frame
  convert_weights.py       PyTorch checkpoint -> .npz
tests/test_api.py
```

The frontend fetches the method list and the upload limits from `/api/config`
rather than hard-coding them, so the interface cannot drift out of sync with
the backend.

---

## A note on the sample

`tools/make_sample.py` generates **synthetic** terrain — fractional Brownian
motion, power-law crater sizes, Lambertian shading from a low sun, then a sensor
model with shot noise, read noise and hot pixels. It shares the statistical
awkwardness of real low-light planetary imagery so the pipeline can be exercised
honestly.

It is not a real lunar observation, and nothing in this project ships anyone
else's imagery.
