"""Test suite for the Lunar Low-Light Lab.

Uses unittest from the standard library so it runs with no extra install:

    python -m unittest discover -s tests -v

pytest will also collect and run these unchanged if you prefer it.
"""
from __future__ import annotations

import io
import os
import sys
import time
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.app import create_app                      # noqa: E402
from backend.enhance import enhance                     # noqa: E402
from backend.errors import NotFound, ResultExpired      # noqa: E402
from backend.metrics import compare, compute_metrics    # noqa: E402
from backend.security import HEADERS                    # noqa: E402
from backend.storage import STORE, ResultStore          # noqa: E402
from backend.validation import decode_image, validate_options  # noqa: E402
from backend.zerodce import analytic_curve_enhance, enhance_zerodce  # noqa: E402


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def dark_frame(width: int = 120, height: int = 96, seed: int = 3) -> np.ndarray:
    """A deterministic underexposed frame with real structure in it."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    terrain = (
        6.0
        + 4.0 * np.sin(xx / 13.0)
        + 3.0 * np.cos(yy / 9.0)
        + rng.normal(0, 1.2, (height, width))
    )
    rgb = np.stack([terrain * 1.03, terrain, terrain * 0.95], axis=2)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def encode(arr: np.ndarray, fmt: str = "PNG") -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(arr).save(buffer, format=fmt)
    return buffer.getvalue()


def upload(data: bytes = None, name: str = "frame.png", **options):
    payload = {"image": (io.BytesIO(data if data is not None else encode(dark_frame())), name)}
    payload.update({k: str(v) for k, v in options.items()})
    return payload


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------

class TestValidation(unittest.TestCase):

    def test_decodes_png_to_rgb(self):
        arr, meta = decode_image(encode(dark_frame()), "frame.png")
        self.assertEqual(arr.shape, (96, 120, 3))
        self.assertEqual(arr.dtype, np.uint8)
        self.assertEqual(meta["detected_format"], "PNG")

    def test_format_comes_from_bytes_not_extension(self):
        """A JPEG named .png must be recognised as a JPEG."""
        _, meta = decode_image(encode(dark_frame(), "JPEG"), "lying_name.png")
        self.assertEqual(meta["detected_format"], "JPEG")

    def test_greyscale_is_promoted_to_three_channels(self):
        grey = Image.fromarray(dark_frame()[:, :, 0], mode="L")
        buffer = io.BytesIO()
        grey.save(buffer, format="PNG")
        arr, _ = decode_image(buffer.getvalue(), "grey.png")
        self.assertEqual(arr.shape[2], 3)

    def test_rgba_is_composited_not_dropped(self):
        rgba = np.dstack([dark_frame(), np.full((96, 120), 255, np.uint8)])
        buffer = io.BytesIO()
        Image.fromarray(rgba, mode="RGBA").save(buffer, format="PNG")
        arr, _ = decode_image(buffer.getvalue(), "alpha.png")
        self.assertEqual(arr.shape[2], 3)

    def test_rejects_non_image(self):
        from backend.errors import UnsupportedImage
        with self.assertRaises(UnsupportedImage):
            decode_image(b"#!/bin/sh\necho not an image\n", "evil.png")

    def test_rejects_empty_upload(self):
        from backend.errors import ValidationError
        with self.assertRaises(ValidationError):
            decode_image(b"", "empty.png")

    def test_rejects_tiny_image(self):
        from backend.errors import ValidationError
        with self.assertRaises(ValidationError):
            decode_image(encode(np.zeros((4, 4, 3), np.uint8)), "tiny.png")

    def test_oversize_image_is_downscaled(self):
        from backend.config import CONFIG
        big = np.zeros((200, CONFIG.MAX_EDGE + 400, 3), np.uint8)
        arr, meta = decode_image(encode(big), "wide.png")
        self.assertTrue(meta["resized_for_processing"])
        self.assertLessEqual(max(arr.shape[:2]), CONFIG.MAX_EDGE)

    def test_option_bounds_are_enforced(self):
        from backend.errors import ValidationError
        with self.assertRaises(ValidationError):
            validate_options({"mode": "scientific", "gamma": "99"})
        with self.assertRaises(ValidationError):
            validate_options({"mode": "does_not_exist"})
        with self.assertRaises(ValidationError):
            validate_options({"mode": "scientific", "gamma": "abc"})

    def test_option_defaults_fill_in(self):
        opts = validate_options({})
        self.assertEqual(opts["mode"], "scientific")
        self.assertIn("gamma", opts)


# --------------------------------------------------------------------------
# enhancement
# --------------------------------------------------------------------------

class TestEnhancement(unittest.TestCase):

    def setUp(self):
        self.frame = dark_frame()
        self.options = validate_options({})

    def _run(self, mode: str):
        opts = dict(self.options, mode=mode)
        if mode == "zerodce":
            return enhance_zerodce(self.frame, **opts)
        return enhance(self.frame, opts)

    def test_every_mode_preserves_shape_and_type(self):
        for mode in ("naive", "scientific", "clahe", "zerodce"):
            with self.subTest(mode=mode):
                out, report = self._run(mode)
                self.assertEqual(out.shape, self.frame.shape)
                self.assertEqual(out.dtype, np.uint8)
                self.assertIn("method", report)
                self.assertIn("operations", report)

    def test_every_mode_brightens_a_dark_frame(self):
        before = compute_metrics(self.frame)["mean_luminance"]["value"]
        for mode in ("naive", "scientific", "clahe", "zerodce"):
            with self.subTest(mode=mode):
                out, _ = self._run(mode)
                after = compute_metrics(out)["mean_luminance"]["value"]
                self.assertGreater(after, before)

    def test_input_is_not_modified_in_place(self):
        original = self.frame.copy()
        for mode in ("naive", "scientific", "clahe", "zerodce"):
            self._run(mode)
        np.testing.assert_array_equal(self.frame, original)

    def test_scientific_adapts_gamma_to_darkness(self):
        """A darker frame must receive more lift than a brighter one."""
        dark = np.clip(dark_frame() // 3, 0, 255).astype(np.uint8)
        bright = np.clip(dark_frame().astype(int) * 4, 0, 255).astype(np.uint8)
        _, dark_report = enhance(dark, dict(self.options, mode="scientific"))
        _, bright_report = enhance(bright, dict(self.options, mode="scientific"))
        self.assertGreater(dark_report["gamma"], bright_report["gamma"])

    def test_scientific_does_not_crush_shadows_further(self):
        """Regression: float denoising, so quantized shadows do not tear into speckle."""
        out, _ = enhance(self.frame, dict(self.options, mode="scientific"))
        before = compute_metrics(self.frame)["shadow_clipping"]["value"]
        after = compute_metrics(out)["shadow_clipping"]["value"]
        self.assertLessEqual(after, before + 0.5)

    def test_analytic_fallback_is_labelled_non_neural(self):
        _, report = analytic_curve_enhance(self.frame)
        self.assertFalse(report["neural"])
        self.assertIn("fallback", report["method"].lower())

    def test_zerodce_reports_its_provenance_either_way(self):
        _, report = enhance_zerodce(self.frame)
        self.assertIn("neural", report)
        if not report["neural"]:
            self.assertIn("fallback_reason", report)

    def test_curve_is_monotonic(self):
        """Zero-DCE's curve must not invert tonal ordering: a ramp stays sorted."""
        ramp = np.tile(np.arange(256, dtype=np.uint8), (32, 1))
        rgb = np.repeat(ramp[:, :, None], 3, axis=2)
        out, _ = enhance_zerodce(rgb)
        row = out[16, :, 0].astype(int)
        self.assertTrue(np.all(np.diff(row) >= 0), "curve inverted tonal order")


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

class TestMetrics(unittest.TestCase):

    def test_every_metric_declares_its_kind(self):
        for name, metric in compute_metrics(dark_frame()).items():
            with self.subTest(metric=name):
                self.assertIn(metric["kind"], ("measured", "estimated"))
                self.assertTrue(metric["label"])
                self.assertTrue(metric["description"])

    def test_known_values(self):
        flat = np.full((40, 40, 3), 128, np.uint8)
        m = compute_metrics(flat)
        self.assertAlmostEqual(m["mean_luminance"]["value"], 128.0, places=1)
        self.assertAlmostEqual(m["rms_contrast"]["value"], 0.0, places=3)
        self.assertAlmostEqual(m["entropy"]["value"], 0.0, places=3)

    def test_clipping_counts_are_exact(self):
        frame = np.zeros((10, 10, 3), np.uint8)
        frame[:5] = 255
        m = compute_metrics(frame)
        self.assertAlmostEqual(m["shadow_clipping"]["value"], 50.0, places=1)
        self.assertAlmostEqual(m["highlight_clipping"]["value"], 50.0, places=1)

    def test_neutral_metrics_have_no_improvement_direction(self):
        a = dark_frame()
        b = np.clip(a.astype(int) * 3, 0, 255).astype(np.uint8)
        result = compare(compute_metrics(a), compute_metrics(b))
        self.assertIsNone(result["mean_luminance"]["improved"])
        self.assertIsNone(result["colorfulness"]["improved"])

    def test_lower_is_better_metrics_score_correctly(self):
        dark = dark_frame()
        lifted = np.clip(dark.astype(int) + 60, 0, 255).astype(np.uint8)
        result = compare(compute_metrics(dark), compute_metrics(lifted))
        self.assertTrue(result["shadow_clipping"]["improved"])


# --------------------------------------------------------------------------
# storage
# --------------------------------------------------------------------------

class TestStorage(unittest.TestCase):

    def test_ttl_expiry(self):
        store = ResultStore(ttl=1, max_items=10)
        rid = store.put({"a": 1}, {"enhanced": b"x"})
        self.assertEqual(store.get(rid)["payload"]["a"], 1)
        time.sleep(1.05)
        with self.assertRaises(ResultExpired):
            store.get(rid)

    def test_capacity_eviction_drops_oldest(self):
        store = ResultStore(ttl=600, max_items=3)
        ids = [store.put({"n": i}, {"enhanced": b"x"}) for i in range(5)]
        self.assertEqual(store.stats()["held"], 3)
        with self.assertRaises(NotFound):
            store.get(ids[0])
        self.assertEqual(store.get(ids[-1])["payload"]["n"], 4)

    def test_sweep_reclaims_expired(self):
        store = ResultStore(ttl=1, max_items=10)
        for _ in range(3):
            store.put({}, {"enhanced": b"x"})
        time.sleep(1.05)
        self.assertEqual(store.sweep(), 3)
        self.assertEqual(store.stats()["held"], 0)

    def test_background_sweeper_starts_and_stops(self):
        store = ResultStore(ttl=1, max_items=10)
        store.start_sweeper(interval=1)
        self.assertTrue(store.stats()["sweeper_running"])
        store.put({}, {"enhanced": b"x"})
        time.sleep(2.3)
        self.assertEqual(store.stats()["held"], 0, "sweeper did not reclaim")
        store.stop_sweeper()
        self.assertFalse(store.stats()["sweeper_running"])

    def test_unknown_image_name_is_rejected(self):
        store = ResultStore()
        rid = store.put({}, {"enhanced": b"x"})
        with self.assertRaises(NotFound):
            store.get_image(rid, "../../etc/passwd")


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------

class TestApi(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.app = create_app(start_sweeper=False)
        cls.app.config["TESTING"] = True

    def setUp(self):
        self.client = self.app.test_client()

    def tearDown(self):
        STORE.clear()

    # -- basics --

    def test_index_serves_the_page(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Lunar Low-Light Lab", response.data)

    def test_health(self):
        body = self.client.get("/api/health").get_json()
        self.assertEqual(body["status"], "ok")
        self.assertIn("zero_dce", body)
        self.assertIn("store", body)

    def test_config_describes_modes_and_limits(self):
        body = self.client.get("/api/config").get_json()
        ids = {m["id"] for m in body["modes"]}
        self.assertEqual(ids, {"naive", "scientific", "clahe", "zerodce"})
        self.assertIn("max_upload_bytes", body["limits"])

    def test_security_headers_present(self):
        response = self.client.get("/")
        for header in HEADERS:
            with self.subTest(header=header):
                self.assertIn(header, response.headers)
        self.assertNotIn("unsafe-inline", response.headers["Content-Security-Policy"])

    def test_json_responses_are_not_cached(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.headers.get("Cache-Control"), "no-store")

    # -- the happy path --

    def test_enhance_returns_a_complete_payload(self):
        response = self.client.post("/api/enhance", data=upload(),
                                    content_type="multipart/form-data")
        self.assertEqual(response.status_code, 201)
        body = response.get_json()
        for key in ("id", "source", "options", "enhancement", "metrics",
                    "comparison", "histogram", "difference", "telemetry",
                    "timings_ms", "images", "engine"):
            with self.subTest(key=key):
                self.assertIn(key, body)
        self.assertEqual(set(body["images"]), {"original", "enhanced", "difference"})

    def test_all_modes_work_over_http(self):
        for mode in ("naive", "scientific", "clahe", "zerodce"):
            with self.subTest(mode=mode):
                response = self.client.post(
                    "/api/enhance", data=upload(mode=mode),
                    content_type="multipart/form-data")
                self.assertEqual(response.status_code, 201)
                self.assertEqual(response.get_json()["options"]["mode"], mode)

    def test_images_are_served_as_png(self):
        body = self.client.post("/api/enhance", data=upload(),
                                content_type="multipart/form-data").get_json()
        for name, url in body["images"].items():
            with self.subTest(image=name):
                response = self.client.get(url)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.mimetype, "image/png")
                self.assertTrue(response.data.startswith(b"\x89PNG"))

    def test_result_is_retrievable_and_identical(self):
        posted = self.client.post("/api/enhance", data=upload(),
                                  content_type="multipart/form-data").get_json()
        fetched = self.client.get("/api/results/" + posted["id"]).get_json()
        self.assertEqual(posted, fetched)

    def test_download_flag_sets_attachment(self):
        body = self.client.post("/api/enhance", data=upload(),
                                content_type="multipart/form-data").get_json()
        response = self.client.get(body["images"]["enhanced"] + "?download=1")
        self.assertIn("attachment", response.headers.get("Content-Disposition", ""))

    def test_delete_removes_the_result(self):
        body = self.client.post("/api/enhance", data=upload(),
                                content_type="multipart/form-data").get_json()
        self.assertTrue(self.client.delete("/api/results/" + body["id"]).get_json()["deleted"])
        self.assertEqual(self.client.get("/api/results/" + body["id"]).status_code, 404)

    def test_sweep_endpoint(self):
        body = self.client.post("/api/maintenance/sweep").get_json()
        self.assertIn("expired_removed", body)
        self.assertIn("store", body)

    # -- the error contract --

    def _assert_error_shape(self, response, status, code):
        self.assertEqual(response.status_code, status)
        body = response.get_json()
        self.assertIn("error", body)
        self.assertEqual(body["error"]["code"], code)
        self.assertTrue(body["error"]["message"])

    def test_missing_file_is_a_validation_error(self):
        self._assert_error_shape(
            self.client.post("/api/enhance", data={}, content_type="multipart/form-data"),
            400, "validation_failed")

    def test_non_image_upload_is_rejected(self):
        self._assert_error_shape(
            self.client.post("/api/enhance",
                             data=upload(b"<html>not an image</html>", "x.png"),
                             content_type="multipart/form-data"),
            415, "unsupported_image")

    def test_bad_mode_is_rejected(self):
        self._assert_error_shape(
            self.client.post("/api/enhance", data=upload(mode="hallucinate"),
                             content_type="multipart/form-data"),
            400, "validation_failed")

    def test_out_of_range_parameter_is_rejected(self):
        self._assert_error_shape(
            self.client.post("/api/enhance", data=upload(gamma="500"),
                             content_type="multipart/form-data"),
            400, "validation_failed")

    def test_unknown_result_is_404_json(self):
        self._assert_error_shape(
            self.client.get("/api/results/does-not-exist"), 404, "not_found")

    def test_unknown_image_name_is_404(self):
        body = self.client.post("/api/enhance", data=upload(),
                                content_type="multipart/form-data").get_json()
        self._assert_error_shape(
            self.client.get("/api/results/" + body["id"] + "/image/secrets"),
            404, "not_found")

    def test_oversized_upload_is_413(self):
        from backend.config import CONFIG
        blob = b"\x00" * (CONFIG.MAX_UPLOAD_BYTES + 2048)
        response = self.client.post("/api/enhance", data=upload(blob, "huge.png"),
                                    content_type="multipart/form-data")
        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.get_json()["error"]["code"], "payload_too_large")

    def test_unknown_api_route_returns_json_not_html(self):
        response = self.client.get("/api/nope")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.mimetype, "application/json")

    # -- behaviour --

    def test_metrics_show_improvement_on_a_dark_frame(self):
        body = self.client.post("/api/enhance", data=upload(mode="scientific"),
                                content_type="multipart/form-data").get_json()
        entropy = body["comparison"]["entropy"]
        self.assertGreater(entropy["after"], entropy["before"])

    def test_estimated_metrics_are_flagged_in_the_payload(self):
        body = self.client.post("/api/enhance", data=upload(),
                                content_type="multipart/form-data").get_json()
        self.assertEqual(body["metrics"]["after"]["noise_sigma"]["kind"], "estimated")
        self.assertEqual(body["metrics"]["after"]["mean_luminance"]["kind"], "measured")

    def test_jpeg_upload_works(self):
        response = self.client.post(
            "/api/enhance", data=upload(encode(dark_frame(), "JPEG"), "frame.jpg"),
            content_type="multipart/form-data")
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.get_json()["source"]["detected_format"], "JPEG")

    def test_expired_result_reports_410(self):
        original_ttl = STORE.ttl
        STORE.ttl = 0
        try:
            body = self.client.post("/api/enhance", data=upload(),
                                    content_type="multipart/form-data").get_json()
            time.sleep(0.01)
            self._assert_error_shape(
                self.client.get("/api/results/" + body["id"]), 410, "result_expired")
        finally:
            STORE.ttl = original_ttl


if __name__ == "__main__":
    unittest.main(verbosity=2)
