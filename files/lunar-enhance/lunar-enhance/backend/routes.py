"""HTTP surface.

Contract
--------
POST /api/enhance            multipart form: image=<file>, plus options
GET  /api/results/<id>       the stored analysis payload
GET  /api/results/<id>/image/<name>   original | enhanced | difference (PNG)
DELETE /api/results/<id>     drop a result immediately
GET  /api/health             liveness, engine status, store stats
GET  /api/config             limits and defaults, so the UI is not hard-coded
POST /api/maintenance/sweep  force expiry collection

Every error leaves as {"error": {"code", "message", "details"}}.
"""
from __future__ import annotations

from flask import Blueprint, jsonify, request, send_file
import io

from . import __version__
from .config import CONFIG
from .errors import ApiError, ValidationError
from .pipeline import process
from .storage import STORE
from .validation import VALID_MODES, decode_image, validate_options
from .zerodce import WEIGHTS

api = Blueprint("api", __name__, url_prefix="/api")

IMAGE_NAMES = ("original", "enhanced", "difference")


@api.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "version": __version__,
        "zero_dce": WEIGHTS.status(),
        "store": STORE.stats(),
    })


@api.get("/config")
def config():
    return jsonify({
        "modes": [
            {
                "id": "scientific",
                "name": "Scientific",
                "summary": "Luminance-only tone mapping with adaptive gamma and "
                           "shadow-weighted denoise. Colour relationships are preserved.",
            },
            {
                "id": "clahe",
                "name": "Adaptive equalisation",
                "summary": "CLAHE on the L channel. Strongest at local texture "
                           "such as crater rims, at the cost of global brightness comparability.",
            },
            {
                "id": "zerodce",
                "name": "Zero-DCE",
                "summary": "Learned per-pixel tone curves. Falls back to a "
                           "labelled analytic approximation when no weights are installed.",
            },
            {
                "id": "naive",
                "name": "Fixed gamma",
                "summary": "A plain gamma lift, included as the baseline the "
                           "other modes are measured against.",
            },
        ],
        "limits": {
            "max_upload_bytes": CONFIG.MAX_UPLOAD_BYTES,
            "max_pixels": CONFIG.MAX_PIXELS,
            "max_edge": CONFIG.MAX_EDGE,
            "min_edge": CONFIG.MIN_EDGE,
            "formats": sorted(CONFIG.ALLOWED_FORMATS),
        },
        "defaults": {
            "mode": CONFIG.DEFAULT_MODE,
            "gamma": CONFIG.GAMMA_DEFAULT,
            "clahe_clip": CONFIG.CLAHE_CLIP_DEFAULT,
            "clahe_grid": CONFIG.CLAHE_GRID_DEFAULT,
        },
        "result_ttl_seconds": CONFIG.RESULT_TTL_SECONDS,
    })


@api.post("/enhance")
def enhance_endpoint():
    if "image" not in request.files:
        raise ValidationError(
            "No image was attached. Send a multipart form with an 'image' field.",
            details={"supported_modes": list(VALID_MODES)},
        )
    upload = request.files["image"]
    if not upload.filename:
        raise ValidationError("The attached file has no name and may not have uploaded.")

    raw = upload.read()
    rgb, source_meta = decode_image(raw, filename=upload.filename)
    options = validate_options(request.form.to_dict())
    return jsonify(process(rgb, options, source_meta)), 201


@api.get("/results/<result_id>")
def get_result(result_id: str):
    return jsonify(STORE.get(result_id)["payload"])


@api.get("/results/<result_id>/image/<name>")
def get_result_image(result_id: str, name: str):
    data = STORE.get_image(result_id, name)
    response = send_file(
        io.BytesIO(data),
        mimetype="image/png",
        as_attachment=request.args.get("download") == "1",
        download_name=f"{name}-{result_id[:8]}.png",
    )
    response.headers["Cache-Control"] = "private, max-age=300"
    return response


@api.delete("/results/<result_id>")
def delete_result(result_id: str):
    return jsonify({"deleted": STORE.delete(result_id), "id": result_id})


@api.post("/maintenance/sweep")
def sweep():
    return jsonify({"expired_removed": STORE.sweep(), "store": STORE.stats()})
