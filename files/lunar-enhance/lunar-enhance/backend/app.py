"""Application factory and error handling."""
from __future__ import annotations

import logging
import traceback

from flask import Flask, jsonify, render_template
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

from . import __version__
from .config import CONFIG
from .errors import ApiError, NotFound, PayloadTooLarge
from .routes import api
from .security import register_security
from .storage import STORE

log = logging.getLogger("lunar")


def create_app(start_sweeper: bool = True) -> Flask:
    app = Flask(
        __name__,
        template_folder="../frontend/templates",
        static_folder="../frontend/static",
        static_url_path="/static",
    )
    app.config["MAX_CONTENT_LENGTH"] = CONFIG.MAX_UPLOAD_BYTES
    app.config["JSON_SORT_KEYS"] = False
    app.config["VERSION"] = __version__

    app.register_blueprint(api)
    register_security(app)

    @app.get("/")
    def index():
        return render_template("index.html", version=__version__)

    # ---- error handling: one JSON shape for /api, HTML only for the page ----

    def wants_json() -> bool:
        from flask import request
        return request.path.startswith("/api")

    @app.errorhandler(ApiError)
    def handle_api_error(exc: ApiError):
        return jsonify(exc.to_dict()), exc.status

    @app.errorhandler(RequestEntityTooLarge)
    def handle_too_large(_exc):
        # Raised by Werkzeug before our own validation sees the body.
        err = PayloadTooLarge(
            f"That upload exceeds the {CONFIG.MAX_UPLOAD_BYTES / 1e6:.0f} MB limit."
        )
        return jsonify(err.to_dict()), err.status

    @app.errorhandler(HTTPException)
    def handle_http(exc: HTTPException):
        if not wants_json():
            return exc
        return jsonify({
            "error": {
                "code": exc.name.lower().replace(" ", "_"),
                "message": exc.description or exc.name,
            }
        }), exc.code or 500

    @app.errorhandler(Exception)
    def handle_unexpected(exc: Exception):
        log.error("Unhandled error: %s\n%s", exc, traceback.format_exc())
        body = {
            "error": {
                "code": "internal_error",
                "message": "The server hit an unexpected error while handling that request.",
            }
        }
        # Only leak internals when explicitly running in debug.
        if CONFIG.DEBUG:
            body["error"]["details"] = {"exception": repr(exc)}
        return jsonify(body), 500

    if start_sweeper:
        STORE.start_sweeper()

    return app
