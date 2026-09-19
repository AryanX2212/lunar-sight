"""A single error contract for the whole API.

Any failure the client can act on is raised as an ApiError. The handler in
app.py turns it into {"error": {...}} with a stable machine-readable code, so
the frontend never has to parse prose to decide what to do.
"""
from __future__ import annotations

from typing import Any


class ApiError(Exception):
    """Base class for all client-visible failures."""

    status = 400
    code = "bad_request"

    def __init__(self, message: str, *, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.details:
            payload["details"] = self.details
        return {"error": payload}


class ValidationError(ApiError):
    status = 400
    code = "validation_failed"


class UnsupportedImage(ApiError):
    status = 415
    code = "unsupported_image"


class PayloadTooLarge(ApiError):
    status = 413
    code = "payload_too_large"


class NotFound(ApiError):
    status = 404
    code = "not_found"


class ResultExpired(ApiError):
    status = 410
    code = "result_expired"


class ProcessingFailed(ApiError):
    status = 500
    code = "processing_failed"
