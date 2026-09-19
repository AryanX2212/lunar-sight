"""Response hardening for a locally-served app.

The threat model is modest but real: this app renders user-supplied images and
serves them back. The headers below stop a crafted upload (an SVG-ish payload,
an HTML file renamed .png) from executing in the page's origin, and stop the
browser from second-guessing the content type we declare.
"""
from __future__ import annotations

from flask import Flask, Response

# No external origins are needed. Everything is served from the app itself,
# and 'unsafe-inline' is omitted so an injected <script> cannot run.
CONTENT_SECURITY_POLICY = "; ".join([
    "default-src 'self'",
    "script-src 'self'",
    "style-src 'self'",
    "img-src 'self' blob: data:",
    "connect-src 'self'",
    "font-src 'self'",
    "object-src 'none'",
    "base-uri 'none'",
    "form-action 'self'",
    "frame-ancestors 'none'",
])

HEADERS = {
    "Content-Security-Policy": CONTENT_SECURITY_POLICY,
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Resource-Policy": "same-origin",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=(), interest-cohort=()",
}


def register_security(app: Flask) -> None:
    @app.after_request
    def _apply(response: Response) -> Response:
        for key, value in HEADERS.items():
            response.headers.setdefault(key, value)
        # Results are session-scoped and mutable; never let a proxy or the
        # browser cache the analysis JSON.
        if response.mimetype == "application/json":
            response.headers.setdefault("Cache-Control", "no-store")
        return response
