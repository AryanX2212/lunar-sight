#!/usr/bin/env python3
"""Start the Lunar Low-Light Lab on http://127.0.0.1:8000/

    python run.py                # default
    python run.py --port 8080    # different port
    python run.py --debug        # reloader and verbose errors

Bound to 127.0.0.1 by default. The app has no authentication, so do not
expose it on 0.0.0.0 on a shared network without putting something in front.
"""
from __future__ import annotations

import argparse
import logging

from backend.app import create_app
from backend.config import CONFIG
from backend.storage import STORE
from backend.zerodce import WEIGHTS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default=CONFIG.HOST)
    parser.add_argument("--port", type=int, default=CONFIG.PORT)
    parser.add_argument("--debug", action="store_true", default=CONFIG.DEBUG)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("lunar")

    app = create_app()

    status = WEIGHTS.status()
    log.info("Lunar Low-Light Lab  ->  http://%s:%s/", args.host, args.port)
    if status["neural_available"]:
        log.info("Zero-DCE: %s", status["detail"])
    else:
        log.warning("Zero-DCE: %s", status["detail"])
        log.warning("The Zero-DCE mode will use the labelled analytic fallback.")
    log.info("Results expire after %d minutes.", CONFIG.RESULT_TTL_SECONDS // 60)

    try:
        # threaded so a slow Zero-DCE inference does not block the static files
        app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
    finally:
        STORE.stop_sweeper()
        STORE.clear()
        log.info("Stopped. Held results cleared.")


if __name__ == "__main__":
    main()
