"""In-memory result store with TTL, capacity eviction and a background sweeper.

Results hold decoded PNG bytes, which are large, so nothing is kept
indefinitely. Three independent mechanisms bound memory:

  1. TTL      - a result older than RESULT_TTL_SECONDS is unreadable and swept.
  2. Capacity - beyond MAX_RESULTS, the oldest entries are evicted immediately.
  3. Sweeper  - a daemon thread reclaims expired entries even when no requests
                arrive, so an idle server does not sit on stale image data.

The store is process-local and deliberately not persistent: this is a local
analysis tool, and uploaded imagery should not outlive the session.
"""
from __future__ import annotations

import secrets
import threading
import time
from collections import OrderedDict
from typing import Any

from .config import CONFIG
from .errors import NotFound, ResultExpired


class ResultStore:
    def __init__(self, ttl: int | None = None, max_items: int | None = None):
        self.ttl = ttl if ttl is not None else CONFIG.RESULT_TTL_SECONDS
        self.max_items = max_items if max_items is not None else CONFIG.MAX_RESULTS
        self._items: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._lock = threading.RLock()
        self._sweeper: threading.Thread | None = None
        self._stop = threading.Event()
        self.evicted_count = 0
        self.expired_count = 0

    # -- writing ---------------------------------------------------------

    def put(self, payload: dict[str, Any], images: dict[str, bytes]) -> str:
        result_id = secrets.token_urlsafe(16)
        now = time.time()
        with self._lock:
            self._items[result_id] = {
                "id": result_id,
                "created_at": now,
                "expires_at": now + self.ttl,
                "payload": payload,
                "images": images,
                "bytes": sum(len(b) for b in images.values()),
            }
            self._items.move_to_end(result_id)
            while len(self._items) > self.max_items:
                self._items.popitem(last=False)
                self.evicted_count += 1
        return result_id

    def set_payload(self, result_id: str, payload: dict[str, Any]) -> None:
        """Replace a stored payload without running the expiry check.

        The pipeline writes the payload, then rewrites it with the public URLs
        once the id exists. That second write is part of the same operation and
        must not be refused because the entry is already past its TTL.
        """
        with self._lock:
            entry = self._items.get(result_id)
            if entry is not None:
                entry["payload"] = payload

    # -- reading ---------------------------------------------------------

    def get(self, result_id: str) -> dict[str, Any]:
        with self._lock:
            entry = self._items.get(result_id)
            if entry is None:
                raise NotFound(
                    "That result is not in this session. Results are held in "
                    "memory only and are cleared when the server restarts."
                )
            if entry["expires_at"] <= time.time():
                del self._items[result_id]
                self.expired_count += 1
                raise ResultExpired(
                    f"That result expired. Results are kept for "
                    f"{self.ttl // 60} minutes. Run the enhancement again."
                )
            return entry

    def get_image(self, result_id: str, name: str) -> bytes:
        entry = self.get(result_id)
        image = entry["images"].get(name)
        if image is None:
            raise NotFound(
                f"'{name}' is not an image in this result.",
                details={"available": sorted(entry["images"].keys())},
            )
        return image

    # -- maintenance -----------------------------------------------------

    def delete(self, result_id: str) -> bool:
        with self._lock:
            return self._items.pop(result_id, None) is not None

    def sweep(self) -> int:
        now = time.time()
        with self._lock:
            dead = [k for k, v in self._items.items() if v["expires_at"] <= now]
            for k in dead:
                del self._items[k]
            self.expired_count += len(dead)
        return len(dead)

    def clear(self) -> int:
        with self._lock:
            count = len(self._items)
            self._items.clear()
        return count

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "held": len(self._items),
                "capacity": self.max_items,
                "ttl_seconds": self.ttl,
                "bytes_held": sum(v["bytes"] for v in self._items.values()),
                "evicted_for_capacity": self.evicted_count,
                "expired": self.expired_count,
                "sweeper_running": bool(self._sweeper and self._sweeper.is_alive()),
            }

    # -- background sweeper ----------------------------------------------

    def start_sweeper(self, interval: int | None = None) -> None:
        if self._sweeper and self._sweeper.is_alive():
            return
        seconds = interval or CONFIG.SWEEP_INTERVAL_SECONDS
        self._stop.clear()

        def loop() -> None:
            # wait() rather than sleep() so shutdown is immediate, not delayed
            # by up to a full interval.
            while not self._stop.wait(seconds):
                try:
                    self.sweep()
                except Exception:
                    pass  # a sweep failure must never kill the thread

        self._sweeper = threading.Thread(
            target=loop, name="result-sweeper", daemon=True
        )
        self._sweeper.start()

    def stop_sweeper(self) -> None:
        self._stop.set()
        if self._sweeper:
            self._sweeper.join(timeout=2.0)
            self._sweeper = None


STORE = ResultStore()
