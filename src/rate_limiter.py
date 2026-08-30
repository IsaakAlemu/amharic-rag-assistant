"""Thread-safe in-memory rolling-window global rate limiter for public hosted demos."""

from __future__ import annotations

import threading
import time
from collections import deque


class GlobalRateLimiter:
    """
    Limits the total number of requests across all users/sessions
    within a rolling time window.
    """

    def __init__(self, max_requests: int = 60, window_seconds: int = 3600):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._timestamps: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> bool:
        """
        Check if a request is allowed and record its timestamp.
        Returns True if allowed, False if the rate limit is exceeded.
        """
        now = time.time()
        with self._lock:
            # Purge timestamps outside the rolling window
            while self._timestamps and self._timestamps[0] < now - self.window_seconds:
                self._timestamps.popleft()

            if len(self._timestamps) < self.max_requests:
                self._timestamps.append(now)
                return True
            return False

    def remaining(self) -> int:
        """Return the number of requests remaining in the current window."""
        now = time.time()
        with self._lock:
            while self._timestamps and self._timestamps[0] < now - self.window_seconds:
                self._timestamps.popleft()
            return max(0, self.max_requests - len(self._timestamps))

    def reset(self) -> None:
        """Reset the rate limiter counter (useful for testing)."""
        with self._lock:
            self._timestamps.clear()
