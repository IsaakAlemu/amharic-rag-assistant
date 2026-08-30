"""Unit tests for the thread-safe global rate limiter."""

from __future__ import annotations

import unittest
from src.rate_limiter import GlobalRateLimiter


class GlobalRateLimiterTests(unittest.TestCase):
    def test_acquire_under_limit(self):
        limiter = GlobalRateLimiter(max_requests=3, window_seconds=60)
        self.assertTrue(limiter.acquire())
        self.assertTrue(limiter.acquire())
        self.assertTrue(limiter.acquire())
        self.assertEqual(limiter.remaining(), 0)

    def test_acquire_over_limit_fails(self):
        limiter = GlobalRateLimiter(max_requests=2, window_seconds=60)
        self.assertTrue(limiter.acquire())
        self.assertTrue(limiter.acquire())
        self.assertFalse(limiter.acquire())
        self.assertEqual(limiter.remaining(), 0)

    def test_reset(self):
        limiter = GlobalRateLimiter(max_requests=1, window_seconds=60)
        self.assertTrue(limiter.acquire())
        self.assertFalse(limiter.acquire())
        limiter.reset()
        self.assertTrue(limiter.acquire())


if __name__ == "__main__":
    unittest.main()
