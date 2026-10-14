"""Evaluation strategies over several providers, with a hard per-provider timeout.

* single: only the first provider.
* sequential: providers in order; stop as soon as one returns an ok answer whose confidence meets
  `confident_at` (or a deterministic ok answer). Cheaper: later providers run only when needed.
* parallel: all providers concurrently. Use when every opinion is wanted for disagreement detection.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Literal

from action_sentinel.contracts import DecisionRequest, ProviderError, ProviderResult

from .base import DecisionProvider

Strategy = Literal["single", "sequential", "parallel"]


def _timeout_result(p: DecisionProvider, request: DecisionRequest, timeout_s: float) -> ProviderResult:
    return ProviderResult(
        request_id=request.request_id, provider_name=p.name, provider_version=p.version, provider_kind=p.kind,
        status="timeout", latency_ms=round(timeout_s * 1000, 3), is_test_output=p.kind == "test_double",
        error=ProviderError(code="TIMEOUT", message="no result within %.2f s (enforced by Sentinel)" % timeout_s),
        calibration_status="not_applicable" if p.kind == "deterministic" else "unknown")


class ProviderRunner:
    def __init__(self, timeout_s: float = 30.0, confident_at: float = 0.85, max_workers: int = 8):
        self.timeout_s = timeout_s
        self.confident_at = confident_at
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="provider")

    def _call(self, p: DecisionProvider, request: DecisionRequest) -> ProviderResult:
        fut = self._pool.submit(p.decide, request)
        try:
            return fut.result(timeout=self.timeout_s)
        except FutureTimeout:
            # The worker thread cannot be killed; its eventual result is discarded.
            return _timeout_result(p, request, self.timeout_s)

    def run(self, providers: list[DecisionProvider], request: DecisionRequest,
            strategy: Strategy = "parallel") -> list[ProviderResult]:
        if not providers:
            return []
        if strategy == "single":
            return [self._call(providers[0], request)]
        if strategy == "sequential":
            results = []
            for p in providers:
                r = self._call(p, request)
                results.append(r)
                if r.succeeded and (p.kind == "deterministic" or (r.confidence or 0.0) >= self.confident_at):
                    break
            return results
        started = time.monotonic()
        futures = [(p, self._pool.submit(p.decide, request)) for p in providers]
        results = []
        for p, fut in futures:
            remaining = max(0.0, self.timeout_s - (time.monotonic() - started))
            try:
                results.append(fut.result(timeout=remaining))
            except FutureTimeout:
                results.append(_timeout_result(p, request, self.timeout_s))
        return results

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
