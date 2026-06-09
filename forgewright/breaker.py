"""Velocity-based circuit breaker - the safe re-introduction of a budget on an unbounded run.

The old step-counter guardrails were removed so the swarm can "run forever until the goal is
met". This is the principled replacement: it does NOT cap total work - it trips only when the
agent stops producing VALUE per unit of cost (no progress over a cost window). So a long, healthy
run never trips, but a silent spin (every tool call failing/repeating, tokens burning, nothing
advancing) is caught instead of running up an unbounded bill.

Cost is measured along three independent dimensions; any one with a positive limit can trip:
  - idle STEPS    (loop iterations since the last progress signal),
  - idle TOKENS   (model tokens spent since the last progress signal),
  - idle SECONDS  (wall-clock since the last progress signal).
A limit of 0 disables that dimension. Defaults come from the environment so operators can tune or
fully disable the breaker (FORGEWRIGHT_BREAKER_IDLE_STEPS=0) without a code change.
"""
from __future__ import annotations

import os
import time
from typing import Optional


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return max(0, int(raw))
    except ValueError:
        return default


class CircuitBreaker:
    """Tracks cost-since-last-progress and trips when an idle window is exceeded.

    Usage: call `record_progress()` whenever the run advances (a novel successful tool result,
    a produced artifact, a passed gate); call `record_step(tokens)` once per loop iteration with
    the tokens that step spent. `tripped()` returns a human-readable reason or None.
    """

    def __init__(
        self,
        *,
        max_idle_steps: Optional[int] = None,
        max_idle_tokens: Optional[int] = None,
        max_idle_seconds: Optional[int] = None,
    ) -> None:
        # Defaults: a generous step window (catches a genuine dead spin without capping real work),
        # token/time windows off unless the operator opts in. All overridable via env.
        self.max_idle_steps = _env_int("FORGEWRIGHT_BREAKER_IDLE_STEPS", 30) if max_idle_steps is None else max_idle_steps
        self.max_idle_tokens = _env_int("FORGEWRIGHT_BREAKER_IDLE_TOKENS", 0) if max_idle_tokens is None else max_idle_tokens
        self.max_idle_seconds = _env_int("FORGEWRIGHT_BREAKER_IDLE_SECONDS", 0) if max_idle_seconds is None else max_idle_seconds
        self._idle_steps = 0
        self._idle_tokens = 0
        self._last_progress = time.time()
        # cumulative, for reporting
        self.total_steps = 0
        self.total_tokens = 0

    @property
    def enabled(self) -> bool:
        return self.max_idle_steps > 0 or self.max_idle_tokens > 0 or self.max_idle_seconds > 0

    def record_progress(self) -> None:
        """The run advanced - reset every idle window."""
        self._idle_steps = 0
        self._idle_tokens = 0
        self._last_progress = time.time()

    def record_step(self, tokens: int = 0) -> None:
        """One loop iteration cost `tokens` and produced no progress (yet)."""
        self.total_steps += 1
        self.total_tokens += max(0, tokens)
        self._idle_steps += 1
        self._idle_tokens += max(0, tokens)

    def tripped(self) -> Optional[str]:
        """A reason string if an idle window is exceeded, else None."""
        if self.max_idle_steps and self._idle_steps >= self.max_idle_steps:
            return (f"circuit breaker: {self._idle_steps} steps with no progress "
                    f"(no new artifact/gate/tool result). Stopping to avoid an unbounded spin.")
        if self.max_idle_tokens and self._idle_tokens >= self.max_idle_tokens:
            return (f"circuit breaker: {self._idle_tokens} tokens spent with no progress. "
                    f"Stopping to avoid an unbounded spin.")
        if self.max_idle_seconds:
            idle = time.time() - self._last_progress
            if idle >= self.max_idle_seconds:
                return (f"circuit breaker: {int(idle)}s with no progress. "
                        f"Stopping to avoid an unbounded spin.")
        return None

    def state(self) -> dict:
        return {"idle_steps": self._idle_steps, "idle_tokens": self._idle_tokens,
                "total_steps": self.total_steps, "total_tokens": self.total_tokens}
