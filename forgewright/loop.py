"""The agent loop: plan -> act (tool) -> observe -> reflect, until the goal is met.

Model-agnostic (any Brain), tool-driven (any ToolRegistry), with a permission gate,
an append-only ledger, a context manager, a doom-loop guard, and an optional live
reporter (for the interactive CLI). This is the core the ML-ops skills plug into.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Optional

from forgewright.brain.provider import Brain, ToolCall
from forgewright.context.manager import ContextManager
from forgewright.ledger.ledger import Ledger
from forgewright.permissions import PermissionPolicy
from forgewright.tools.base import ToolRegistry, ToolResult

SYSTEM_PROMPT = """\
You are Forgewright, an autonomous post-training engineer that operates real GPUs
(local or over SSH) to fine-tune, abliterate, quantize, and serving-optimize LLMs.

How you work:
- Inspect the hardware (gpu_inspect) before choosing a strategy; match precision and
  parallelism to the GPU (Blackwell SM120/121 = NVFP4-native).
- Work in small, verified steps. Read/write files and run commands to make progress.
- LONG training/quant runs MUST be launched as DETACHED jobs (launch_job) and polled
  with monitor_job / tail_logs. Never block waiting on a long job; check back on it.
- Recover from failures yourself: OOM -> lower batch / raise grad-accum / shard;
  NaN or reward collapse -> lower LR, add a KL anchor + PPO clip, enable dynamic
  sampling; watch for repetition/format degeneration mid-train; dependency or CUDA
  errors -> diagnose and fix the environment.
- Prefer model-forge skills when they cover the case; otherwise write the code yourself.
- Drive the model-forge engine: quick stages (plan, nvfp4-gate, reports, small evals) via the
  `forge` tool; LONG stages (quantize export, finetune/ablate run) via launch_job
  (command="bash forge <stage> ...", cwd=the model-forge repo) then poll monitor_job / tail_logs.
- NVFP4 quantize flow: `forge quantize plan ...` -> launch_job `forge quantize export ...` ->
  `forge serve <family> <variant>` -> `forge eval <family> <variant>` -> `forge quantize nvfp4-gate ...`;
  publish ONLY after the gate passes, via `forge_publish`. Rehearse with dry_run / `plan` first.
- Gate every stage on evals (capability must not regress; for abliteration, refusal
  must drop AND capability must hold).
- Ask before irreversible actions (publishing weights or datasets). Everything else: act.
- When the goal is met, stop and summarize what you did and where the artifacts live.
"""


@dataclass
class LoopResult:
    done: bool
    steps: int
    final: str


class Agent:
    def __init__(
        self,
        brain: Brain,
        tools: ToolRegistry,
        *,
        permissions: Optional[PermissionPolicy] = None,
        ledger: Optional[Ledger] = None,
        context: Optional[ContextManager] = None,
        max_steps: int = 80,
        system_prompt: str = SYSTEM_PROMPT,
        reporter: Optional[Callable[[str, dict], object]] = None,
    ) -> None:
        self.brain = brain
        self.tools = tools
        self.permissions = permissions or PermissionPolicy()
        self.ledger = ledger
        self.ctx = context or ContextManager(system_prompt=system_prompt)
        self.max_steps = max_steps
        self.reporter = reporter

    def _log(self, kind: str, **data: object) -> None:
        if self.ledger:
            self.ledger.event(kind, **data)

    def _emit(self, kind: str, data: dict) -> None:
        if self.reporter:
            try:
                self.reporter(kind, data)
            except Exception:  # noqa: BLE001 - display must never break the loop
                pass

    def run(self, goal: str) -> LoopResult:
        """Run one user turn to completion (reuses the persistent context across calls)."""
        self.ctx.add_user(goal)
        self._log("goal", goal=goal)
        recent: list[str] = []

        for step in range(1, self.max_steps + 1):
            turn = self.brain.chat(self.ctx.messages(), tools=self.tools.schemas())
            self.ctx.add_assistant(turn)
            self._log(
                "assistant",
                step=step,
                content=turn.content[:2000],
                tool_calls=[tc.name for tc in turn.tool_calls],
                usage=turn.usage,
            )
            self._emit(
                "assistant",
                {"content": turn.content, "tool_calls": [tc.name for tc in turn.tool_calls], "usage": turn.usage},
            )

            if not turn.tool_calls:
                return LoopResult(True, step, turn.content)

            for tc in turn.tool_calls:
                result = self._dispatch(tc, recent)
                self.ctx.add_tool_result(tc, result)
                self._log(
                    "tool",
                    step=step,
                    tool=tc.name,
                    args=tc.arguments,
                    ok=result.ok,
                    output=result.output[:2000],
                )
                self._emit(
                    "tool",
                    {"tool": tc.name, "args": tc.arguments, "ok": result.ok, "output": result.output},
                )
            self.ctx.maybe_compact()

        return LoopResult(False, self.max_steps, "step budget exhausted")

    def _dispatch(self, tc: ToolCall, recent: list[str]) -> ToolResult:
        tool = self.tools.get(tc.name)
        if tool is None:
            return ToolResult(False, f"unknown tool: {tc.name!r}")

        sig = tc.name + json.dumps(tc.arguments, sort_keys=True, default=str)
        recent.append(sig)
        if recent[-3:].count(sig) >= 3:
            return ToolResult(False, "doom-loop detected: identical call repeated 3x — change approach")

        decision = self.permissions.check(tool, tc.arguments)
        if not decision.allowed:
            return ToolResult(False, f"blocked by permission policy: {decision.reason}")

        try:
            return tool.run(**tc.arguments)
        except Exception as e:  # noqa: BLE001 - tools shouldn't crash the loop
            return ToolResult(False, f"tool {tc.name} raised: {e}")
