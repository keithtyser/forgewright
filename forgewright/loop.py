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
- Drive the model-forge engine with the `forge` tool — it ALREADY knows the repo path, so call
  it directly (args like "eval qwen35_9b base --internal" or "quantize plan --config ..."). NEVER
  use bash to locate, cd into, or run ./forge yourself. Quick stages (plan, eval, gate, reports)
  go through the `forge` tool; LONG stages (quantize export, finetune/ablate run) go through
  launch_job (command="bash forge <stage> ...", cwd=the model-forge repo) then poll
  monitor_job / tail_logs.
- NVFP4 quantize flow: `forge quantize plan` -> launch_job `forge quantize export` -> ALWAYS
  benchmark the BASE bf16 variant's tok/s FIRST, then the quantized variant's tok/s (serve each,
  then `forge bench serve ...`), and report the quantized-vs-bf16 speedup -> `forge eval` the
  quantized variant -> `forge quantize nvfp4-gate` (it compares quantized vs base for speedup +
  quality); publish ONLY after the gate passes, via `forge_publish`. Never claim a speedup without
  the measured bf16 baseline. Rehearse with dry_run / `plan` first.
- First-time quant gotchas: use scaffold_quant_config for a family lacking a config; the gate needs
  `forge quantize export --write-plan` (the export-plan artifact), a BASE eval too (for card/behavior),
  and a speedup-gated config (gates.nvfp4.min_output_speedup; no static tok/s floor); forge_publish
  handles the HF cache/Xet fix; lift any family-config promotion.blocked_actions:[hf_upload] once the
  gate passes. Fix bad source snapshots first: synthesize a missing generation_config.json from
  config.json, and rename model.safetensors-* shards to model-* (and update the safetensors index).
- Fine-tune flow (the core capability): pick the mode for the goal — UPLIFT (broad reasoning/style via
  teacher-distillation LoRA SFT) or TASK (a verifiable target metric via GRPO/RLVR). For uplift, use
  scaffold_finetune_config <family> --source <hf_model> --data-path <distill.jsonl> (it bakes the scars:
  assistant_only_loss/train_on_responses_only, conservative LR, strict <think> + holdout-overlap hygiene),
  then `forge finetune --config <cfg> plan` and `... prepare --overwrite`. Training runs INSIDE the
  model-forge-posttrain-tf5 container (the host .venv is torch+cpu; its run.sh uses systemd-run which fails
  over SSH) — launch_job the command from build_container_train_command(<name>) (DETACHED; poll). The LoRA
  adapter lands in the config's model.output_dir. Eval-gate the adapter vs BASE: for a verifiable task use the
  self-contained held-out gate (skills.eval_gate: write_eval_gate + build_container/eval_gate command → PASS/
  REGRESSION on a held-out {prompt,answer} set, scored with the training reward); for uplift, serve the adapter
  and forge eval --internal vs base (capability_preservation_challenge must not regress). Report the delta. Watch the loss/grad
  logs for repetition/format degeneration; if it collapses, lower LR and resume from the last good checkpoint.
  TASK mode uses Forgewright's GRPO trainer (KL-anchor + PPO-clip + dynamic sampling), NOT a plain SFT loop.
  Iterate recipes on a SMALL model first (Qwen3.5-0.8B).
- Abliterate flow (remove refusals, keep capability): scaffold_abliterate_config <family> --source <model>
  (contrastive refusal-direction projection; scar defaults: mid-layer projection only, norm-preserve, leave
  embeddings/lm_head/MoE-experts untouched, conservative strength). Eval the SOURCE first (BEFORE refusal +
  capability), then `forge ablate --config <cfg> plan`, launch_job `... collect` then `... export --execute`
  (edits weights in place -> standalone model). Eval the abliterated model AFTER and apply the DUAL gate
  (skills.abliterate.abliterate_gate via read_abliterate_metrics on each scores.csv): refusal_rate_harmful
  must DROP and capability_preservation_challenge must HOLD and benign_refusal_rate must not blow up. If
  capability regresses, lower `strength` / raise layer_skip_first and re-run.
- Serving-opt flow: use the `serving_opt` tool on a quantized variant. Pick the objective the
  USER cares about — `latency` (single-stream tok/s, interactive) or `throughput` (aggregate tok/s
  under concurrency, batch serving). It sweeps env-based candidates through model-forge's serve
  (VLLM_SPECULATIVE_CONFIG / VLLM_EXTRA_ARGS), benches each, re-evals quality vs the source quant,
  and returns the best QUALITY-PRESERVING config. Pass source_quality = the source quant's
  capability_preservation_challenge pass rate so the gate is real. Proven levers on Blackwell:
  ngram speculative decoding (drafter-free, ~+10% latency on structured output, LOSSLESS) and wide
  batching (--max-num-seqs/--max-num-batched-tokens, ~6x aggregate throughput, quality preserved).
  EAGLE needs a TRAINED drafter (build one via finetune, then serving_opt's eagle_candidate). Run a
  fast eval_each=false bench sweep first, then eval-gate the winner.
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
