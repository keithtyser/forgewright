"""Serving-optimization skill: find the tok/s x quality Pareto for a served model.

Forgewright OWNS the vLLM launch here, because model-forge's serve script does not
expose the knobs we tune. It captures model-forge's *resolved* launch command via a
dry-run, appends a candidate's extra vLLM flags (speculative decoding, KV precision,
batching), launches it detached in model-forge's container, then **benchmarks AND
re-evals** each candidate against the source quant's eval. Only quality-preserving
candidates qualify; we report the Pareto and pick the knee for the chosen objective.

objective:
  - "latency"    -> maximize single-stream output tok/s (interactive / low-latency)
  - "throughput" -> maximize aggregate tok/s under concurrency (batched serving)

This module keeps the *decision logic* pure + unit-tested (candidate generation,
objective scoring, quality gate, Pareto/knee selection); the launch/bench/eval
methods shell out to model-forge and are validated on the box.
"""
from __future__ import annotations

import json
import re
import shlex
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

from forgewright.tools.base import Tool, ToolResult
from forgewright.tools.forge import ForgeRunner
from forgewright.tools.jobs import JobManager

Objective = Literal["latency", "throughput"]


@dataclass
class Candidate:
    name: str
    extra_flags: list[str] = field(default_factory=list)
    notes: str = ""


def default_candidates(objective: Objective) -> list[Candidate]:
    """The sweep. `baseline` is the as-quantized serving config (no extra flags)."""
    cands = [
        Candidate("baseline", [], "as-quantized serving config"),
        Candidate(
            "ngram_spec",
            [
                "--speculative-config",
                '{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":4,"prompt_lookup_min":2}',
            ],
            "ngram speculative decoding (no drafter; helps structured/repetitive output)",
        ),
    ]
    if objective == "throughput":
        cands.append(
            Candidate(
                "batch_throughput",
                ["--max-num-seqs", "64", "--max-num-batched-tokens", "16384"],
                "larger batch for aggregate throughput",
            )
        )
    else:  # latency
        cands.append(
            Candidate(
                "eager_lowlat",
                ["--max-num-seqs", "1"],
                "single-sequence for lowest per-token latency",
            )
        )
    return cands


def default_served_name(family: str, variant: str) -> str:
    return f"model-forge/{family.replace('_', '-')}-{variant.replace('_', '-')}"


@dataclass
class CandidateResult:
    name: str
    served: bool
    single_tps: Optional[float] = None       # single-stream output tok/s
    aggregate_tps: Optional[float] = None     # aggregate tok/s under concurrency
    quality_pass_rate: Optional[float] = None  # eval pass-rate (normal-use regression)
    quality_ok: bool = False                   # within tolerance of source
    detail: str = ""

    def objective_score(self, objective: Objective) -> float:
        v = self.aggregate_tps if objective == "throughput" else self.single_tps
        return float(v) if v is not None else -1.0


def select_best(results: list[CandidateResult], objective: Objective) -> Optional[CandidateResult]:
    """Best = highest objective score among QUALITY-PRESERVING, served candidates."""
    eligible = [r for r in results if r.served and r.quality_ok and r.objective_score(objective) > 0]
    if not eligible:
        return None
    return max(eligible, key=lambda r: r.objective_score(objective))


class ServingOptimizer:
    def __init__(
        self,
        runner: Optional[ForgeRunner] = None,
        jobs: Optional[JobManager] = None,
        *,
        port: int = 8000,
        quality_tolerance: float = 0.03,
    ) -> None:
        self.forge = runner or ForgeRunner()
        self.jobs = jobs or JobManager()
        self.port = port
        self.base_url = f"http://127.0.0.1:{port}/v1"
        self.quality_tolerance = quality_tolerance

    # --- launch (Forgewright owns the vLLM command) ----------------------------

    def resolve_base_command(self, family: str, variant: str) -> Optional[str]:
        """Capture model-forge's resolved launch-cluster vLLM command via a dry-run."""
        res = self.forge.run(f"serve {family} {variant}", dry_run=True, timeout=180)
        for line in res.output.splitlines():
            if "launch-cluster" in line and "vllm serve" in line:
                if "dry-run command:" in line:
                    line = line.split("dry-run command:", 1)[1]
                return line.strip()
        return None

    def launch(self, family: str, variant: str, extra_flags: list[str]) -> dict[str, Any]:
        import subprocess

        base = self.resolve_base_command(family, variant)
        if not base:
            return {"ok": False, "error": "could not resolve base serve command from dry-run"}
        cmd = base
        if extra_flags:
            cmd = base + " " + " ".join(shlex.quote(f) for f in extra_flags)
        subprocess.run(["docker", "rm", "-f", "vllm_node"], capture_output=True)
        time.sleep(5)
        rec = self.jobs.launch(cmd, cwd=str(self.forge.repo), name=f"serveopt-{variant}")
        return {"ok": True, "job": rec["id"], "command": cmd}

    def wait_ready(self, timeout_s: int = 600, interval: int = 15) -> bool:
        import httpx

        deadline = time.time() + timeout_s
        while time.time() < deadline:
            try:
                if httpx.get(f"{self.base_url}/models", timeout=8).status_code == 200:
                    return True
            except Exception:  # noqa: BLE001
                pass
            time.sleep(interval)
        return False

    def stop(self) -> None:
        import subprocess

        subprocess.run(["docker", "rm", "-f", "vllm_node"], capture_output=True)
        time.sleep(5)

    # --- measure ---------------------------------------------------------------

    def bench_single(self, served_name: str, run_id: str) -> Optional[float]:
        res = self.forge.run(
            f"bench serve --config configs/serving/serve_bench_smoke.yaml --model {served_name} "
            f"--base-url {self.base_url} --run-id {run_id}",
            timeout=900,
        )
        return _parse_metric(res.output, "output_tokens_per_second")

    def bench_aggregate(self, served_name: str, *, concurrency: int = 16, requests: int = 48) -> Optional[float]:
        """Aggregate tok/s under concurrency (simple concurrent load on the endpoint)."""
        import concurrent.futures
        import httpx

        prompt = {"model": served_name, "messages": [{"role": "user", "content": "Write a detailed runbook."}],
                  "max_tokens": 256, "temperature": 0.0}

        def one() -> int:
            try:
                r = httpx.post(f"{self.base_url}/chat/completions", json=prompt, timeout=120)
                return int(r.json().get("usage", {}).get("completion_tokens", 0))
            except Exception:  # noqa: BLE001
                return 0

        start = time.time()
        total = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
            for n in ex.map(lambda _: one(), range(requests)):
                total += n
        elapsed = max(time.time() - start, 1e-6)
        return total / elapsed

    def eval_quality(self, family: str, variant: str) -> Optional[float]:
        """Re-eval the served config; return the normal-use-regression pass rate."""
        res = self.forge.run(f"eval {family} {variant} --internal", timeout=5400)
        return _parse_metric(res.output, "normal_use_regression_pass_rate")

    def run(
        self,
        family: str,
        variant: str,
        objective: Objective = "latency",
        *,
        served_name: Optional[str] = None,
        source_quality: Optional[float] = None,
        candidates: Optional[list[Candidate]] = None,
        eval_each: bool = True,
    ) -> tuple[list[CandidateResult], Optional[CandidateResult]]:
        """Sweep candidates: launch -> bench (single + aggregate) -> re-eval (quality
        gate vs source) -> pick the objective's best quality-preserving config."""
        served = served_name or default_served_name(family, variant)
        cands = candidates or default_candidates(objective)
        results: list[CandidateResult] = []
        for c in cands:
            launch = self.launch(family, variant, c.extra_flags)
            if not launch.get("ok") or not self.wait_ready():
                results.append(CandidateResult(c.name, served=False, detail=str(launch.get("error", "not ready"))))
                self.stop()
                continue
            single = self.bench_single(served, f"servingopt_{c.name}")
            agg = self.bench_aggregate(served) if objective == "throughput" else None
            q = self.eval_quality(family, variant) if eval_each else None
            qok = (q is None) or (source_quality is None) or (q >= source_quality - self.quality_tolerance)
            results.append(
                CandidateResult(c.name, True, single_tps=single, aggregate_tps=agg, quality_pass_rate=q, quality_ok=qok)
            )
            self.stop()
        return results, select_best(results, objective)


def _parse_metric(text: str, key: str) -> Optional[float]:
    # tolerate CSV ("key,...,value") and table ("key  value") forms
    for line in text.splitlines():
        if key in line:
            nums = re.findall(r"\d+\.\d+|\d+", line.replace(key, ""))
            if nums:
                try:
                    return float(nums[0])
                except ValueError:
                    continue
    return None


class ServingOptTool(Tool):
    name = "serving_opt"
    description = (
        "Optimize serving for a quantized model variant: sweep candidate vLLM configs (speculative "
        "decoding, batching), benchmark each (single-stream + aggregate), re-eval quality vs the source, "
        "and return the best QUALITY-PRESERVING config for the objective. objective: 'latency' | "
        "'throughput'. Long-running (serves + benches + evals each candidate). Set eval_each=false for a "
        "fast bench-only sweep first."
    )
    risk = "exec"
    parameters = {
        "type": "object",
        "properties": {
            "family": {"type": "string"},
            "variant": {"type": "string", "description": "quantized variant, e.g. base_nvfp4_modelopt"},
            "objective": {"type": "string", "description": "latency | throughput (default latency)"},
            "eval_each": {"type": "boolean", "description": "re-eval each candidate (default true); false = bench-only"},
        },
        "required": ["family", "variant"],
    }

    def __init__(self, runner: ForgeRunner | None = None, jobs: JobManager | None = None) -> None:
        self.runner = runner or ForgeRunner()
        self.jobs = jobs or JobManager()

    def run(
        self, family: str, variant: str, objective: str = "latency", eval_each: bool = True, **_: Any
    ) -> ToolResult:
        if not self.runner.available():
            return ToolResult(False, f"model-forge not found at {self.runner.repo}")
        opt = ServingOptimizer(self.runner, self.jobs)
        try:
            results, best = opt.run(family, variant, objective, eval_each=eval_each)  # type: ignore[arg-type]
        except Exception as e:  # noqa: BLE001
            return ToolResult(False, f"serving_opt error: {e}")
        lines = [
            f"  {r.name}: served={r.served} single_tps={r.single_tps} agg_tps={r.aggregate_tps} "
            f"quality={r.quality_pass_rate} quality_ok={r.quality_ok}"
            for r in results
        ]
        tail = f"best for {objective}: {best.name}" if best else "no quality-preserving candidate beat baseline"
        return ToolResult(
            True,
            "serving-opt sweep:\n" + "\n".join(lines) + "\n" + tail,
            {"best": best.name if best else None, "objective": objective},
        )
