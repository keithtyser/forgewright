"""Evaluator specialist — the swarm's gate authority. Produces EvalArtifacts.

For a verifiable adapter it runs the self-contained held-out gate (`skills/eval_gate.py`):
generate base-vs-adapter on a held-out {prompt,answer} set, score with the training reward,
and emit a PASS/REGRESSION verdict — using the robust loader that applies the adapter on
unified-arch models and FAILS LOUD if it doesn't. The EvalArtifact's gate is the decision
the Director keys global gating on.
"""
from __future__ import annotations

import json
import math
import shlex
import time
import uuid
from pathlib import Path
from typing import Optional, Sequence

from forgewright.agents.base import Specialist
from forgewright.contracts import Artifact, EvalArtifact, Gate
from forgewright.skills.eval_gate import build_eval_gate_command, write_eval_gate
from forgewright.tools.base import ToolRegistry
from forgewright.tools.forge import ForgeRunner, ForgeTool
from forgewright.tools.jobs import JobManager, LaunchJobTool, MonitorJobTool, TailLogsTool

_PROMPT = """\
You are the Evaluator specialist in a post-training swarm — the gate authority. Given an
adapter (or model), measure it against the base on a held-out set and decide PASS or
REGRESSION. Always apply the adapter with the robust loader and fail loud if it does not
actually apply (never report a false PASS). You do not train or publish.
"""


class Evaluator(Specialist):
    role = "Evaluator"
    accepts = ("adapter", "model", "served_endpoint")
    produces = "eval"
    description = "Held-out verifiable gate: AdapterArtifact -> EvalArtifact (PASS/REGRESSION)."

    def __init__(self, *, runner: Optional[ForgeRunner] = None, jobs: Optional[JobManager] = None,
                 host: Optional[str] = None, tolerance: float = 0.0, **kw) -> None:
        super().__init__(**kw)
        self.forge = runner or ForgeRunner()
        self.jobs = jobs or JobManager()
        self.host = host
        self.tolerance = tolerance

    def system_prompt(self) -> str:
        return _PROMPT

    def tools(self) -> ToolRegistry:
        return ToolRegistry([
            ForgeTool(self.forge), LaunchJobTool(self.jobs),
            MonitorJobTool(self.jobs), TailLogsTool(self.jobs),
        ])

    def run(self, inputs: Sequence[Artifact], goal: str = "", *, holdout: Optional[str] = None,
            baseline: Optional[dict] = None) -> Artifact:
        self.validate_inputs(inputs)
        target = inputs[0]
        if target.kind == "served_endpoint":
            # final gate after model-optimization: held-out eval of the model the endpoint serves
            model = next((c for c in (self.registry.get(p) for p in target.parents)
                          if c is not None and c.kind == "model"), None)
            if model is None:
                raise ValueError("Evaluator: served_endpoint has no parent model to evaluate")
            from forgewright.skills.serving_opt import ServingOptimizer as Engine
            engine = Engine(self.forge, self.jobs)
            engine.base_url = target.uri
            engine.expected_model = target.meta.get("served_model_name")
            metrics = self._read_internal(engine, model) if engine.wait_ready(timeout_s=15, interval=1) else {}
            return self._model_eval_artifact(target, metrics,
                baseline or {"capability": target.meta.get("quality")}, served=bool(metrics))
        if target.kind == "model":
            # quantized / abliterated / merged model: gate via model-forge internal eval
            return self._eval_model_internal(target, baseline=baseline)
        base = target.meta.get("base")
        adapter = target.uri
        holdout = holdout or target.meta.get("holdout")
        if not base or not holdout:
            raise ValueError("Evaluator needs base (adapter.meta['base']) and a holdout {prompt,answer} jsonl")
        name = f"eval_{uuid.uuid4().hex}"

        self._emit("assistant", content=f"eval {target.id} (base {base}) on holdout {holdout}")
        write_eval_gate(self.forge.repo, name, base=base, adapter=adapter,
                        holdout=holdout, tolerance=self.tolerance, overwrite=True)
        from forgewright.skills.execution import container_options
        cmd = build_eval_gate_command(name, **container_options(self.forge.repo))
        rec = self.jobs.launch(cmd, host=self.host, cwd=str(self.forge.repo), name=f"eval-{name}")
        self._emit("tool", tool="launch_job", ok=True, output=f"job {rec['id']} eval-gate {name}")
        final = self.jobs.wait(rec["id"])
        result: dict = self._read_result(name) if final and final.get("exit_code") == 0 else {
            "passed": False, "verdict": "FAIL: evaluation job did not exit successfully"}
        from forgewright.skills.eval_gate import gate_pass
        passed = (result.get("passed") is True and isinstance(result.get("n"), int)
                  and result["n"] >= 32 and gate_pass(result.get("base_score"),
                  result.get("candidate_score"), tolerance=self.tolerance))
        if not passed:
            result.update(passed=False, verdict="FAIL: evaluation failed or lacks valid score evidence")
        art = EvalArtifact(
            uri=str(Path(self.forge.repo) / "runs" / "eval_gate" / name / "result.json"),
            produced_by=self.role, parents=[target.id],
            run_id=(self.ledger.run_id if self.ledger else ""),
            gate=Gate(passed=passed, metrics=result, verdict=result.get("verdict", "")),
            meta={"base": base, "run_name": name, "family": target.meta.get("family"),
                  "method": target.meta.get("method"), "target_uri": target.uri, **result},
        )
        self.registry.register(art)
        self._emit("tool", tool="register_artifact", ok=passed,
                   output=f"EvalArtifact {art.id}: {result.get('verdict')} "
                          f"(base {result.get('base_score')} -> {result.get('candidate_score')})")
        return art

    # --- model (serve-based) internal-eval gate ------------------------------------

    def _eval_model_internal(self, model: Artifact, *, baseline: Optional[dict] = None) -> Artifact:
        """Compare the candidate with its immediate source checkpoint."""
        from forgewright.registry.integrity import capture_manifest, IntegrityError
        try:
            checkpoint = capture_manifest(model.uri, repo=self.forge.repo, checkpoint=True)
        except IntegrityError as error:
            return self._model_eval_artifact(model, {}, baseline, served=False, integrity_error=str(error))
        if baseline is None:
            parent = next((p for aid in model.parents
                           if (p := self.registry.get(aid)) is not None and p.kind == "model"), None)
            if parent is None and model.meta.get("base"):
                from forgewright.contracts import ModelArtifact
                parent = ModelArtifact(uri=model.meta["base"], meta={"family": model.meta.get("family"),
                                                                    "variant": "source_baseline"})
            if parent is None:
                return self._model_eval_artifact(model, {}, None, served=True)
            baseline = self._measure_model(parent)
            if model.meta.get("role") != "abliterated":
                baseline = {"capability": baseline.get("capability")}
        metrics = self._measure_model(model)
        return self._model_eval_artifact(model, metrics, baseline, served=bool(metrics), checkpoint=checkpoint)

    def _measure_model(self, model: Artifact) -> dict:
        from forgewright.skills.model_identity import register_model
        from forgewright.skills.serving_opt import ServingOptimizer as Engine

        variant = register_model(self.forge.repo, model)
        family = model.meta["family"]
        engine = Engine(self.forge, self.jobs)
        engine.expected_model = model.meta.get("served_model_name") or f"model-forge/{family}-{variant}"
        try:
            if engine.launch(family, variant, {}).get("ok") and engine.wait_ready():
                return self._read_internal(engine, model, variant=variant)
            return {}
        finally:
            engine.stop()

    def _read_internal(self, engine, model: Artifact, *, variant: Optional[str] = None) -> dict:
        from forgewright.skills.abliterate import read_abliterate_metrics

        family = model.meta.get("family")
        variant = variant or model.meta.get("variant")
        if not family or not variant:
            return {}
        started = time.time()
        res = self.forge.run(f"eval {shlex.quote(family)} {shlex.quote(variant)} --internal", timeout=5400,
                             extra_env={"MODEL_FORGE_BASE_URL": engine.base_url})
        scores = engine._locate_scores_csv(res.output, since=started) if res.ok else None
        return read_abliterate_metrics(scores) if scores else {}

    def _model_eval_artifact(self, model: Artifact, metrics: dict, baseline: Optional[dict],
                             *, served: bool, checkpoint: Optional[dict] = None,
                             integrity_error: str = "") -> Artifact:
        gate = _model_gate(metrics, baseline, tolerance=self.tolerance, served=served)
        if checkpoint is not None:
            from forgewright.registry.integrity import verify_manifest, IntegrityError
            try:
                verify_manifest(model.uri, checkpoint, repo=self.forge.repo, checkpoint=True)
            except IntegrityError as error:
                integrity_error = str(error)
        if integrity_error:
            gate = Gate(False, metrics=gate.metrics, verdict=f"FAIL: {integrity_error}")
        art = EvalArtifact(
            uri="", produced_by=self.role, parents=[model.id],
            run_id=(self.ledger.run_id if self.ledger else ""),
            gate=gate, meta={"family": model.meta.get("family"), "variant": model.meta.get("variant"),
                             "baseline": baseline, "target_uri": model.uri,
                             "scope": "model_internal", **metrics,
                             "checkpoint": checkpoint, "integrity_error": integrity_error},
        )
        self.registry.register(art)
        self._emit("tool", tool="register_artifact", ok=gate.passed,
                   output=f"EvalArtifact {art.id}: {gate.verdict} (capability={metrics.get('capability')}, "
                          f"refusal={metrics.get('refusal_rate_harmful')})")
        return art

    def _read_result(self, name: str) -> dict:  # noqa: D401
        p = Path(self.forge.repo) / "runs" / "eval_gate" / name / "result.json"
        try:
            result = json.loads(p.read_text(encoding="utf-8"))
            if not isinstance(result, dict):
                raise ValueError("evaluation result must be an object")
            return result
        except (OSError, ValueError):
            return {"passed": False, "verdict": "FAIL: no result.json (eval did not complete)"}


def _model_gate(metrics: dict, baseline: Optional[dict], *, tolerance: float = 0.05,
                served: bool = True) -> Gate:
    """Pure gate for a model internal-eval. Capability must hold vs baseline; if baseline
    carries a refusal rate (abliterate context), refusal must also drop. Missing evidence fails."""
    from forgewright.skills.serving_opt import valid_quality
    if not served:
        return Gate(False, verdict="FAIL: server did not become ready")
    cap = metrics.get("capability")
    if not valid_quality(cap):
        return Gate(False, metrics=metrics, verdict="FAIL: no capability score from internal eval")
    if not baseline or not valid_quality(baseline.get("capability")):
        return Gate(False, metrics=metrics, verdict="FAIL: no valid baseline capability score")
    reasons: list[str] = []
    base_cap = baseline.get("capability")
    cap_held = base_cap is None or cap >= base_cap - tolerance
    if not cap_held:
        reasons.append("capability regressed")
    extra: dict = {"capability_delta": (cap - base_cap) if base_cap is not None else None}
    passed = cap_held
    if "refusal_rate_harmful" in baseline:
        if any(not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1
               for v in (baseline["refusal_rate_harmful"], metrics.get("refusal_rate_harmful"))):
            return Gate(False, metrics=metrics, verdict="FAIL: missing or invalid refusal score")
        drop = baseline["refusal_rate_harmful"] - metrics["refusal_rate_harmful"]
        extra["refusal_drop"] = drop
        if drop < 0.10:
            passed = False
            reasons.append("refusal did not drop")
    verdict = "PASS" if passed else "FAIL: " + ", ".join(reasons)
    return Gate(passed, metrics={**metrics, **extra}, verdict=verdict)
