"""Evaluator specialist — the swarm's gate authority. Produces EvalArtifacts.

For a verifiable adapter it runs the self-contained held-out gate (`skills/eval_gate.py`):
generate base-vs-adapter on a held-out {prompt,answer} set, score with the training reward,
and emit a PASS/REGRESSION verdict — using the robust loader that applies the adapter on
unified-arch models and FAILS LOUD if it doesn't. The EvalArtifact's gate is the decision
the Director keys global gating on.
"""
from __future__ import annotations

import json
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
    accepts = ("adapter", "model")
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

    def run(self, inputs: Sequence[Artifact], goal: str = "", *, holdout: Optional[str] = None) -> Artifact:
        self.validate_inputs(inputs)
        target = inputs[0]
        base = target.meta.get("base")
        adapter = target.uri
        holdout = holdout or target.meta.get("holdout")
        if not base or not holdout:
            raise ValueError("Evaluator needs base (adapter.meta['base']) and a holdout {prompt,answer} jsonl")
        name = target.meta.get("run_name") or target.id

        self._emit("assistant", content=f"eval {target.id} (base {base}) on holdout {holdout}")
        write_eval_gate(self.forge.repo, name, base=base, adapter=adapter,
                        holdout=holdout, tolerance=self.tolerance, overwrite=True)
        cmd = build_eval_gate_command(name)
        rec = self.jobs.launch(cmd, host=self.host, cwd=str(self.forge.repo), name=f"eval-{name}")
        self._emit("tool", tool="launch_job", ok=True, output=f"job {rec['id']} eval-gate {name}")
        self.jobs.wait(rec["id"])

        result = self._read_result(name)
        passed = bool(result.get("passed"))
        art = EvalArtifact(
            uri=str(Path(self.forge.repo) / "runs" / "eval_gate" / name / "result.json"),
            produced_by=self.role, parents=[target.id],
            run_id=(self.ledger.run_id if self.ledger else ""),
            gate=Gate(passed=passed, metrics=result, verdict=result.get("verdict", "")),
            meta={"base": base, "run_name": name, **result},
        )
        self.registry.register(art)
        self._emit("tool", tool="register_artifact", ok=passed,
                   output=f"EvalArtifact {art.id}: {result.get('verdict')} "
                          f"(base {result.get('base_score')} -> {result.get('candidate_score')})")
        return art

    def _read_result(self, name: str) -> dict:
        p = Path(self.forge.repo) / "runs" / "eval_gate" / name / "result.json"
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"passed": False, "verdict": "FAIL: no result.json (eval did not complete)"}
