"""RLTrainer specialist — verifiable-reward RL (GRPO/RLVR). Dataset -> AdapterArtifact.

Wraps `trainers/rl.py` (TRL GRPOTrainer with the KL-anchor + PPO-clip + DAPO scars) run in
the posttrain container. Consumes a {prompt, answer} dataset; produces a LoRA adapter. Gate
is "trained cleanly" (job exit 0, final reward present); the held-out capability gate is the
Evaluator's job.
"""
from __future__ import annotations

import re
from typing import Optional, Sequence

from forgewright.agents.base import Specialist
from forgewright.contracts import AdapterArtifact, Artifact, Gate
from forgewright.tools.base import ToolRegistry
from forgewright.tools.forge import ForgeRunner, ForgeTool
from forgewright.tools.jobs import JobManager, LaunchJobTool, MonitorJobTool, TailLogsTool
from forgewright.trainers.rl import build_grpo_train_command, write_grpo_run

_PROMPT = """\
You are the RLTrainer specialist: improve a model on a VERIFIABLE task via GRPO/RLVR. The
reward is checkable (numeric correctness + <think> format). Keep it stable with the KL anchor,
PPO clip, and DAPO overlong masking already baked in. You never publish. If reward collapses,
raise the KL anchor (beta) and lower LR.
"""


class RLTrainer(Specialist):
    role = "RLTrainer"
    accepts = ("dataset",)
    produces = "adapter"
    description = "GRPO/RLVR on a verifiable {prompt,answer} dataset -> AdapterArtifact."

    def __init__(self, *, runner: Optional[ForgeRunner] = None, jobs: Optional[JobManager] = None,
                 host: Optional[str] = None, **kw) -> None:
        super().__init__(**kw)
        self.forge = runner or ForgeRunner()
        self.jobs = jobs or JobManager()
        self.host = host

    def system_prompt(self) -> str:
        return _PROMPT

    def tools(self) -> ToolRegistry:
        return ToolRegistry([
            ForgeTool(self.forge), LaunchJobTool(self.jobs),
            MonitorJobTool(self.jobs), TailLogsTool(self.jobs),
        ])

    def run(self, inputs: Sequence[Artifact], goal: str = "", *, max_steps: int = 60) -> Artifact:
        self.validate_inputs(inputs)
        ds = next(a for a in inputs if a.kind == "dataset")
        source = ds.meta.get("source") or "Qwen/Qwen3.5-0.8B"
        family = ds.meta.get("family") or "model"
        name = ds.meta.get("run_name") or f"{family}_grpo_v0"
        self._emit("assistant", content=f"GRPO on {source} from {ds.id} ({ds.uri})")
        write_grpo_run(self.forge.repo, name, source=source, data_path=ds.uri,
                       max_steps=max_steps, overwrite=True)
        rec = self.jobs.launch(build_grpo_train_command(name), host=self.host,
                               cwd=str(self.forge.repo), name=f"grpo-{name}")
        self._emit("tool", tool="launch_job", ok=True, output=f"job {rec['id']} GRPO {name}")
        final = self.jobs.wait(rec["id"])
        reward = _final_reward(self.jobs.tail(rec["id"], n=300))
        ok = bool(final and final.get("exit_code") == 0)
        out_dir = f"~/models/{source.rstrip('/').split('/')[-1]}-{name}"
        art = AdapterArtifact(
            uri=out_dir, produced_by=self.role, parents=[ds.id],
            run_id=(self.ledger.run_id if self.ledger else ""),
            gate=Gate(passed=ok, metrics={"final_reward": reward, "exit_code": (final or {}).get("exit_code")},
                      verdict="TRAINED" if ok else "FAIL: GRPO did not exit cleanly"),
            meta={"base": source, "family": family, "method": "grpo", "run_name": name,
                  **({"holdout": ds.meta["holdout"]} if ds.meta.get("holdout") else {})},
        )
        self.registry.register(art)
        self._emit("tool", tool="register_artifact", ok=ok, output=f"AdapterArtifact {art.id} (reward={reward})")
        return art


def _final_reward(log: str) -> Optional[float]:
    vals = re.findall(r"reward_correct/mean.?:\s*.?([0-9.]+)", log)
    try:
        return float(vals[-1]) if vals else None
    except ValueError:
        return None
