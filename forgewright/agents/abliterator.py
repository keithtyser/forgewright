"""Abliterator specialist — remove refusals. ModelArtifact -> ModelArtifact(abliterated).

Wraps `skills/abliterate.py` + model-forge's pipeline run in the posttrain container via the
canonical `scripts/run_in_container.sh`. Scaffolds the config (capability-preserving scar
defaults), runs collect then export, and registers the abliterated standalone model. The
dual gate (refusal must DROP and capability HOLD) is enforced downstream by the Evaluator.
"""
from __future__ import annotations

from typing import Optional, Sequence

from forgewright.agents.base import Specialist
from forgewright.contracts import Artifact, Gate, ModelArtifact
from forgewright.skills.abliterate import write_abliterate_config
from forgewright.tools.base import ToolRegistry
from forgewright.tools.forge import ForgeRunner, ForgeTool
from forgewright.tools.jobs import JobManager, LaunchJobTool, MonitorJobTool, TailLogsTool

_PROMPT = """\
You are the Abliterator specialist: remove refusal behavior via contrastive refusal-direction
projection while preserving capability. Project mid layers only; leave embeddings/lm_head/MoE
experts untouched; keep strength conservative (over-abliteration breaks benign answers). You
never publish; hand the abliterated model on for eval (refusal must drop AND capability hold).
"""

_IMG_RUN = "bash scripts/run_in_container.sh python3 -m model_forge.pipelines.abliterate"


class Abliterator(Specialist):
    role = "Abliterator"
    accepts = ("model",)
    produces = "model"
    description = "Refusal-direction abliteration of a ModelArtifact -> abliterated ModelArtifact."

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

    def run(self, inputs: Sequence[Artifact], goal: str = "", *, strength: float = 3.0,
            layer_skip_first: int = 4, layer_skip_last: int = 2, layer_start: int = 4,
            layer_end: int = 24) -> Artifact:
        self.validate_inputs(inputs)
        model = inputs[0]
        family = model.meta.get("family") or "model"
        source = model.uri
        name = f"{family}_abliterated_v0"
        self._emit("assistant", content=f"abliterate {model.id} (family {family}, strength {strength})")
        cfg = write_abliterate_config(
            self.forge.repo, family, name=name, source=source, local_dir=source,
            strength=strength, layer_skip_first=layer_skip_first, layer_skip_last=layer_skip_last,
            layer_start=layer_start, layer_end=layer_end, overwrite=True,
        )
        rel = f"configs/abliteration/{name}.yaml"
        # collect refusal directions, then project + export the standalone model
        cmd = f"{_IMG_RUN} --config {rel} collect --execute && {_IMG_RUN} --config {rel} export --execute"
        rec = self.jobs.launch(cmd, host=self.host, cwd=str(self.forge.repo), name=f"ablate-{family}")
        self._emit("tool", tool="launch_job", ok=True, output=f"job {rec['id']} abliterating {family}")
        final = self.jobs.wait(rec["id"])
        ok = bool(final and final.get("exit_code") == 0)
        stem = source.rstrip("/").split("/")[-1]
        out = f"~/models/{stem}-{name}"
        art = ModelArtifact(
            uri=out, produced_by=self.role, parents=[model.id],
            config_hash="", run_id=(self.ledger.run_id if self.ledger else ""),
            gate=Gate(passed=ok, metrics={"exit_code": (final or {}).get("exit_code"), "strength": strength},
                      verdict="ABLITERATED" if ok else "FAIL: abliteration did not exit cleanly"),
            meta={"role": "abliterated", "family": family, "base": source},
        )
        self.registry.register(art)
        self._emit("tool", tool="register_artifact", ok=ok, output=f"abliterated ModelArtifact {art.id}")
        return art
