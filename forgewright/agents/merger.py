"""Merger specialist — fold a LoRA adapter into its base. AdapterArtifact -> ModelArtifact.

This bridges the typed gap between training (which produces an adapter) and the stages that need
a standalone model (abliterate, quantize, serving-opt), so a full pipeline can run in one chain.
Wraps model-forge's merge container (scripts/run_merge_peft_container.sh) and gates on real
provenance: it only claims success if THIS run wrote fresh weights into a unique output dir.
"""
from __future__ import annotations

import time
from typing import Optional, Sequence

from forgewright.agents.base import Specialist
from forgewright.contracts import Artifact, Gate, ModelArtifact
from forgewright.skills.hostexec import resolve_local, wrote_fresh_weights
from forgewright.tools.base import ToolRegistry
from forgewright.tools.forge import ForgeRunner, ForgeTool
from forgewright.tools.jobs import JobManager, LaunchJobTool, MonitorJobTool, TailLogsTool

_PROMPT = """\
You are the Merger specialist: merge a LoRA adapter into its base model to produce a standalone
checkpoint that downstream stages (abliterate, quantize, serving-opt) can consume. Preserve the
tokenizer/chat template and the model architecture. You never publish.
"""

_MERGE = "MODEL_FORGE_MIN_FREE_DISK_FRACTION=0.05 bash scripts/run_merge_peft_container.sh"


class Merger(Specialist):
    role = "Merger"
    accepts = ("adapter",)
    produces = "model"
    description = "Merge a LoRA adapter into its base -> standalone ModelArtifact (for abliterate/quantize)."

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

    def run(self, inputs: Sequence[Artifact], goal: str = "") -> Artifact:
        self.validate_inputs(inputs)
        adapter = inputs[0]
        family = adapter.meta.get("family") or "model"
        base = resolve_local(self.host, adapter.meta.get("base") or "")
        if not base:
            raise ValueError("Merger needs the adapter's base model (adapter.meta['base'])")
        stem = base.rstrip("/").split("/")[-1]
        out = f"~/models/{stem}-merged-{time.strftime('%Y%m%d-%H%M%S')}"   # unique -> no stale reuse
        self._emit("assistant", content=f"merge adapter {adapter.id} into {base}")
        cmd = (f"{_MERGE} --base-model {base} --adapter {adapter.uri} --output-dir {out} "
               "--trust-remote-code --overwrite")
        since = time.time()
        rec = self.jobs.launch(cmd, host=self.host, cwd=str(self.forge.repo), name=f"merge-{family}")
        self._emit("tool", tool="launch_job", ok=True, output=f"job {rec['id']} merging {family}")
        final = self.jobs.wait(rec["id"])
        ok = bool(final and final.get("exit_code") == 0) and wrote_fresh_weights(self.host, out, since)
        art = ModelArtifact(
            uri=out, produced_by=self.role, parents=[adapter.id],
            run_id=(self.ledger.run_id if self.ledger else ""),
            gate=Gate(passed=ok, metrics={"exit_code": (final or {}).get("exit_code")},
                      verdict="MERGED" if ok else "FAIL: merge did not write fresh weights"),
            meta={"role": "merged", "family": family, "base": base},
        )
        self.registry.register(art)
        self._emit("tool", tool="register_artifact", ok=ok, output=f"merged ModelArtifact {art.id}")
        return art
