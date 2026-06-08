"""SFTTrainer specialist — consumes a DatasetArtifact, produces an AdapterArtifact.

Wraps the proven uplift path (`skills/finetune.py` + the posttrain container runner):
scaffold the model-forge finetune config/manifest/registry pointing at the dataset, run
`forge finetune prepare`, launch the container train as a detached job, wait, parse the
final loss, and register a lineaged AdapterArtifact. Its gate is "trained successfully"
(job exit 0, finite final loss); the *capability* gate is the Evaluator's job.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Sequence

from forgewright.agents.base import Specialist
from forgewright.contracts import AdapterArtifact, Artifact, Gate
from forgewright.skills.finetune import build_container_train_command, write_finetune_config
from forgewright.skills.modelspec import model_spec
from forgewright.tools.base import ToolRegistry
from forgewright.tools.forge import ForgeRunner, ForgeTool
from forgewright.tools.jobs import JobManager, LaunchJobTool, MonitorJobTool, TailLogsTool

_PROMPT = """\
You are the SFTTrainer specialist in a post-training swarm. You take a curated dataset and
produce a LoRA adapter via uplift (distillation) SFT, then hand the adapter to the Evaluator.
Use the model-forge finetune engine through the posttrain container (assistant-only loss,
conservative LR, strict <think>/holdout hygiene already baked into the scaffolder). Never
publish. If training degenerates (repetition/format collapse), lower LR and resume.
"""


class SFTTrainer(Specialist):
    role = "SFTTrainer"
    accepts = ("dataset",)
    produces = "adapter"
    description = "Uplift (distillation) SFT: DatasetArtifact -> AdapterArtifact."

    def __init__(self, *, runner: Optional[ForgeRunner] = None, jobs: Optional[JobManager] = None,
                 host: Optional[str] = None, **kw) -> None:
        super().__init__(**kw)
        self.forge = runner or ForgeRunner()
        self.jobs = jobs or JobManager()
        self.host = host  # None = run on this box; else ssh target

    def system_prompt(self) -> str:
        return _PROMPT

    def tools(self) -> ToolRegistry:
        return ToolRegistry([
            ForgeTool(self.forge), LaunchJobTool(self.jobs),
            MonitorJobTool(self.jobs), TailLogsTool(self.jobs),
        ])

    def run(self, inputs: Sequence[Artifact], goal: str = "", *, max_steps: int = 60) -> Artifact:
        self.validate_inputs(inputs)
        dataset = next(a for a in inputs if a.kind == "dataset")
        family = dataset.meta.get("family") or "model"
        source = dataset.meta.get("source") or "Qwen/Qwen3.5-0.8B"
        name = dataset.meta.get("run_name") or f"{family}_uplift_v0"

        self._emit("assistant", content=f"SFT on {source} from dataset {dataset.id} ({dataset.uri})")
        # attach LoRA to the modules this architecture actually has (default decoder-LLM set if unknown)
        spec = model_spec(self.forge, source)
        lora_targets = (spec or {}).get("lora_target_modules") or None
        cfg, man, reg = write_finetune_config(
            self.forge.repo, family, name=name, source=source,
            data_path=dataset.uri, max_steps=max_steps, save_steps=max_steps, overwrite=True,
            lora_target_modules=lora_targets,
        )
        prep = self.forge.run(f"finetune --config configs/finetuning/{name}.yaml prepare --overwrite", timeout=900)
        if not prep.ok:
            raise RuntimeError(f"finetune prepare failed: {prep.output[:400]}")

        cmd = build_container_train_command(name)
        rec = self.jobs.launch(cmd, host=self.host, cwd=str(self.forge.repo), name=f"sft-{name}")
        self._emit("tool", tool="launch_job", ok=True, output=f"job {rec['id']} training {name}")
        final = self.jobs.wait(rec["id"])
        log = self.jobs.tail(rec["id"], n=200)
        loss = _final_loss(log)
        ok = bool(final and final.get("exit_code") == 0)

        out_dir = f"~/models/{source.rstrip('/').split('/')[-1]}-{name}"
        art = AdapterArtifact(
            uri=out_dir, produced_by=self.role, parents=[dataset.id],
            config_hash=_hash_file(cfg), run_id=(self.ledger.run_id if self.ledger else ""),
            gate=Gate(passed=ok and loss is not None, metrics={"final_loss": loss, "exit_code": (final or {}).get("exit_code")},
                      verdict="TRAINED" if ok else "FAIL: training did not exit cleanly"),
            meta={"base": source, "family": family, "method": "sft", "run_name": name},
        )
        self.registry.register(art)
        self._emit("tool", tool="register_artifact", ok=art.gate.passed,
                   output=f"AdapterArtifact {art.id} (final_loss={loss})")
        return art


def _final_loss(log: str) -> Optional[float]:
    losses = re.findall(r"'loss':\s*'?([0-9.]+)'?", log)
    try:
        return float(losses[-1]) if losses else None
    except ValueError:
        return None


def _hash_file(path: Path) -> str:
    import hashlib

    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]
    except OSError:
        return ""
