"""Quantizer specialist — NVFP4 quantization. ModelArtifact -> ModelArtifact(quantized).

Wraps `skills/quantize.py` + model-forge's ModelOpt container export. Scaffolds the quant
config (speedup-based gate, no static floor) if missing, runs `forge quantize export
--execute` as a detached job, and registers the quantized model. The speedup/quality gate
itself is enforced downstream by the ServingOptimizer/Evaluator (or `forge quantize nvfp4-gate`).
"""
from __future__ import annotations

from typing import Optional, Sequence

from forgewright.agents.base import Specialist
from forgewright.contracts import Artifact, Gate, ModelArtifact
from forgewright.skills.quantize import write_quant_config
from forgewright.tools.base import ToolRegistry
from forgewright.tools.forge import ForgeRunner, ForgeTool
from forgewright.tools.jobs import JobManager, LaunchJobTool, MonitorJobTool, TailLogsTool

_PROMPT = """\
You are the Quantizer specialist: quantize a model to Blackwell NVFP4 (NVIDIA ModelOpt),
keeping routers/embeddings/lm_head in higher precision. Gate on the measured speedup vs the
bf16 baseline (no static tok/s floor). You never publish; hand the quantized model on for
serving-opt and eval.
"""


class Quantizer(Specialist):
    role = "Quantizer"
    accepts = ("model",)
    produces = "model"
    description = "NVFP4 quantize a ModelArtifact -> quantized ModelArtifact."

    def __init__(self, *, runner: Optional[ForgeRunner] = None, jobs: Optional[JobManager] = None,
                 host: Optional[str] = None, arch: str = "qwen", **kw) -> None:
        super().__init__(**kw)
        self.forge = runner or ForgeRunner()
        self.jobs = jobs or JobManager()
        self.host = host
        self.arch = arch

    def system_prompt(self) -> str:
        return _PROMPT

    def tools(self) -> ToolRegistry:
        return ToolRegistry([
            ForgeTool(self.forge), LaunchJobTool(self.jobs),
            MonitorJobTool(self.jobs), TailLogsTool(self.jobs),
        ])

    def run(self, inputs: Sequence[Artifact], goal: str = "", *,
            source_variant: str = "base", target_variant: str = "base_nvfp4_modelopt") -> Artifact:
        self.validate_inputs(inputs)
        model = inputs[0]
        family = model.meta.get("family") or "model"
        self._emit("assistant", content=f"NVFP4 quantize {model.id} (family {family})")
        write_quant_config(self.forge.repo, family, arch=self.arch, source_variant=source_variant)
        cmd = (f"bash forge quantize export {family} {source_variant} "
               f"--config configs/quantization/{family}_nvfp4_modelopt.yaml --execute")
        rec = self.jobs.launch(cmd, host=self.host, cwd=str(self.forge.repo), name=f"quant-{family}")
        self._emit("tool", tool="launch_job", ok=True, output=f"job {rec['id']} quantizing {family}")
        final = self.jobs.wait(rec["id"])
        ok = bool(final and final.get("exit_code") == 0)
        out = f"~/models/model-forge-quantized/{family}/{target_variant}"
        art = ModelArtifact(
            uri=out, produced_by=self.role, parents=[model.id],
            run_id=(self.ledger.run_id if self.ledger else ""),
            gate=Gate(passed=ok, metrics={"exit_code": (final or {}).get("exit_code")},
                      verdict="QUANTIZED" if ok else "FAIL: export did not exit cleanly"),
            meta={"role": "quantized", "family": family, "method": "nvfp4_modelopt",
                  "variant": target_variant, "served_model_name": f"model-forge/{family}-{target_variant}"},
        )
        self.registry.register(art)
        self._emit("tool", tool="register_artifact", ok=ok, output=f"quantized ModelArtifact {art.id}")
        return art
