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
from forgewright.skills.quantize import choose_quant_method, quant_config_name, write_quant_config
from forgewright.tools.base import ToolRegistry
from forgewright.tools.forge import ForgeRunner, ForgeTool
from forgewright.tools.jobs import JobManager, LaunchJobTool, MonitorJobTool, TailLogsTool

# methods model-forge can export end to end (all via ModelOpt): NVFP4/FP8/INT8/AWQ.
_EXECUTABLE = ("nvfp4", "fp8", "int8", "awq")

_PROMPT = """\
You are the Quantizer specialist: quantize a model with the best method the GPU's arch supports
(NVFP4 on Blackwell, FP8 on Hopper/Ada, INT8/AWQ on Ampere), keeping routers/embeddings/lm_head
in higher precision. Gate on the measured speedup vs the bf16 baseline (no static tok/s floor).
You never publish; hand the quantized model on for serving-opt and eval.
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
            source_variant: str = "base", method: Optional[str] = None,
            supported_quant: Optional[Sequence[str]] = None) -> Artifact:
        self.validate_inputs(inputs)
        model = inputs[0]
        family = model.meta.get("family") or "model"
        # choose the method for the GPU's arch (explicit `method` wins; else best of supported_quant;
        # default nvfp4 when nothing is known, i.e. the Blackwell box).
        chosen = choose_quant_method(supported_quant, method) if (supported_quant or method) else "nvfp4"
        if chosen is None:
            self._emit("assistant", content="this GPU supports no quantization; serving the bf16 model unchanged")
            return self._passthrough(model, family, "NO_QUANT: GPU arch supports no quantization (serve bf16)")
        if chosen not in _EXECUTABLE:
            self._emit("assistant", content=f"selected {chosen} for this arch; model-forge {chosen} export "
                       "lands in Phase 2, skipping export for now")
            return self._passthrough(model, family, f"NO_QUANT: {chosen} export not yet available (serve bf16)")

        target_variant = f"base_{chosen}_modelopt"
        self._emit("assistant", content=f"{chosen} quantize {model.id} (family {family})")
        write_quant_config(self.forge.repo, family, method=chosen, arch=self.arch, source_variant=source_variant)
        cmd = (f"bash forge quantize export {family} {source_variant} "
               f"--config configs/quantization/{quant_config_name(family, chosen)}.yaml --execute")
        rec = self.jobs.launch(cmd, host=self.host, cwd=str(self.forge.repo), name=f"quant-{family}")
        self._emit("tool", tool="launch_job", ok=True, output=f"job {rec['id']} {chosen}-quantizing {family}")
        final = self.jobs.wait(rec["id"])
        ok = bool(final and final.get("exit_code") == 0)
        out = f"~/models/model-forge-quantized/{family}/{target_variant}"
        art = ModelArtifact(
            uri=out, produced_by=self.role, parents=[model.id],
            run_id=(self.ledger.run_id if self.ledger else ""),
            gate=Gate(passed=ok, metrics={"exit_code": (final or {}).get("exit_code"), "method": chosen},
                      verdict="QUANTIZED" if ok else "FAIL: export did not exit cleanly"),
            meta={"role": "quantized", "family": family, "method": f"{chosen}_modelopt",
                  "variant": target_variant, "served_model_name": f"model-forge/{family}-{target_variant}"},
        )
        self.registry.register(art)
        self._emit("tool", tool="register_artifact", ok=ok, output=f"quantized ModelArtifact {art.id}")
        return art

    def _passthrough(self, model: Artifact, family: str, verdict: str) -> Artifact:
        """No-quant outcome: register the source model as the served artifact (bf16), honestly gated."""
        art = ModelArtifact(
            uri=model.uri, produced_by=self.role, parents=[model.id],
            run_id=(self.ledger.run_id if self.ledger else ""),
            gate=Gate(passed=True, metrics={"method": "none"}, verdict=verdict),
            meta={"role": "base", "family": family, "method": "none"},
        )
        self.registry.register(art)
        return art
