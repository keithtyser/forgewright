"""ServingOptimizer specialist — tune serving. ModelArtifact -> ServedEndpoint.

Wraps `skills/serving_opt.ServingOptimizer`: sweep candidate vLLM configs (speculative
decoding, batching) for the chosen objective (latency or throughput), benchmark each,
re-eval quality vs the source, and pick the best quality-preserving config. Registers a
ServedEndpoint carrying the winning serving config + measured tok/s.
"""
from __future__ import annotations

from typing import Optional, Sequence

from forgewright.agents.base import Specialist
from forgewright.contracts import Artifact, Gate, ServedEndpoint
from forgewright.skills.serving_opt import ServingOptimizer as ServingOptEngine
from forgewright.skills.serving_opt import default_served_name
from forgewright.tools.base import ToolRegistry
from forgewright.tools.forge import ForgeRunner, ForgeTool
from forgewright.tools.jobs import JobManager, LaunchJobTool, TailLogsTool

_PROMPT = """\
You are the ServingOptimizer specialist: maximize tok/s for the requested objective (latency
single-stream OR aggregate throughput) WITHOUT degrading quality. Sweep speculative decoding
and batching, benchmark, and re-eval against the source quant. Only quality-preserving configs
qualify. You never publish.
"""


class ServingOptimizer(Specialist):
    role = "ServingOptimizer"
    accepts = ("model",)
    produces = "served_endpoint"
    description = "Sweep serving configs for latency|throughput -> best quality-preserving ServedEndpoint."

    def __init__(self, *, runner: Optional[ForgeRunner] = None, jobs: Optional[JobManager] = None,
                 host: Optional[str] = None, **kw) -> None:
        super().__init__(**kw)
        self.forge = runner or ForgeRunner()
        self.jobs = jobs or JobManager()
        self.host = host
        self.engine = ServingOptEngine(self.forge, self.jobs)

    def system_prompt(self) -> str:
        return _PROMPT

    def tools(self) -> ToolRegistry:
        return ToolRegistry([ForgeTool(self.forge), LaunchJobTool(self.jobs), TailLogsTool(self.jobs)])

    def run(self, inputs: Sequence[Artifact], goal: str = "", *, objective: str = "latency",
            eval_each: bool = True) -> Artifact:
        self.validate_inputs(inputs)
        model = inputs[0]
        family = model.meta.get("family") or "model"
        variant = model.meta.get("variant") or "base_nvfp4_modelopt"
        self._emit("assistant", content=f"serving-opt {model.id} objective={objective}")
        results, best = self.engine.run(family, variant, objective, eval_each=eval_each)  # type: ignore[arg-type]
        served = model.meta.get("served_model_name") or default_served_name(family, variant)
        ok = best is not None
        art = ServedEndpoint(
            uri=self.engine.base_url, produced_by=self.role, parents=[model.id],
            run_id=(self.ledger.run_id if self.ledger else ""),
            gate=Gate(passed=ok, metrics={"best": best.name if best else None, "objective": objective,
                                          "single_tps": getattr(best, "single_tps", None) if best else None,
                                          "aggregate_tps": getattr(best, "aggregate_tps", None) if best else None},
                      verdict=(f"OPTIMIZED:{best.name}" if ok else "FAIL: no quality-preserving config beat baseline")),
            meta={"served_model_name": served, "family": family, "variant": variant,
                  "objective": objective, "best_config": best.name if best else None},
        )
        self.registry.register(art)
        self._emit("tool", tool="register_artifact", ok=ok,
                   output=f"ServedEndpoint {art.id} best={best.name if best else None}")
        return art
