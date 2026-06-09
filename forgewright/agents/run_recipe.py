"""RunRecipeTool — the bridge that lets the conversational agent dispatch the swarm.

The chat front is a tool-using Agent (its brain reads the user's goal). This tool is how
that agent hands a post-training job to the Director: it builds a named recipe from the
agent's params, runs the Director (which dispatches the specialist swarm, gates globally,
rolls back on failure), and returns the lineage. The Director is given this turn's reporter
and permissions, so specialist progress streams into the one transcript and the Publisher's
approval surfaces at the single prompt.
"""
from __future__ import annotations

import json
from typing import Any, Optional

from forgewright.agents.director import Director
from forgewright.agents.memory import OutcomeMemory
from forgewright.agents.recipes import RECIPES, build_recipe, plan_recipe_name
from forgewright.contracts import DatasetArtifact, ModelArtifact
from forgewright.registry import Registry
from forgewright.tools.base import Tool, ToolResult
from forgewright.tools.forge import ForgeRunner
from forgewright.tools.jobs import JobManager


def _resolve(recipe: str, params: dict, registry: Registry):
    """Map the agent's flat params into the recipe builder's args + seed artifacts.

    Recipes that start from a model/dataset get a seed artifact built from params
    (model_uri / dataset_uri) or pulled from the registry (latest of that kind)."""
    p = dict(params)
    if recipe in ("uplift", "uplift_publish", "full"):
        p.setdefault("seed_paths", [p.pop("seed_path")] if p.get("seed_path") else [])
        return build_recipe(recipe, **p)
    if recipe == "task_grpo":
        ds_uri = p.pop("dataset_uri", None) or p.pop("dataset", None)
        ds = (registry.latest("dataset") if not ds_uri
              else DatasetArtifact(uri=ds_uri, meta={"family": p.get("family"), "source": p.get("source"),
                                                     "run_name": p.get("run_name"), "holdout": p.get("holdout")}))
        if ds is None:
            raise ValueError("task_grpo needs a dataset_uri or a dataset in the registry")
        return build_recipe("task_grpo", family=p.get("family", "model"), source=p.get("source", ""),
                            dataset=ds, holdout=p["holdout"], max_steps=p.get("max_steps", 120))
    if recipe in ("quantize_serve", "abliterate"):
        m_uri = p.pop("model_uri", None)
        model = (registry.latest("model") if not m_uri
                 else ModelArtifact(uri=m_uri, meta={"family": p.get("family"), "role": "base",
                                                     "variant": p.get("variant", "base")}))
        if model is None:
            raise ValueError(f"{recipe} needs a model_uri or a model in the registry")
        kw = {"model": model}
        if recipe == "quantize_serve":
            kw["objective"] = p.get("objective", "latency")
        else:
            kw["strength"] = p.get("strength", 3.0)
        return build_recipe(recipe, **kw)
    raise ValueError(f"unknown recipe '{recipe}'; known: {sorted(RECIPES)}")


class RunRecipeTool(Tool):
    name = "run_recipe"
    description = (
        "Dispatch the post-training SPECIALIST SWARM via the Director by running a named recipe. "
        "Use this for end-to-end post-training jobs (vs calling forge stages yourself). "
        "recipe: 'uplift' (curate->SFT->eval), 'task_grpo' (GRPO->eval), 'quantize_serve' "
        "(quant->serving-opt), 'abliterate', 'uplift_publish', or 'auto' (pick from the goal). params: "
        "uplift -> {family, source, seed_paths|seed_path, holdout, max_steps}; task_grpo -> "
        "{family, source, dataset_uri, holdout, max_steps}; quantize_serve/abliterate -> "
        "{family, model_uri, objective|strength}. Long-running; returns the artifact lineage + gate."
    )
    risk = "exec"
    parameters = {
        "type": "object",
        "properties": {
            "recipe": {"type": "string", "description": "uplift|task_grpo|quantize_serve|abliterate|uplift_publish"},
            "params": {"type": "object", "description": "recipe params (see description)"},
            "goal": {"type": "string", "description": "the user's goal, for the transcript"},
        },
        "required": ["recipe"],
    }

    def __init__(self, *, registry: Optional[Registry] = None, jobs: Optional[JobManager] = None,
                 forge: Optional[ForgeRunner] = None, host: Optional[str] = None, brain=None) -> None:
        self.registry = registry or Registry()
        self.jobs = jobs or JobManager()
        self.forge = forge or ForgeRunner()
        self.host = host
        self.brain = brain   # used to LLM-plan a stage chain for 'auto'/novel goals
        self.memory = OutcomeMemory()   # cross-run learning loop (grounds planning + repair)
        # bound by the CLI each turn so the swarm streams into the one transcript + uses the gate
        self.reporter = None
        self.permissions = None

    def _note(self, content: str) -> None:
        if self.reporter:
            try:
                self.reporter("assistant", {"role": "Director", "content": content})
            except Exception:  # noqa: BLE001
                pass

    def bind(self, *, reporter=None, permissions=None) -> None:
        """Wire this turn's transcript reporter + approval policy into the swarm."""
        if reporter is not None:
            self.reporter = reporter
        if permissions is not None:
            self.permissions = permissions

    def run(self, recipe: str, params: Optional[dict] = None, goal: str = "", **_: Any) -> ToolResult:
        params = params or {}
        if isinstance(params, str):  # some brains pass JSON as a string
            try:
                params = json.loads(params)
            except json.JSONDecodeError:
                return ToolResult(False, f"params must be a JSON object, got: {params[:200]}")
        # 'auto' (or an unrecognized name) -> plan the pipeline from the goal so the swarm can run
        # even when the user didn't name a recipe. Try the LLM stage-DAG planner first (it can
        # compose novel multi-stage chains), then fall back to the keyword heuristic.
        if recipe == "auto" or recipe not in RECIPES:
            steps = seed = None
            try:
                from forgewright.agents.planner import build_plan
                planned = build_plan(goal or recipe, params, self.registry, self.brain, self.memory)
                if planned:
                    steps, seed = planned
                    self._note(f"planned {len(steps)}-stage swarm: "
                               + " -> ".join(s.specialist_cls.role for s in steps))
            except Exception as e:  # noqa: BLE001 - planning must never crash the run
                self._note(f"LLM planning unavailable ({e}); using the keyword heuristic")
            if steps is None:
                recipe = plan_recipe_name(goal or recipe)
                self._note(f"planned recipe '{recipe}' from the goal")
                try:
                    steps, seed = _resolve(recipe, params, self.registry)
                except Exception as e:  # noqa: BLE001
                    return ToolResult(False, f"recipe setup failed: {e}")
        else:
            try:
                steps, seed = _resolve(recipe, params, self.registry)
            except Exception as e:  # noqa: BLE001
                return ToolResult(False, f"recipe setup failed: {e}")
        director = Director(registry=self.registry, reporter=self.reporter,
                            permissions=self.permissions, host=self.host,
                            brain=self.brain, memory=self.memory)
        res = director.run_recipe(goal or recipe, steps, seed)
        lineage = [f"{a.kind}:{a.id}" for a in res.artifacts]
        summary = (f"recipe '{recipe}' {'completed' if res.ok else 'HALTED at ' + res.failed_at}: "
                   f"{res.reason or 'ok'}\nlineage: {lineage}")
        return ToolResult(res.ok, summary,
                          {"ok": res.ok, "final": res.final.id if res.final else None,
                           "failed_at": res.failed_at, "lineage": lineage})
