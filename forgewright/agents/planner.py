"""LLM stage-DAG planner: turn a free-form goal into a typed, validated chain of specialists.

The keyword heuristic (`recipes.plan_recipe_name`) only maps a goal onto ONE named recipe, so it
can't compose a novel multi-stage pipeline (e.g. "fine-tune on my data, then abliterate, then
quantize for my GPU, eval, and publish"). This planner asks the brain to order the specialist
roster into a chain, then **validates it against the typed artifact contracts** (each stage's
``accepts`` must match the prior stage's ``produces`` or the seed) before it runs. If the brain is
unavailable or the plan is invalid, the caller falls back to the keyword heuristic -- so planning
degrades safely rather than failing.
"""
from __future__ import annotations

import json
from typing import Any, Optional

from forgewright.agents.abliterator import Abliterator
from forgewright.agents.data_curator import DataCurator
from forgewright.agents.director import Step
from forgewright.agents.evaluator import Evaluator
from forgewright.agents.merger import Merger
from forgewright.agents.publisher import Publisher
from forgewright.agents.quantizer import Quantizer
from forgewright.agents.rl_trainer import RLTrainer
from forgewright.agents.serving_optimizer import ServingOptimizer
from forgewright.agents.sft import SFTTrainer
from forgewright.contracts import DatasetArtifact, ModelArtifact

# The roster, keyed by role, kept in sync with the specialist classes' typed contracts.
SPECIALISTS = {
    c.role: c for c in (DataCurator, SFTTrainer, RLTrainer, Merger, Abliterator, Quantizer,
                        ServingOptimizer, Evaluator, Publisher)
}

# Per-role: which flat params to forward as run_kwargs (the LLM doesn't set these; the user's
# params do). Keeps planned stages from receiving kwargs their run() doesn't accept.
_RUN_KWARGS = {
    "DataCurator": ("mode", "seed_paths", "family", "source", "run_name", "holdout"),
    "SFTTrainer": ("max_steps",),
    "RLTrainer": ("max_steps",),
    "Merger": (),
    "Abliterator": ("strength",),
    "Quantizer": ("source_variant", "method", "supported_quant"),
    "ServingOptimizer": ("objective",),
    "Evaluator": ("holdout",),
    "Publisher": ("release_class",),
}


def _roster_doc() -> str:
    lines = []
    for role, cls in SPECIALISTS.items():
        accepts = ", ".join(cls.accepts) if cls.accepts else "(a fresh goal)"
        lines.append(f"  {role}: accepts [{accepts}] -> produces '{cls.produces}'. {cls.description}")
    return "\n".join(lines)


_SYSTEM = (
    "You are the Director's planner for an LLM post-training swarm. Given a goal, output the "
    "ORDERED list of specialist stages that accomplishes it. Use ONLY these specialists, and make "
    "the chain TYPE-VALID: each stage's accepted artifact kind must equal the previous stage's "
    "produced kind (or the seed for the first stage).\n\nRoster:\n" + _roster_doc() +
    "\n\nRules: a fine-tune/uplift starts at DataCurator (produces a dataset) then SFTTrainer; "
    "RL starts from a dataset (RLTrainer); abliterate/quantize/serving-opt start from a model; put "
    "Evaluator after training/abliteration to gate; Publisher only last and only if the goal asks "
    "to publish/release. Keep it minimal -- no stage that the goal does not need.\n\n"
    "Respond with ONLY a JSON array of stage objects: "
    '[{"role": "<Role>"}, ...]. No prose.'
)


def validate_chain(roles: list[str], seed_kind: Optional[str]) -> tuple[bool, str]:
    """A chain is valid iff every stage accepts the current artifact kind (or a fresh goal for the
    first stage) and we track the produced kind forward. seed_kind is the kind of the seed input
    (e.g. 'model' or 'dataset'), or None when starting from a goal."""
    if not roles:
        return False, "empty plan"
    current = seed_kind
    for role in roles:
        cls = SPECIALISTS.get(role)
        if cls is None:
            return False, f"unknown specialist '{role}'"
        if cls.accepts:
            if current is None:
                return False, f"{role} needs a {cls.accepts} input but the chain has no upstream artifact"
            if current not in cls.accepts:
                return False, f"{role} accepts {cls.accepts}, but the previous stage produced '{current}'"
        # Evaluator is a GATE: it validates but passes the evaluated artifact through, so the next
        # stage (e.g. Publisher) sees the model/adapter, not the eval report.
        if cls.produces != "eval":
            current = cls.produces
    return True, "ok"


def _seed_kind(goal: str, params: dict) -> Optional[str]:
    """Infer the seed artifact kind from the params/goal (model vs dataset vs fresh goal)."""
    if params.get("model_uri"):
        return "model"
    if params.get("dataset_uri") or params.get("dataset"):
        return "dataset"
    return None


def plan_stages(goal: str, brain: Any, params: Optional[dict] = None) -> Optional[list[str]]:
    """Ask the brain for an ordered, type-valid list of specialist roles for the goal.
    Returns the role list, or None if the brain is unavailable / the plan is invalid."""
    if brain is None:
        return None
    params = params or {}
    try:
        turn = brain.chat([
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": f"Goal: {goal}\nAvailable seed: {_seed_kind(goal, params) or 'none'}"},
        ])
    except Exception:  # noqa: BLE001 - planning must never crash the run; fall back
        return None
    roles = _parse_roles(getattr(turn, "content", "") or "")
    if not roles:
        return None
    ok, _ = validate_chain(roles, _seed_kind(goal, params))
    return roles if ok else None


def _parse_roles(content: str) -> list[str]:
    """Pull a JSON array of {role} (or bare role strings) out of the model's reply."""
    start, end = content.find("["), content.rfind("]")
    if start == -1 or end == -1 or end < start:
        return []
    try:
        arr = json.loads(content[start:end + 1])
    except json.JSONDecodeError:
        return []
    roles = []
    for item in arr:
        if isinstance(item, str):
            roles.append(item)
        elif isinstance(item, dict) and item.get("role"):
            roles.append(str(item["role"]))
    return [r for r in roles if r in SPECIALISTS]


def build_plan(goal: str, params: dict, registry, brain: Any) -> Optional[tuple[list[Step], list]]:
    """Plan a typed stage chain with the brain and convert it to (steps, seed_inputs) for the
    Director. Returns None when planning is unavailable/invalid (caller uses the heuristic)."""
    roles = plan_stages(goal, brain, params)
    if not roles:
        return None
    steps = [Step(SPECIALISTS[r], run_kwargs=_run_kwargs(r, params)) for r in roles]
    seed = _seed_inputs(roles[0], params, registry)
    return steps, seed


def _run_kwargs(role: str, params: dict) -> dict:
    return {k: params[k] for k in _RUN_KWARGS.get(role, ()) if k in params}


def _seed_inputs(first_role: str, params: dict, registry) -> list:
    """Build the seed artifact the first stage consumes (model/dataset), or none for DataCurator."""
    accepts = SPECIALISTS[first_role].accepts
    if not accepts:                       # DataCurator: starts from the goal
        return []
    if "model" in accepts:
        uri = params.get("model_uri")
        model = (ModelArtifact(uri=uri, meta={"family": params.get("family"), "role": "base",
                                              "variant": params.get("variant", "base")})
                 if uri else registry.latest("model"))
        if model is None:
            raise ValueError(f"{first_role} needs a model_uri or a model in the registry")
        return [model]
    if "dataset" in accepts:
        uri = params.get("dataset_uri") or params.get("dataset")
        ds = (DatasetArtifact(uri=uri, meta={"family": params.get("family"), "source": params.get("source"),
                                             "run_name": params.get("run_name"), "holdout": params.get("holdout")})
              if uri else registry.latest("dataset"))
        if ds is None:
            raise ValueError(f"{first_role} needs a dataset_uri or a dataset in the registry")
        return [ds]
    return []
