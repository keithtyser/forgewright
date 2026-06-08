"""Recipe library — named DAGs (chains) over the specialist roster for the Director.

A recipe is a builder that returns ``(steps, seed_inputs)``. The Director runs the steps,
passing each produced artifact to the next, gated globally. NL-goal -> recipe selection is
the Director's brain job (later); these are the canonical, hand-built pipelines.
"""
from __future__ import annotations

from typing import Callable, Sequence

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
from forgewright.contracts import Artifact, ModelArtifact


def uplift(*, family: str, source: str, seed_paths: Sequence[str], holdout: str,
           max_steps: int = 60) -> tuple[list[Step], list[Artifact]]:
    """Curate -> distillation SFT -> held-out eval gate. (Proven end-to-end.)"""
    return [
        Step(DataCurator, run_kwargs={"mode": "curate_seed", "seed_paths": list(seed_paths),
             "family": family, "source": source, "run_name": f"{family}_uplift", "holdout": holdout}),
        Step(SFTTrainer, run_kwargs={"max_steps": max_steps}),
        Step(Evaluator, run_kwargs={"holdout": holdout}),
    ], []


def task_grpo(*, family: str, source: str, dataset: Artifact, holdout: str,
              max_steps: int = 120) -> tuple[list[Step], list[Artifact]]:
    """GRPO/RLVR on a verifiable {prompt,answer} dataset -> held-out eval gate."""
    return [
        Step(RLTrainer, run_kwargs={"max_steps": max_steps}),
        Step(Evaluator, run_kwargs={"holdout": holdout}),
    ], [dataset]


def uplift_publish(*, family: str, source: str, seed_paths: Sequence[str], holdout: str,
                   release_class: str = "public_model", max_steps: int = 60
                   ) -> tuple[list[Step], list[Artifact]]:
    """Curate -> SFT -> eval -> human-gated publish."""
    steps, seed = uplift(family=family, source=source, seed_paths=seed_paths, holdout=holdout, max_steps=max_steps)
    steps.append(Step(Publisher, run_kwargs={"release_class": release_class}))
    return steps, seed


def quantize_serve(*, model: ModelArtifact, objective: str = "latency") -> tuple[list[Step], list[Artifact]]:
    """Quantize a model to NVFP4 -> serving-opt (which gates quality vs the source)."""
    return [
        Step(Quantizer),
        Step(ServingOptimizer, run_kwargs={"objective": objective},
             compensate=_stop_served_endpoint),
    ], [model]


def abliterate(*, model: ModelArtifact, strength: float = 3.0) -> tuple[list[Step], list[Artifact]]:
    """Refusal-direction abliteration of an existing model (Evaluator model-gate is a follow-on)."""
    return [Step(Abliterator, run_kwargs={"strength": strength})], [model]


def full(*, family: str, source: str, seed_paths: Sequence[str], holdout: str,
         strength: float = 3.0, max_steps: int = 60) -> tuple[list[Step], list[Artifact]]:
    """The full pipeline in one run, gated after every transform so a regression can't flow
    downstream: curate -> SFT -> EVAL -> merge -> abliterate -> EVAL -> quantize -> EVAL ->
    serving-opt. (Merge bridges the adapter->model gap; ServingOptimizer self-gates quality.)"""
    return [
        Step(DataCurator, run_kwargs={"mode": "curate_seed", "seed_paths": list(seed_paths),
             "family": family, "source": source, "run_name": f"{family}_uplift", "holdout": holdout}),
        Step(SFTTrainer, run_kwargs={"max_steps": max_steps}),
        Step(Evaluator, run_kwargs={"holdout": holdout}),          # uplift quality
        Step(Merger),
        Step(Abliterator, run_kwargs={"strength": strength}),
        Step(Evaluator, run_kwargs={"holdout": holdout}),          # refusal drop + capability hold
        Step(Quantizer),
        Step(Evaluator, run_kwargs={"holdout": holdout}),          # quant quality retention
        Step(ServingOptimizer, run_kwargs={"objective": "latency"}, compensate=_stop_served_endpoint),
    ], []


def _stop_served_endpoint(art: Artifact) -> None:
    """Saga compensation: tear down a served endpoint if a later step fails."""
    import subprocess

    subprocess.run(["docker", "rm", "-f", "vllm_node"], capture_output=True)


# name -> builder, for discoverability (the brain selects by name later)
RECIPES: dict[str, Callable[..., tuple[list[Step], list[Artifact]]]] = {
    "uplift": uplift,
    "task_grpo": task_grpo,
    "uplift_publish": uplift_publish,
    "quantize_serve": quantize_serve,
    "abliterate": abliterate,
    "full": full,
}


def build_recipe(name: str, **params) -> tuple[list[Step], list[Artifact]]:
    if name not in RECIPES:
        raise ValueError(f"unknown recipe '{name}'; known: {sorted(RECIPES)}")
    return RECIPES[name](**params)


def plan_recipe_name(goal: str) -> str:
    """Pick the recipe that best fits a free-form goal, so the swarm can run a job even when the
    user didn't name a recipe. Maps intent keywords to the canonical pipelines."""
    g = (goal or "").lower()
    publishing = "publish" in g or "release" in g or "upload" in g
    trains = any(k in g for k in ("fine-tune", "finetune", "fine tune", "uplift", "sft", "distil", "train"))
    abliterates = any(k in g for k in ("abliterate", "uncensor", "uncensored", "refus", "jailbreak", "decensor"))
    quantizes = any(k in g for k in ("quantize", "quantization", "nvfp4", "fp8", "int8", "awq", "compress"))
    # a full pipeline: training plus a model transform, or explicitly "everything/end to end"
    if any(k in g for k in ("full pipeline", "everything", "end to end", "end-to-end", "whole pipeline")) \
            or (trains and (abliterates or quantizes)):
        return "full"
    if abliterates:
        return "abliterate"
    if any(k in g for k in ("grpo", "rlvr", "reinforcement", "reward", "verifiable", " rl ")):
        return "task_grpo"
    if any(k in g for k in ("quantize", "quantization", "nvfp4", "fp8", "int8", "awq", "compress",
                            "speed up", "speedup", "throughput", "latency", "faster serving")):
        return "quantize_serve"
    if any(k in g for k in ("fine-tune", "finetune", "fine tune", "uplift", "sft", "distil",
                            "train on", "improve", "teach")):
        return "uplift_publish" if publishing else "uplift"
    return "uplift"   # sensible default: curate -> SFT -> eval
