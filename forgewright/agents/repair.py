"""Repair policies - the 'repair' arm of generate -> verify -> repair.

A specialist stage either fails outright or its gate rejects the result. Without a repair arm
that failure is terminal: the Director halts and the run is lost. A repair policy turns a gate
failure into an ADJUSTED retry - it inspects the failed artifact (its gate verdict + metrics) and
returns new run_kwargs for the next attempt, or None to give up (so a genuinely stuck stage still
halts honestly rather than retrying forever).

Each policy first consults OUTCOME MEMORY: if a prior run of the same (stage, family) PASSED the
gate, seed the retry from those hyperparameters instead of blindly nudging. Failing that, apply a
conservative, domain-aware delta. Policies are pure functions (no I/O beyond the read-only memory
lookup) so they are trivially testable.

Signature: repair(attempt, artifact, run_kwargs, memory=None) -> Optional[dict]
  attempt   - the retry number about to be made (2 = the first retry), 1-based on the NEXT try.
  artifact  - the failed artifact (artifact.gate has the verdict + metrics).
  run_kwargs- the run_kwargs used for the attempt that just failed.
  memory    - an OutcomeMemory, or None.
"""
from __future__ import annotations

from typing import Optional

from forgewright.contracts import Artifact


def _family(artifact: Artifact, run_kwargs: dict) -> str:
    return (artifact.meta.get("family") if artifact else None) or run_kwargs.get("family") or "model"


def _verdict(artifact: Artifact) -> str:
    return (artifact.gate.verdict if artifact and artifact.gate else "").lower()


def default_repair(attempt: int, artifact: Artifact, run_kwargs: dict, memory=None) -> Optional[dict]:
    """Plain bounded retry with the SAME params - for transient failures (a flaky env, an OOM the
    specialist already self-corrects, a load hiccup). Returns the kwargs unchanged so the Director
    re-runs the stage; bounding is enforced by the step's max_attempts, not here."""
    return dict(run_kwargs)


def abliterate_repair(attempt: int, artifact: Artifact, run_kwargs: dict, memory=None) -> Optional[dict]:
    """Capability regressed or weights didn't land -> back off the edit: lower strength and skip
    more early layers (the classic abliteration safety knobs). Seed from a known-good config in
    memory if we have one; otherwise nudge conservatively. Give up once strength bottoms out."""
    kw = dict(run_kwargs)
    fam = _family(artifact, run_kwargs)

    # 1) prefer a configuration that previously PASSED for this family
    if memory is not None:
        best = memory.best_params(stage="Abliterator", family=fam)
        if best and attempt == 2:  # only on the first retry, then fall through to nudging
            for k in ("strength", "layer_skip_first", "layer_skip_last"):
                if k in best:
                    kw[k] = best[k]
            return kw

    # 2) conservative back-off
    strength = float(kw.get("strength", 3.0))
    new_strength = round(strength * 0.7, 3)
    if new_strength < 0.5:
        return None  # bottomed out: retrying weaker won't remove refusals - halt honestly
    kw["strength"] = new_strength
    kw["layer_skip_first"] = int(kw.get("layer_skip_first", 4)) + 1
    return kw


def finetune_repair(attempt: int, artifact: Artifact, run_kwargs: dict, memory=None) -> Optional[dict]:
    """A regressed/degenerate fine-tune -> shorten the run (the trainer lowers LR / resumes from the
    last good checkpoint itself; fewer steps reduces over-fit/format collapse). Seed from memory if
    a passing run exists. Give up once steps get too small to learn anything."""
    kw = dict(run_kwargs)
    fam = _family(artifact, run_kwargs)
    if memory is not None:
        best = memory.best_params(stage="SFTTrainer", family=fam)
        if best and "max_steps" in best and attempt == 2:
            kw["max_steps"] = best["max_steps"]
            return kw
    steps = int(kw.get("max_steps", 60))
    new_steps = int(steps * 0.6)
    if new_steps < 10:
        return None
    kw["max_steps"] = new_steps
    return kw


# nvfp4 is the most aggressive; step DOWN only when no method was pinned.
_QUANT_METHOD_LADDER = {"nvfp4": "fp8", "fp8": "int8", "int8": "awq"}


def quantize_repair(attempt: int, artifact: Artifact, run_kwargs: dict, memory=None) -> Optional[dict]:
    """Recover quant QUALITY while KEEPING the requested method (the user picked it for their
    hardware -- we do not silently swap NVFP4 for something else). Each rung keeps more of the model
    in higher precision / improves calibration, trading a little speedup for accuracy (the user
    accepts a minimal perf drop). The method is only stepped DOWN when none was pinned (or fallback
    is explicitly allowed) AND the within-method ladder is exhausted; otherwise we stop honestly."""
    kw = dict(run_kwargs)
    excl = list(kw.get("extra_exclusions") or [])

    def add(*mods):
        for m in mods:
            if m not in excl:
                excl.append(m)

    if attempt == 2:
        # cheapest, highest-yield: better calibration + keep the KV cache in high precision
        # (recovers JSON/format adherence and refusal-boundary metrics first).
        kw["calib_samples"] = max(int(kw.get("calib_samples") or 256), 512)
        kw["keep_kv_high_precision"] = True
        return kw
    if attempt == 3:
        # keep the quality-sensitive projections in higher precision (mixed precision)
        add("down_proj", "o_proj")
        kw.update(extra_exclusions=excl, keep_kv_high_precision=True,
                  calib_samples=max(int(kw.get("calib_samples") or 256), 512),
                  min_speedup=min(float(kw.get("min_speedup") or 1.3), 1.2))
        return kw
    if attempt == 4:
        # also keep the most sensitive (first) decoder blocks in higher precision
        add("down_proj", "o_proj", "model.layers.0.", "model.layers.1.")
        kw.update(extra_exclusions=excl, keep_kv_high_precision=True,
                  calib_samples=max(int(kw.get("calib_samples") or 256), 768),
                  min_speedup=min(float(kw.get("min_speedup") or 1.3), 1.15))
        return kw
    # within-method recovery exhausted: switch method only if NOT pinned (or fallback allowed)
    pinned = bool(run_kwargs.get("method")) and not run_kwargs.get("allow_method_fallback")
    if not pinned:
        nxt = _QUANT_METHOD_LADDER.get(kw.get("method") or "nvfp4")
        if nxt:
            return {**kw, "method": nxt, "extra_exclusions": [], "keep_kv_high_precision": False}
    return None  # pinned method, ladder exhausted -> stop honestly (we tried hard)


# stage role -> default repair policy (recipes opt a step in by setting max_attempts > 1)
DEFAULT_POLICIES = {
    "Abliterator": abliterate_repair,
    "SFTTrainer": finetune_repair,
    "RLTrainer": finetune_repair,
    "Quantizer": quantize_repair,
}


def policy_for(role: str):
    """The domain repair policy for a role, or the plain bounded-retry policy."""
    return DEFAULT_POLICIES.get(role, default_repair)
