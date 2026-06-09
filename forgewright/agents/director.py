"""Director — the orchestrator. Plans a recipe (DAG, currently a chain) over specialists,
dispatches them, passes each produced artifact to the next, and enforces the GLOBAL gate.

The user talks only to the Director; it spawns specialists on the backend and threads its
own (role-labelled) reporter into each so all their activity streams into the one transcript.
If a specialist's gate fails, the Director halts (saga-compensation hook in Slice D) — a
regression never silently flows downstream.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

from forgewright.agents.base import Specialist, label_reporter
from forgewright.agents.memory import OutcomeMemory
from forgewright.agents.repair import policy_for
from forgewright.contracts import Artifact
from forgewright.ledger.ledger import Ledger
from forgewright.permissions import PermissionPolicy
from forgewright.registry import Registry

Reporter = "callable"


@dataclass
class Step:
    """One stage of a recipe: which specialist + how to build/run it.

    `compensate(artifact)` is an optional saga rollback for this step's output, run by the
    Director (in reverse) when a LATER step fails (e.g. stop a served endpoint, mark a
    superseded artifact). Most stages produce immutable files and need none.

    `max_attempts` + `repair` add the generate -> verify -> repair arm: when a gate fails (or the
    stage raises), the Director re-runs the stage up to `max_attempts` times, calling `repair` to
    adjust run_kwargs between tries (returning None = give up). With max_attempts == 1 (the default)
    a failure halts immediately, exactly as before - repair is opt-in per recipe step.
    """
    specialist_cls: type[Specialist]
    init_kwargs: dict = field(default_factory=dict)   # constructor kwargs (host, tolerance, runner, jobs)
    run_kwargs: dict = field(default_factory=dict)     # run() kwargs (holdout, max_steps)
    compensate: Optional[Callable[[Artifact], None]] = None
    max_attempts: int = 1                              # bounded retries on gate failure (1 = no retry)
    repair: Optional[Callable[..., Optional[dict]]] = None  # (attempt, art, run_kwargs, memory) -> new kwargs|None


@dataclass
class DirectorResult:
    ok: bool
    artifacts: list[Artifact]
    final: Optional[Artifact] = None
    failed_at: str = ""
    reason: str = ""

    def lineage_ids(self) -> list[str]:
        return [a.id for a in self.artifacts]


class Director:
    role = "Director"

    def __init__(
        self,
        *,
        registry: Optional[Registry] = None,
        reporter=None,
        permissions: Optional[PermissionPolicy] = None,
        ledger: Optional[Ledger] = None,
        brain=None,
        host: Optional[str] = None,
        memory: Optional[OutcomeMemory] = None,
    ) -> None:
        self.registry = registry or Registry()
        self.reporter = label_reporter(reporter, self.role)
        self._base_reporter = reporter
        self.permissions = permissions or PermissionPolicy()
        self.ledger = ledger
        self.brain = brain
        self.host = host
        # the cross-run learning loop: every gated stage is recorded here, and repair policies +
        # the planner read it back to ground future runs.
        self.memory = memory or OutcomeMemory()

    def _emit(self, event_type: str, **data) -> None:
        if self.reporter:
            try:
                # an explicit role in `data` wins (e.g. artifact events colored by their
                # producer), otherwise the event is attributed to the Director. Naming the
                # positional `event_type` lets data carry its own `kind` (an artifact's kind).
                self.reporter(event_type, {"role": self.role, **data})
            except Exception:  # noqa: BLE001
                pass

    def run_recipe(
        self, goal: str, recipe: Sequence[Step], seed_inputs: Optional[Sequence[Artifact]] = None
    ) -> DirectorResult:
        """Execute the chain, gated globally, with a two-level generate -> verify -> repair loop:

          1. SELF-REPAIR: a stage that fails its OWN gate is re-run with repaired run_kwargs, up to
             its max_attempts (e.g. an abliterate that did not write fresh weights, a quant export
             that crashed).
          2. UPSTREAM-REPAIR: when a GATE stage (an Evaluator) fails, the transform that produced
             the artifact it evaluated is re-run with repaired params, then re-evaluated. This is
             how a quality regression flows back to its cause -- a quantized model that fails the
             quality gate re-quantizes LESS aggressively (keeping the requested method), an
             abliteration that regressed capability re-runs at lower strength -- instead of giving
             up. The transform's max_attempts bounds the recovery.

        Returns the produced artifacts (lineage) + final."""
        current: list[Artifact] = list(seed_inputs or [])
        produced: list[Artifact] = []
        completed: list[tuple[Step, Artifact, int]] = []   # (step, art, idx) of stages that PASSED
        inputs_at: dict[int, list[Artifact]] = {}          # idx -> the `current` list when it ran
        attempts: dict[int, int] = {}                      # idx -> attempts used (self + upstream)
        stages = [s.specialist_cls.role for s in recipe]
        total = len(recipe)
        # the full pipeline up front, so the UI can render a live progress map of the swarm.
        self._emit("pipeline", stages=stages, goal=goal)

        i = 0
        while i < len(recipe):
            step = recipe[i]
            role = step.specialist_cls.role
            attempt = attempts.get(i, 0) + 1
            attempts[i] = attempt
            inputs_at[i] = list(current)
            art, passed = self._run_attempt(step, current, goal, i, total, attempt)
            if art is not None:
                produced.append(art)
            if passed:
                self._emit("stage", name=role, index=i, total=total, state="done")
                completed.append((step, art, i))
                # A gate (Evaluator -> eval report) passes the evaluated artifact through, so a
                # later stage receives the model/adapter that passed, not the report itself.
                if art.kind != "eval":
                    current = [art]
                i += 1
                continue

            reason = (art.gate.verdict if (art is not None and art.gate) else "failed")
            # 1) self-repair this stage
            if self._try_repair(step, attempt, art, role, upstream=False):
                current = inputs_at[i]            # re-run the same stage with adjusted kwargs
                continue
            # 2) upstream-repair: re-run the transform that produced what this gate evaluated
            j = self._upstream_transform(completed)
            if j is not None and self._try_repair(recipe[j], attempts.get(j, 1), art,
                                                  recipe[j].specialist_cls.role, upstream=True):
                completed = [(s, a, k) for (s, a, k) in completed if k < j]   # rewind
                current = inputs_at[j]
                i = j
                continue
            # 3) give up honestly
            self._emit("stage", name=role, index=i, total=total, state="failed")
            self._emit("assistant", content=f"GLOBAL GATE halt at {role}: {reason}")
            self._compensate([(s, a) for (s, a, _k) in completed])
            return DirectorResult(False, produced, failed_at=role, reason=reason)

        final = produced[-1] if produced else None
        self._emit("assistant", content=f"recipe complete: {len(produced)} artifacts, "
                   f"lineage {[a.id for a in produced]}")
        return DirectorResult(True, produced, final=final)

    def _run_attempt(self, step: Step, current, goal: str, i: int, total: int, attempt: int):
        """Run ONE attempt of a stage. Returns (artifact, passed). Emits stage/artifact events and
        records the outcome; a raised stage becomes a failed-gate artifact so the caller can repair
        it uniformly."""
        from forgewright.contracts import Gate  # local import: contracts has no director dependency

        role = step.specialist_cls.role
        self._emit("stage", name=role, index=i, total=total,
                   state=("active" if attempt == 1 else "retry"), attempt=attempt)
        family = step.run_kwargs.get("family") or (current[0].meta.get("family") if current else "") or ""
        spec = step.specialist_cls(
            registry=self.registry,
            reporter=self._base_reporter,   # specialists self-label; one transcript
            permissions=self.permissions,   # all approvals bubble to the one prompt
            ledger=self.ledger,
            brain=self.brain,
            host=self.host,
            **step.init_kwargs,
        )
        eff_goal = goal if attempt == 1 else (
            f"{goal}\n[repair attempt {attempt}: the previous attempt failed; "
            f"adjusted parameters: {step.run_kwargs}.]")
        raised = False
        try:
            art = spec.run(current, eff_goal, **step.run_kwargs)
        except Exception as e:  # noqa: BLE001 - surface the failing stage, don't crash the chain
            raised = True
            self._emit("tool", tool=role, ok=False, output=f"error: {e}")
            art = Artifact(
                kind=(step.specialist_cls.produces or "model"), produced_by=role,
                parents=[a.id for a in current], meta={"family": family},
                gate=Gate(passed=False, metrics={}, verdict=f"raised: {e}"),
            )

        gate_passed = art.gate.passed if art.gate is not None else None
        gate_metrics = art.gate.metrics if art.gate is not None else {}
        verdict = art.gate.verdict if art.gate is not None else ""
        if not raised:
            self._emit("artifact", role=art.produced_by or role, kind=art.kind, id=art.id,
                       parents=list(art.parents), metrics=gate_metrics, passed=gate_passed)
        self._record_outcome(role, family, step.run_kwargs, gate_passed, gate_metrics, verdict,
                             ("" if raised else art.id), attempt)
        return art, (art.gate is None or art.gate.passed)

    def _try_repair(self, step: Step, attempt_used: int, failed_art: Artifact, role: str,
                    *, upstream: bool) -> bool:
        """Attempt to repair `step` (mutating its run_kwargs in place) given a failure. Returns True
        if a repair was applied (so the caller re-runs), False if no budget/policy or the policy gave
        up. `failed_art` carries the failure that motivates the repair (for upstream-repair it is the
        gate stage's artifact -- its verdict says WHY)."""
        repair = step.repair or (policy_for(step.specialist_cls.role) if step.max_attempts > 1 else None)
        if repair is None or attempt_used >= step.max_attempts:
            return False
        new_kwargs = self._invoke_repair(repair, attempt_used + 1, failed_art, dict(step.run_kwargs))
        if new_kwargs is None:
            self._emit("assistant", content=f"{role} repair gave up after attempt {attempt_used}")
            return False
        self._emit("repair", name=role, attempt=attempt_used + 1, upstream=upstream,
                   reason=(failed_art.gate.verdict if (failed_art and failed_art.gate) else ""),
                   changes=self._diff(step.run_kwargs, new_kwargs))
        step.run_kwargs = new_kwargs
        return True

    @staticmethod
    def _upstream_transform(completed: "list[tuple[Step, Artifact, int]]") -> Optional[int]:
        """The index of the most recent completed TRANSFORM (non-eval) stage -- the producer of the
        artifact a failing gate just evaluated. None if there is no upstream transform to repair."""
        for step, _art, idx in reversed(completed):
            if step.specialist_cls.produces != "eval":
                return idx
        return None

    def _invoke_repair(self, repair, attempt: int, art: Artifact, run_kwargs: dict):
        """Call a repair policy defensively - a buggy policy must not crash the run (treat an error
        as 'give up')."""
        try:
            return repair(attempt, art, run_kwargs, memory=self.memory)
        except Exception as e:  # noqa: BLE001
            self._emit("tool", tool="repair", ok=False, output=str(e))
            return None

    @staticmethod
    def _diff(before: dict, after: dict) -> dict:
        """The keys that the repair policy changed, for the transcript/UI."""
        return {k: after[k] for k in after if before.get(k) != after.get(k)}

    def _record_outcome(self, role: str, family: str, params: dict, passed, metrics: dict,
                        verdict: str, artifact_id: str, attempt: int) -> None:
        """Persist a stage outcome to the learning loop (best-effort)."""
        try:
            self.memory.record(
                stage=role, family=family, params=params, passed=passed, metrics=metrics,
                verdict=verdict, artifact_id=artifact_id, attempt=attempt,
                run_id=(self.ledger.run_id if self.ledger else ""),
            )
        except Exception:  # noqa: BLE001 - memory must never break the run
            pass

    def _compensate(self, completed: "list[tuple[Step, Artifact]]") -> None:
        """Saga: run each completed step's compensation in reverse (best-effort)."""
        for step, art in reversed(completed):
            if step.compensate is None:
                continue
            self._emit("assistant", content=f"compensating {step.specialist_cls.role} ({art.id})")
            try:
                step.compensate(art)
            except Exception as e:  # noqa: BLE001 - compensation is best-effort
                self._emit("tool", tool="compensate", ok=False, output=str(e))
