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
        """Execute the chain: each step consumes the previous step's output (or the seed for
        the first), gated globally. Returns the produced artifacts (lineage) + final."""
        current: list[Artifact] = list(seed_inputs or [])
        produced: list[Artifact] = []
        completed: list[tuple[Step, Artifact]] = []   # for saga compensation
        stages = [s.specialist_cls.role for s in recipe]
        total = len(recipe)
        # the full pipeline up front, so the UI can render a live progress map of the swarm.
        self._emit("pipeline", stages=stages, goal=goal)

        for i, step in enumerate(recipe):
            role = step.specialist_cls.role
            art, ok, reason = self._run_step_with_repair(step, current, goal, i, total)
            if art is not None:
                produced.append(art)
            if not ok:
                self._emit("stage", name=role, index=i, total=total, state="failed")
                self._emit("assistant", content=f"GLOBAL GATE halt at {role}: {reason}")
                self._compensate(completed)   # roll back the steps that DID complete
                return DirectorResult(False, produced, failed_at=role, reason=reason)
            self._emit("stage", name=role, index=i, total=total, state="done")
            completed.append((step, art))
            # A gate (Evaluator -> eval report) passes the evaluated artifact through, so a later
            # stage (e.g. Publisher) receives the model/adapter that passed, not the report itself.
            if art.kind != "eval":
                current = [art]

        final = produced[-1] if produced else None
        self._emit("assistant", content=f"recipe complete: {len(produced)} artifacts, "
                   f"lineage {[a.id for a in produced]}")
        return DirectorResult(True, produced, final=final)

    def _run_step_with_repair(self, step: Step, current, goal: str, i: int, total: int):
        """Run one stage with the generate -> verify -> repair loop. Returns (artifact, ok, reason).
        On a gate failure (or a raised stage) it re-runs up to step.max_attempts, calling the repair
        policy to adjust run_kwargs between tries and feeding the failure reason back into the goal.
        The outcome of every attempt is recorded to memory for the cross-run learning loop."""
        from forgewright.contracts import Gate  # local import: contracts has no director dependency

        role = step.specialist_cls.role
        attempts = max(1, step.max_attempts)
        repair = step.repair or (policy_for(role) if attempts > 1 else None)
        run_kwargs = dict(step.run_kwargs)
        family = run_kwargs.get("family") or (current[0].meta.get("family") if current else "") or ""
        last_real: Optional[Artifact] = None
        last_reason = ""

        for attempt in range(1, attempts + 1):
            state = "active" if attempt == 1 else "retry"
            self._emit("stage", name=role, index=i, total=total, state=state, attempt=attempt)
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
                f"{goal}\n[repair attempt {attempt}/{attempts}: the previous attempt failed -> "
                f"{last_reason}. Adjusted parameters: {run_kwargs}.]")
            raised = False
            try:
                art = spec.run(current, eff_goal, **run_kwargs)
            except Exception as e:  # noqa: BLE001 - a raised stage becomes a failed-gate artifact
                raised = True
                last_reason = str(e)
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
                last_real = art
                self._emit(
                    "artifact", role=art.produced_by or role, kind=art.kind, id=art.id,
                    parents=list(art.parents), metrics=gate_metrics, passed=gate_passed,
                )
            self._record_outcome(role, family, run_kwargs, gate_passed, gate_metrics, verdict,
                                 ("" if raised else art.id), attempt)

            if art.gate is None or art.gate.passed:
                return art, True, ""   # ungated stage or a clean pass

            last_reason = verdict or last_reason
            if attempt < attempts and repair is not None:
                new_kwargs = self._invoke_repair(repair, attempt + 1, art, run_kwargs)
                if new_kwargs is None:
                    self._emit("assistant", content=f"{role} repair gave up after attempt {attempt}")
                    break
                self._emit("repair", name=role, attempt=attempt + 1, reason=last_reason,
                           changes=self._diff(run_kwargs, new_kwargs))
                run_kwargs = new_kwargs

        return last_real, False, last_reason

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
