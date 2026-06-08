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
    """
    specialist_cls: type[Specialist]
    init_kwargs: dict = field(default_factory=dict)   # constructor kwargs (host, tolerance, runner, jobs)
    run_kwargs: dict = field(default_factory=dict)     # run() kwargs (holdout, max_steps)
    compensate: Optional[Callable[[Artifact], None]] = None


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
    ) -> None:
        self.registry = registry or Registry()
        self.reporter = label_reporter(reporter, self.role)
        self._base_reporter = reporter
        self.permissions = permissions or PermissionPolicy()
        self.ledger = ledger
        self.brain = brain
        self.host = host

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
            self._emit("stage", name=role, index=i, total=total, state="active")
            spec = step.specialist_cls(
                registry=self.registry,
                reporter=self._base_reporter,   # specialists self-label; one transcript
                permissions=self.permissions,   # all approvals bubble to the one prompt
                ledger=self.ledger,
                brain=self.brain,
                host=self.host,
                **step.init_kwargs,
            )
            try:
                art = spec.run(current, goal, **step.run_kwargs)
            except Exception as e:  # noqa: BLE001 - surface the failing stage, don't crash the chain
                self._emit("stage", name=role, index=i, total=total, state="failed")
                self._emit("tool", tool=role, ok=False, output=f"error: {e}")
                self._compensate(completed)
                return DirectorResult(False, produced, failed_at=role, reason=str(e))
            produced.append(art)
            gate_metrics = art.gate.metrics if art.gate is not None else {}
            gate_passed = art.gate.passed if art.gate is not None else None
            self._emit(
                "artifact", role=art.produced_by or role, kind=art.kind, id=art.id,
                parents=list(art.parents), metrics=gate_metrics, passed=gate_passed,
            )
            if art.gate is not None and not art.gate.passed:
                self._emit("stage", name=role, index=i, total=total, state="failed")
                self._emit("assistant", content=f"GLOBAL GATE halt at {role}: {art.gate.verdict}")
                self._compensate(completed)   # roll back the steps that DID complete
                return DirectorResult(False, produced, failed_at=role, reason=art.gate.verdict)
            self._emit("stage", name=role, index=i, total=total, state="done")
            completed.append((step, art))
            current = [art]

        final = produced[-1] if produced else None
        self._emit("assistant", content=f"recipe complete: {len(produced)} artifacts, "
                   f"lineage {[a.id for a in produced]}")
        return DirectorResult(True, produced, final=final)

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
