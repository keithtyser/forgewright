"""Director — the orchestrator. Plans a recipe (DAG, currently a chain) over specialists,
dispatches them, passes each produced artifact to the next, and enforces the GLOBAL gate.

The user talks only to the Director; it spawns specialists on the backend and threads its
own (role-labelled) reporter into each so all their activity streams into the one transcript.
If a specialist's gate fails, the Director halts (saga-compensation hook in Slice D) — a
regression never silently flows downstream.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

from forgewright.agents.base import Specialist, label_reporter
from forgewright.contracts import Artifact
from forgewright.ledger.ledger import Ledger
from forgewright.permissions import PermissionPolicy
from forgewright.registry import Registry

Reporter = "callable"


@dataclass
class Step:
    """One stage of a recipe: which specialist + how to build/run it."""
    specialist_cls: type[Specialist]
    init_kwargs: dict = field(default_factory=dict)   # constructor kwargs (host, tolerance, runner, jobs)
    run_kwargs: dict = field(default_factory=dict)     # run() kwargs (holdout, max_steps)


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

    def _emit(self, kind: str, **data) -> None:
        if self.reporter:
            try:
                self.reporter(kind, {**data, "role": self.role})
            except Exception:  # noqa: BLE001
                pass

    def run_recipe(
        self, goal: str, recipe: Sequence[Step], seed_inputs: Optional[Sequence[Artifact]] = None
    ) -> DirectorResult:
        """Execute the chain: each step consumes the previous step's output (or the seed for
        the first), gated globally. Returns the produced artifacts (lineage) + final."""
        current: list[Artifact] = list(seed_inputs or [])
        produced: list[Artifact] = []
        self._emit("assistant", content=f"plan: {' -> '.join(s.specialist_cls.role for s in recipe)}")

        for step in recipe:
            role = step.specialist_cls.role
            self._emit("assistant", content=f"dispatching {role}")
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
                self._emit("tool", tool=role, ok=False, output=f"error: {e}")
                return DirectorResult(False, produced, failed_at=role, reason=str(e))
            produced.append(art)
            if art.gate is not None and not art.gate.passed:
                self._emit("assistant",
                           content=f"GLOBAL GATE halt at {role}: {art.gate.verdict}")
                return DirectorResult(False, produced, failed_at=role, reason=art.gate.verdict)
            current = [art]

        final = produced[-1] if produced else None
        self._emit("assistant", content=f"recipe complete: {len(produced)} artifacts, "
                   f"lineage {[a.id for a in produced]}")
        return DirectorResult(True, produced, final=final)
