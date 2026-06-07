"""The `Specialist` base + the role-labelled reporter for the one-transcript UX.

A Specialist is a thin wrapper over the shared `Agent` runtime: it declares the artifact
kinds it `accepts` and the kind it `produces`, supplies a focused system prompt + a tool
subset, and implements `run(inputs, goal, registry) -> Artifact`. Subclasses live in
`agents/<role>.py` and reuse the proven `skills/*` for the actual work.

`label_reporter` tags a base reporter callback with the specialist's role so that, when the
Director spawns specialists, all of their activity streams into the *one* CLI transcript
attributed to the right agent (the swarm stays invisible as a management surface).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, Optional, Sequence

from forgewright.contracts import Artifact
from forgewright.ledger.ledger import Ledger
from forgewright.permissions import PermissionPolicy
from forgewright.registry import Registry
from forgewright.tools.base import ToolRegistry

if TYPE_CHECKING:  # the LLM stack (Brain -> config -> pydantic/litellm) is only needed for
    from forgewright.brain.provider import Brain  # the agentic loop, not the deterministic run()

Reporter = Callable[[str, dict], object]


def label_reporter(base: Optional[Reporter], role: str) -> Optional[Reporter]:
    """Wrap a reporter so every event it emits is tagged with the specialist `role`,
    keeping nested specialist activity attributable in the single transcript."""
    if base is None:
        return None

    def report(kind: str, data: dict) -> object:
        return base(kind, {**data, "role": data.get("role", role)})

    return report


class Specialist(ABC):
    """One post-training stage as a focused agent. Subclasses set role/accepts/produces,
    provide a prompt + tools, and implement `run`."""

    role: str = "specialist"
    accepts: tuple[str, ...] = ()      # artifact kinds this consumes ("" = a fresh goal)
    produces: str = ""                 # artifact kind this emits
    description: str = ""

    def __init__(
        self,
        *,
        brain: "Optional[Brain]" = None,
        registry: Optional[Registry] = None,
        permissions: Optional[PermissionPolicy] = None,
        reporter: Optional[Reporter] = None,
        ledger: Optional[Ledger] = None,
        host: Optional[str] = None,
        max_steps: int = 60,
    ) -> None:
        self.brain = brain
        self.registry = registry or Registry()
        self.permissions = permissions or PermissionPolicy()
        self.reporter = label_reporter(reporter, self.role)
        self.ledger = ledger
        self.host = host  # None = run on this box; else an ssh target the Director sets
        self.max_steps = max_steps

    # --- contract --------------------------------------------------------------

    @abstractmethod
    def system_prompt(self) -> str:
        """The specialist's focused prompt (its runbook + its contract)."""

    @abstractmethod
    def tools(self) -> ToolRegistry:
        """The small tool subset this specialist is allowed to use."""

    @abstractmethod
    def run(self, inputs: Sequence[Artifact], goal: str) -> Artifact:
        """Do the stage's work, register the produced artifact, and return it."""

    # --- helpers ---------------------------------------------------------------

    def validate_inputs(self, inputs: Sequence[Artifact]) -> None:
        """Reject inputs whose kinds this specialist does not accept (fail fast at the
        handoff boundary rather than deep inside a GPU job)."""
        if not self.accepts:
            return
        for art in inputs:
            if art.kind not in self.accepts:
                raise ValueError(
                    f"{self.role} accepts {self.accepts}, got artifact kind '{art.kind}' ({art.id})"
                )

    def build_agent(self):
        """Construct the shared Agent loop wearing this specialist's prompt + tools.
        Imported lazily so the contracts/registry layer has no hard agent-loop dependency."""
        from forgewright.context.manager import ContextManager
        from forgewright.loop import Agent

        if self.brain is None:
            raise ValueError(f"{self.role} needs a Brain to run its agent loop")
        return Agent(
            brain=self.brain,
            tools=self.tools(),
            permissions=self.permissions,
            ledger=self.ledger,
            context=ContextManager(self.system_prompt()),
            max_steps=self.max_steps,
            reporter=self.reporter,
        )

    def _emit(self, kind: str, **data) -> None:
        if self.reporter:
            try:
                self.reporter(kind, {**data, "role": self.role})
            except Exception:  # noqa: BLE001 - display must never break a stage
                pass
