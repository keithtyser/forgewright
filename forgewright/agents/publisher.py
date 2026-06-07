"""Publisher specialist — the only outward-facing stage. Artifact -> PublishedArtifact.

Publishing weights/datasets is irreversible, so it is HUMAN-GATED: the publish goes through
`PermissionPolicy.check` on the destructive ForgePublishTool, which (in the conversational
frontend) surfaces an approval_request at the single prompt. If denied, nothing is published
and a blocked artifact is recorded. The Director places this last in a recipe.
"""
from __future__ import annotations

from typing import Optional, Sequence

from forgewright.agents.base import Specialist
from forgewright.contracts import Artifact, Gate, PublishedArtifact
from forgewright.tools.base import ToolRegistry
from forgewright.tools.forge import ForgePublishTool, ForgeRunner

_PROMPT = """\
You are the Publisher specialist: publish a gated artifact to Hugging Face. This is
irreversible and requires human approval at the prompt. Only publish models/datasets that
passed their gates; include the evidence. Never bypass the approval.
"""


class Publisher(Specialist):
    role = "Publisher"
    accepts = ("model", "adapter", "dataset")
    produces = "published"
    description = "Human-gated publish of a passed artifact to Hugging Face -> PublishedArtifact."

    def __init__(self, *, runner: Optional[ForgeRunner] = None, host: Optional[str] = None, **kw) -> None:
        super().__init__(**kw)
        self.forge = runner or ForgeRunner()
        self.host = host
        self.publish_tool = ForgePublishTool(self.forge)

    def system_prompt(self) -> str:
        return _PROMPT

    def tools(self) -> ToolRegistry:
        return ToolRegistry([self.publish_tool])

    def run(self, inputs: Sequence[Artifact], goal: str = "", *, publish_args: str = "",
            release_class: str = "public_quantized_model") -> Artifact:
        self.validate_inputs(inputs)
        target = inputs[0]
        # Refuse to publish something that did not pass its gate.
        if target.gate is not None and not target.gate.passed:
            return self._blocked(target, "upstream artifact did not pass its gate")

        args = publish_args or (
            f"publish-model {target.meta.get('family','')} {target.meta.get('variant','')} "
            f"--release-class {release_class}"
        )
        # Human gate: this surfaces at the single prompt via PermissionPolicy.ask_fn.
        decision = self.permissions.check(self.publish_tool, {"args": args})
        if not decision.allowed:
            return self._blocked(target, f"publish not approved ({decision.reason})")

        self._emit("assistant", content=f"publishing {target.id}: {args}")
        res = self.publish_tool.run(args=args)
        ok = res.ok
        art = PublishedArtifact(
            uri=_hf_url(res.output) or args, produced_by=self.role, parents=[target.id],
            run_id=(self.ledger.run_id if self.ledger else ""),
            gate=Gate(passed=ok, verdict="PUBLISHED" if ok else "FAIL: publish errored"),
            meta={"release_class": release_class, "args": args},
        )
        self.registry.register(art)
        self._emit("tool", tool="forge_publish", ok=ok, output=res.output[:200])
        return art

    def _blocked(self, target: Artifact, reason: str) -> Artifact:
        art = PublishedArtifact(
            uri="", produced_by=self.role, parents=[target.id],
            gate=Gate(passed=False, verdict=f"BLOCKED: {reason}"), meta={"blocked": True, "reason": reason},
        )
        self.registry.register(art)
        self._emit("assistant", content=f"publish blocked: {reason}")
        return art


def _hf_url(text: str) -> str:
    import re

    m = re.search(r"https?://huggingface\.co/\S+", text or "")
    return m.group(0) if m else ""
