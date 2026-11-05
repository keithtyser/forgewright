"""Publisher specialist — the only outward-facing stage. Artifact -> PublishedArtifact.

Publishing weights/datasets is irreversible, so it is HUMAN-GATED: the publish goes through
`PermissionPolicy.check` on the destructive ForgePublishTool, which (in the conversational
frontend) surfaces an approval_request at the single prompt. If denied, nothing is published
and a blocked artifact is recorded. The Director places this last in a recipe.
"""
from __future__ import annotations

from typing import Optional, Sequence
import json
import shlex

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
        if len(inputs) == 1 and inputs[0].gate is not None and not inputs[0].gate.passed:
            return self._blocked(inputs[0], "upstream artifact did not pass its gate")
        self.validate_inputs(inputs)
        target = inputs[0]
        if target.kind != "model":
            return self._blocked(target, "only standalone model publication is supported; merge adapters first")
        if target.gate is None or not target.gate.passed:
            return self._blocked(target, "upstream artifact did not pass its gate")
        evaluations = [a for a in self.registry.all() if a.kind == "eval" and a.parents == [target.id]]
        evidence = evaluations[-1] if evaluations else None
        if evidence is None or evidence.gate is None or not evidence.gate.passed:
            return self._blocked(target, "a passing evaluation for this exact artifact is required")
        if evidence.meta.get("target_uri") not in (None, target.uri):
            return self._blocked(target, "evaluation names a different checkpoint URI")
        try:
            self.registry.lineage(target.id)
        except ValueError as error:
            return self._blocked(target, str(error))
        family, variant = target.meta.get("family"), target.meta.get("variant")
        if not family or not variant:
            return self._blocked(target, "model family and variant are required")

        args = publish_args or (
            f"publish-model {shlex.quote(family)} {shlex.quote(variant)} "
            f"--release-class {shlex.quote(release_class)} --execute"
        )
        tokens = shlex.split(args)
        if tokens[:3] != ["publish-model", family, variant]:
            return self._blocked(target, "publish command does not name the evaluated model")
        if "--json" not in tokens:
            tokens.append("--json")
        args = shlex.join(tokens)
        execute = "--execute" in tokens
        # Human gate: this surfaces at the single prompt via PermissionPolicy.ask_fn.
        decision = self.permissions.check(self.publish_tool, {"args": args})
        if not decision.allowed:
            return self._blocked(target, f"publish not approved ({decision.reason})")

        self._emit("assistant", content=f"publishing {target.id}: {args}")
        res = self.publish_tool.run(args=args)
        try:
            receipt = json.loads(res.output)
        except (ValueError, TypeError):
            receipt = {}
        if not isinstance(receipt, dict):
            receipt = {}
        ok = bool(res.ok and execute and receipt.get("dry_run") is False
                  and receipt.get("family") == family and receipt.get("variant") == variant
                  and receipt.get("uploaded_at") and receipt.get("uploaded_files")
                  and str(receipt.get("hub_url", "")).startswith("https://huggingface.co/"))
        art = PublishedArtifact(
            uri=receipt.get("hub_url", "") if ok else "", produced_by=self.role, parents=[target.id],
            run_id=(self.ledger.run_id if self.ledger else ""),
            gate=Gate(passed=ok, verdict="PUBLISHED" if ok else "UNVERIFIED: no completed upload receipt"),
            meta={"release_class": release_class, "args": args, "evaluation_id": evidence.id,
                  "receipt": receipt, "executed": execute},
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
