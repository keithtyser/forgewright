"""Tests for the Director orchestration + the frontend event bridge."""
from __future__ import annotations

from forgewright.agents.base import Specialist
from forgewright.agents.director import Director, Step
from forgewright.contracts import AdapterArtifact, Artifact, DatasetArtifact, EvalArtifact, Gate
from forgewright.frontend.bridge import StreamApprover, event_reporter, json_line_emitter
from forgewright.registry import Registry
from forgewright.tools.base import ToolRegistry


# --- fake specialists that register typed artifacts (no GPU) ----------------------

class FakeSFT(Specialist):
    role = "SFTTrainer"
    accepts = ("dataset",)
    produces = "adapter"

    def system_prompt(self): return "sft"
    def tools(self): return ToolRegistry([])

    def run(self, inputs, goal="", **kw):
        ds = inputs[0]
        self._emit("assistant", content="training")
        art = AdapterArtifact(uri="~/models/x-sft", produced_by=self.role, parents=[ds.id],
                              gate=Gate(passed=True, verdict="TRAINED"), meta={"base": "Qwen/Qwen3.5-0.8B"})
        self.registry.register(art)
        return art


class FakeEval(Specialist):
    role = "Evaluator"
    accepts = ("adapter", "model")
    produces = "eval"

    def __init__(self, *, verdict_pass=True, **kw):
        super().__init__(**kw)
        self._pass = verdict_pass

    def system_prompt(self): return "eval"
    def tools(self): return ToolRegistry([])

    def run(self, inputs, goal="", **kw):
        a = inputs[0]
        self._emit("assistant", content="evaluating")
        art = EvalArtifact(uri="scores.json", produced_by=self.role, parents=[a.id],
                           gate=Gate(passed=self._pass, verdict="PASS" if self._pass else "REGRESSION"))
        self.registry.register(art)
        return art


def test_director_chains_and_records_lineage(tmp_path):
    reg = Registry(tmp_path / "artifacts.jsonl")
    ds = reg.register(DatasetArtifact(uri="d.jsonl", meta={"family": "q08"}))
    events = []
    director = Director(registry=reg, reporter=lambda k, d: events.append((k, d)))
    recipe = [Step(FakeSFT), Step(FakeEval)]
    res = director.run_recipe("uplift it", recipe, seed_inputs=[ds])

    assert res.ok and res.final.kind == "eval"
    # full provenance chain dataset -> adapter -> eval
    chain = reg.lineage(res.final.id)
    assert [c.kind for c in chain] == ["eval", "adapter", "dataset"]
    # one transcript: events from Director AND both specialists, each role-tagged
    roles = {d.get("role") for _, d in events}
    assert {"Director", "SFTTrainer", "Evaluator"} <= roles


def test_director_emits_structured_pipeline_events(tmp_path):
    """The Director feeds the UI a live pipeline: a `pipeline` map up front, `stage`
    transitions (active -> done), and an `artifact` per produced output."""
    reg = Registry(tmp_path / "artifacts.jsonl")
    ds = reg.register(DatasetArtifact(uri="d.jsonl"))
    events = []
    director = Director(registry=reg, reporter=lambda k, d: events.append((k, d)))
    director.run_recipe("uplift", [Step(FakeSFT), Step(FakeEval)], seed_inputs=[ds])

    kinds = [k for k, _ in events]
    assert "pipeline" in kinds and kinds.count("stage") == 4 and kinds.count("artifact") == 2

    pipe = next(d for k, d in events if k == "pipeline")
    assert pipe["stages"] == ["SFTTrainer", "Evaluator"]

    stages = [d for k, d in events if k == "stage"]
    assert stages[0] == {"role": "Director", "name": "SFTTrainer", "index": 0, "total": 2, "state": "active"}
    assert any(s["name"] == "SFTTrainer" and s["state"] == "done" for s in stages)

    # artifacts are attributed to their producer (for per-role coloring), with lineage
    arts = [d for k, d in events if k == "artifact"]
    assert arts[0]["role"] == "SFTTrainer" and arts[0]["kind"] == "adapter"
    assert arts[1]["kind"] == "eval" and arts[1]["parents"]


def test_director_gate_passes_evaluated_artifact_through(tmp_path):
    """After an Evaluator (a gate), the next stage should receive the evaluated adapter/model,
    not the eval report -- so e.g. Publisher can act on what passed."""
    seen = {}

    class _Recorder(Specialist):
        role = "Publisher"; accepts = ("adapter", "model"); produces = "published"
        def system_prompt(self): return ""
        def tools(self): return ToolRegistry([])
        def run(self, inputs, goal="", **kw):
            seen["got_kind"] = inputs[0].kind
            art = Artifact(kind="published", parents=[inputs[0].id])
            self.registry.register(art)
            return art

    reg = Registry(tmp_path / "artifacts.jsonl")
    ds = reg.register(DatasetArtifact(uri="d.jsonl"))
    res = Director(registry=reg).run_recipe(
        "x", [Step(FakeSFT), Step(FakeEval), Step(_Recorder)], seed_inputs=[ds])
    assert res.ok
    assert seen["got_kind"] == "adapter"   # the evaluated adapter, not "eval"


def test_director_marks_failed_stage(tmp_path):
    reg = Registry(tmp_path / "artifacts.jsonl")
    ds = reg.register(DatasetArtifact(uri="d.jsonl"))
    events = []
    director = Director(registry=reg, reporter=lambda k, d: events.append((k, d)))
    director.run_recipe("x", [Step(FakeSFT), Step(FakeEval, init_kwargs={"verdict_pass": False})],
                        seed_inputs=[ds])
    stages = [d for k, d in events if k == "stage"]
    assert any(s["name"] == "Evaluator" and s["state"] == "failed" for s in stages)


def test_director_halts_on_global_gate_failure(tmp_path):
    reg = Registry(tmp_path / "artifacts.jsonl")
    ds = reg.register(DatasetArtifact(uri="d.jsonl"))
    director = Director(registry=reg)
    recipe = [Step(FakeSFT), Step(FakeEval, init_kwargs={"verdict_pass": False})]
    res = director.run_recipe("x", recipe, seed_inputs=[ds])
    assert not res.ok and res.failed_at == "Evaluator" and res.reason == "REGRESSION"
    assert len(res.artifacts) == 2  # produced both, halted after the failing gate


def test_director_surfaces_specialist_exception(tmp_path):
    class Boom(Specialist):
        role = "Boom"; accepts = ("dataset",); produces = "adapter"
        def system_prompt(self): return ""
        def tools(self): return ToolRegistry([])
        def run(self, inputs, goal="", **kw): raise RuntimeError("kaboom")

    reg = Registry(tmp_path / "artifacts.jsonl")
    ds = reg.register(DatasetArtifact(uri="d.jsonl"))
    res = Director(registry=reg).run_recipe("x", [Step(Boom)], seed_inputs=[ds])
    assert not res.ok and res.failed_at == "Boom" and "kaboom" in res.reason


def test_event_reporter_serializes_to_json_lines():
    lines = []
    emit = json_line_emitter(lines.append)
    report = event_reporter(emit)
    report("tool", {"tool": "forge", "role": "SFTTrainer", "ok": True})
    import json
    obj = json.loads(lines[0])
    assert obj == {"type": "tool", "tool": "forge", "role": "SFTTrainer", "ok": True}


def test_stream_approver_round_trips():
    sent = []
    inbox = [{"type": "user_msg", "text": "ignore me"}, {"type": "approval_response", "approved": True}]
    approver = StreamApprover(emit=sent.append, read_response=lambda: inbox.pop(0))

    class _Tool:
        name = "forge_publish"; risk = "destructive"

    assert approver(_Tool(), {"args": "publish-model"}) == "yes"   # {approved:true} -> "yes"
    assert sent[0]["type"] == "approval_request" and sent[0]["tool"] == "forge_publish"
    # explicit decision passes through (e.g. "all" / "yolo")
    inbox2 = [{"type": "approval_response", "decision": "yolo"}]
    assert StreamApprover(emit=sent.append, read_response=lambda: inbox2.pop(0))(_Tool(), {}) == "yolo"
    # stream closed -> deny (fail safe)
    assert StreamApprover(emit=sent.append, read_response=lambda: None)(_Tool(), {}) == "no"
