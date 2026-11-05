"""Failure-path checks. No GPU, SSH server, paid API, or upload is used."""
from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import pytest
import yaml

from forgewright.agents.data_curator import DataCurator, _valid_messages
from forgewright.agents.evaluator import Evaluator, _model_gate
from forgewright.agents.planner import _parse_roles, validate_chain
from forgewright.agents.publisher import Publisher
from forgewright.agents.quantizer import Quantizer
from forgewright.brain.codex_oauth import default_auth_path, parse_responses_sse
from forgewright.brain.provider import BrainError
from forgewright.contracts import AdapterArtifact, EvalArtifact, Gate, ModelArtifact
from forgewright.permissions import PermissionPolicy
from forgewright.registry import Registry
from forgewright.skills.data_integrity import verify_split
from forgewright.skills.eval_gate import gate_pass, render_eval_gate
from forgewright.skills.serving_opt import Candidate, ServingOptimizer, stop_owned_container
from forgewright.tools.base import ToolResult
from forgewright.tools.forge import ForgeRunner
from forgewright.tools.jobs import JobManager
from forgewright.trainers.rl import numeric_correctness_reward, render_grpo_trainer


class Jobs:
    def __init__(self, code=0):
        self.code, self.commands = code, []
    def launch(self, command, **kwargs):
        self.commands.append((command, kwargs))
        return {"id": "test"}
    def wait(self, job):
        return {"exit_code": self.code}


def setup_model(tmp_path):
    reg = Registry(tmp_path / "artifacts.jsonl")
    cfg = tmp_path / "configs" / "model_families" / "q.yaml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text("variants:\n  base:\n    local_dir: old-base\n")
    model = reg.register(ModelArtifact(uri="/models/trained", meta={"family": "q", "variant": "trained"},
                                       gate=Gate(True)))
    return reg, model, cfg


def test_quantizer_consumes_exact_input_and_requires_fresh_output(tmp_path, monkeypatch):
    reg, model, cfg = setup_model(tmp_path)
    jobs = Jobs()
    monkeypatch.setattr("forgewright.skills.hostexec.wrote_fresh_weights", lambda *a: False)
    art = Quantizer(runner=ForgeRunner(tmp_path), jobs=jobs, registry=reg).run([model], method="fp8")
    assert "quantize export q trained " in jobs.commands[0][0]
    assert yaml.safe_load(cfg.read_text())["variants"]["trained"]["local_dir"] == model.uri
    assert art.meta["source_uri"] == model.uri and not art.gate.passed
    assert art.config_hash


def test_quantizer_rejects_mismatched_variant_before_job(tmp_path):
    reg, model, _ = setup_model(tmp_path)
    jobs = Jobs()
    with pytest.raises(ValueError, match="does not match"):
        Quantizer(runner=ForgeRunner(tmp_path), jobs=jobs, registry=reg).run(
            [model], method="fp8", source_variant="base")
    assert jobs.commands == []


@pytest.mark.parametrize("score", [None, 0, float("nan"), float("inf"), -1, 1.1])
def test_model_gate_rejects_invalid_capability(score):
    assert not _model_gate({"capability": score}, {"capability": .8}).passed


def test_failed_adapter_job_cannot_reuse_old_pass(tmp_path):
    old = tmp_path / "runs" / "eval_gate" / "old"
    old.mkdir(parents=True)
    (old / "result.json").write_text('{"passed":true,"verdict":"PASS"}')
    reg = Registry(tmp_path / "a.jsonl")
    target = AdapterArtifact(uri="/adapter", meta={"base": "base", "run_name": "old"})
    art = Evaluator(runner=ForgeRunner(tmp_path), jobs=Jobs(1), registry=reg).run([target], holdout="h.jsonl")
    assert not art.gate.passed and art.uri != str(old / "result.json")


def test_scores_locator_never_falls_back_to_other_run(tmp_path):
    p = tmp_path / "results" / "unrelated"
    p.mkdir(parents=True)
    (p / "scores.csv").write_text("old scores")
    engine = ServingOptimizer(ForgeRunner(tmp_path), Jobs())
    assert engine._locate_scores_csv("eval failed") is None
    assert engine._locate_scores_csv("Output: results/unrelated", since=10**12) is None


@pytest.mark.parametrize("quality", [None, 0, .8])
def test_serving_winner_is_live_only_with_valid_evidence(tmp_path, monkeypatch, quality):
    engine = ServingOptimizer(ForgeRunner(tmp_path), Jobs())
    active = {"env": None}
    def launch(family, variant, env):
        active["env"] = dict(env)
        return {"ok": True}
    monkeypatch.setattr(engine, "launch", launch)
    monkeypatch.setattr(engine, "stop", lambda: active.update(env=None))
    monkeypatch.setattr(engine, "wait_ready", lambda: True)
    monkeypatch.setattr(engine, "bench_single", lambda *a: 20 if active["env"] else 10)
    monkeypatch.setattr(engine, "eval_quality", lambda *a: quality)
    _, best = engine.run("q", "v", candidates=[Candidate("baseline"), Candidate("faster", {"X": "1"})])
    if quality == .8:
        assert best.name == "faster" and active["env"] == {"X": "1"}
    else:
        assert best is None and active["env"] is None


def test_container_cleanup_checks_ownership_before_deletion(monkeypatch):
    calls = []
    def run(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=0, stdout='[{"Id":"other","Config":{"Labels":{}}}]')
    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(RuntimeError, match="ownership"):
        stop_owned_container("forgewright-" + "a" * 32)
    assert len(calls) == 1 and calls[0][1] == "inspect"


@pytest.mark.parametrize("receipt,execute,expected", [
    ({"dry_run": True}, True, False),
    ({"dry_run": False, "uploaded_at": "now", "uploaded_files": ["model"],
      "hub_url": "https://huggingface.co/user/model"}, True, True),
    ({"dry_run": False, "uploaded_at": "now", "uploaded_files": ["model"],
      "hub_url": "https://huggingface.co/user/model"}, False, False),
])
def test_publication_requires_execute_and_upload_receipt(tmp_path, monkeypatch, receipt, execute, expected):
    receipt = {**receipt, "family": "q", "variant": "trained"}
    reg, model, _ = setup_model(tmp_path)
    reg.register(EvalArtifact(parents=[model.id], gate=Gate(True)))
    publisher = Publisher(runner=ForgeRunner(tmp_path), registry=reg,
                          permissions=PermissionPolicy(auto_approve=True))
    monkeypatch.setattr(publisher.publish_tool, "run", lambda **kwargs: ToolResult(True, json.dumps(receipt)))
    art = publisher.run([model], publish_args="publish-model q trained" + (" --execute" if execute else ""))
    assert art.gate.passed is expected
    assert bool(art.uri) is expected


def test_publish_requires_current_artifact_evaluation(tmp_path):
    reg, model, _ = setup_model(tmp_path)
    reg.register(EvalArtifact(parents=["another-model"], gate=Gate(True)))
    art = Publisher(runner=ForgeRunner(tmp_path), registry=reg).run([model])
    assert not art.gate.passed and "exact artifact" in art.gate.verdict


def test_planner_rejects_ungated_publish_and_unknown_roles():
    assert not validate_chain(["Publisher"], "model")[0]
    assert not validate_chain(["Evaluator", "Quantizer", "Publisher"], "model")[0]
    assert _parse_roles('["Invented", "Publisher"]') == ["Invented", "Publisher"]


def test_curator_filters_custom_holdout_and_trainer_rechecks(tmp_path):
    (tmp_path / "train.jsonl").write_text('\n'.join(json.dumps({"messages": [
        {"role": "user", "content": prompt}, {"role": "assistant", "content": "1"}]})
        for prompt in ["held question", "new question"]))
    (tmp_path / "h.jsonl").write_text('{"prompt":" HELD   question ","answer":"1"}')
    reg = Registry(tmp_path / "a.jsonl")
    art = DataCurator(runner=ForgeRunner(tmp_path), registry=reg).run(
        [], seed_paths=["train.jsonl"], holdout="h.jsonl")
    assert art.gate.metrics["holdout_overlap"] == 1
    assert verify_split(tmp_path, art.uri, "h.jsonl")["rows"] == 1
    with pytest.raises(ValueError, match="overlaps"):
        verify_split(tmp_path, "train.jsonl", "h.jsonl")


@pytest.mark.parametrize("row", [None, [], {"messages": ["bad"]}, {"messages": [{"role": []}]}])
def test_malformed_curation_rows_do_not_crash(row):
    assert not _valid_messages(row)


def test_failed_remote_kill_does_not_claim_success(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGEWRIGHT_HOME", str(tmp_path))
    manager = JobManager()
    manager._save({"id": "job", "host": "host", "pid": 123, "status": "running"})
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=255))
    assert manager.kill("job") is False
    assert manager._read("job")["status"] == "running"


def test_remote_launch_applies_quoted_cwd(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGEWRIGHT_HOME", str(tmp_path))
    calls = []
    def run(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=0, stdout="123\n", stderr="")
    monkeypatch.setattr(subprocess, "run", run)
    JobManager().launch("pwd", host="u@host:2222", cwd="~/my project")
    assert 'cd "$HOME"/\'my project\'' in calls[0][-1]
    assert "2222" in calls[0] and "StrictHostKeyChecking=yes" in calls[0]


def test_failed_artifact_is_not_selected_as_latest(tmp_path):
    reg = Registry(tmp_path / "a.jsonl")
    good = reg.register(ModelArtifact(gate=Gate(True)))
    reg.register(ModelArtifact(gate=Gate(False)))
    assert reg.latest("model").id == good.id


def test_oauth_uses_codex_home_and_rejects_truncated_response(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    assert default_auth_path() == tmp_path / "auth.json"
    with pytest.raises(BrainError, match="before response.completed"):
        parse_responses_sse('event: response.output_text.delta\ndata: {"delta":"partial"}\n\n')


def test_numeric_fraction_scoring_and_empty_signal():
    assert numeric_correctness_reward("1/2", "3/2") == 0
    assert numeric_correctness_reward("1/2", "0.5") == 1
    assert numeric_correctness_reward("1e3", "1000") == 1
    assert not gate_pass(0, 0)
    compile(render_eval_gate(), "eval_gate.py", "exec")
    compile(render_grpo_trainer(), "train_grpo.py", "exec")


def test_recipe_receipts_prevent_duplicate_work_and_require_reconciliation(tmp_path):
    from forgewright.agents.base import Specialist
    from forgewright.agents.director import Director, Step
    from forgewright.ledger.ledger import Ledger
    from forgewright.tools.base import ToolRegistry
    calls = []
    class Stage(Specialist):
        role = "Stage"
        produces = "model"
        def system_prompt(self): return ""
        def tools(self): return ToolRegistry([])
        def run(self, inputs, goal):
            calls.append(goal)
            return ModelArtifact(uri="out", gate=Gate(True))
    director = Director(registry=Registry(tmp_path / "r.jsonl"), ledger=Ledger("run", tmp_path))
    first = director.run_recipe("goal", [Step(Stage)])
    second = director.run_recipe("goal", [Step(Stage)])
    assert first.ok and second.ok and calls == ["goal"]
    path = next((tmp_path / "recipes").glob("*.json"))
    state = json.loads(path.read_text())
    state["status"] = "active"
    path.write_text(json.dumps(state))
    blocked = director.run_recipe("goal", [Step(Stage)])
    assert not blocked.ok and blocked.failed_at == "recovery" and len(calls) == 1


def test_approval_response_must_match_current_request():
    from forgewright.frontend.bridge import StreamApprover
    events = []
    replies = [{"type": "approval_response", "request_id": "stale", "approved": True}, None]
    decision = StreamApprover(events.append, lambda: replies.pop(0))(SimpleNamespace(name="upload"), {})
    assert decision == "no" and events[0]["request_id"] != "stale"


def test_backend_reports_incomplete_turn():
    from forgewright.frontend.server import BackendServer
    from forgewright.loop import LoopResult
    events = []
    BackendServer(lambda *a: LoopResult(False, 2, "interrupted"), events.append).run_turn("go")
    assert events[-1] == {"type": "done", "ok": False, "reason": "interrupted"}


def test_eof_denies_multiple_pending_approvals():
    import io
    from forgewright.frontend.server import serve_stdio
    tool = SimpleNamespace(name="publish", risk="destructive")
    answers = []
    def turn(text, reporter, permissions):
        answers.extend(permissions.check(tool, {}).allowed for _ in range(2))
    serve_stdio(turn, instream=io.StringIO('{"type":"user_msg","text":"go"}\n'), outstream=io.StringIO())
    assert answers == [False, False]


def test_approval_all_is_scoped_to_exact_arguments():
    answers = iter(["all", "no"])
    policy = PermissionPolicy(ask_fn=lambda *a: next(answers))
    tool = SimpleNamespace(name="bash", risk="exec")
    assert policy.check(tool, {"command": "pwd"}).allowed
    assert policy.check(tool, {"command": "pwd"}).allowed
    assert not policy.check(tool, {"command": "different"}).allowed


def test_registry_rejects_mutation_and_reports_corrupt_line(tmp_path):
    reg = Registry(tmp_path / "r.jsonl")
    artifact = reg.register(ModelArtifact(uri="first"))
    artifact.uri = "different"
    with pytest.raises(ValueError, match="immutable"):
        reg.register(artifact)
    with reg.path.open("a") as stream:
        stream.write("invalid\n")
    with pytest.raises(ValueError, match=":2"):
        reg.all()


def test_redaction_removes_keys_and_known_values(monkeypatch):
    from forgewright.redaction import redact
    monkeypatch.setenv("OPENAI_API_KEY", "a-test-secret-value")
    result = redact({"api_key": "secret", "output": "key=a-test-secret-value"})
    assert result == {"api_key": "[REDACTED]", "output": "key=[REDACTED]"}


def test_unsupported_budget_is_rejected():
    from forgewright.config import Governor
    with pytest.raises(ValueError, match="not implemented"):
        Governor(max_cost_usd=50)


def test_oversized_recent_message_does_not_bypass_context_limit():
    from forgewright.context.manager import ContextManager
    context = ContextManager("system", max_input_tokens=10000)
    context.add_user("x" * 100000)
    with pytest.raises(ValueError, match="exceeds"):
        context.maybe_compact()


def test_full_recipe_does_not_edit_refusals_by_default():
    from forgewright.agents.recipes import full
    stages, _ = full(family="q", source="q", seed_paths=["d"], holdout="h")
    assert "Abliterator" not in [s.specialist_cls.role for s in stages]


def test_registration_keeps_runtime_flags_and_exact_quantized_path(tmp_path):
    from forgewright.skills.model_identity import register_model
    _, model, cfg = setup_model(tmp_path)
    cfg.write_text('variants:\n  trained:\n    quantization: fp8\n    runtime: {tensor_parallel: 2}\n')
    model.meta.update(role='quantized', method='fp8_modelopt')
    register_model(tmp_path, model)
    variant = yaml.safe_load(cfg.read_text())['variants']['trained']
    assert variant['local_dir'] == '/models/trained'
    assert variant['quantization'] == 'fp8' and variant['runtime']['tensor_parallel'] == 2


def test_eval_output_accepts_rich_panel_but_rejects_stale_files(tmp_path):
    path = tmp_path / 'result directory'
    path.mkdir()
    scores = path / 'scores.csv'
    scores.write_text('bucket,metric,value\ncapability,pass_rate,0.8\n')
    engine = ServingOptimizer(ForgeRunner(tmp_path), Jobs())
    output = f'│ Output:  {path}  │\n'
    assert engine._locate_scores_csv(output) == scores
    assert engine._locate_scores_csv(output, since=scores.stat().st_mtime + 1) is None


def test_cancel_refuses_reused_process_id(tmp_path, monkeypatch):
    manager = JobManager()
    manager._save({'id': 'job-identity', 'pid': 123, 'host': 'local', 'status': 'running',
                   'process_identity': 'original'})
    monkeypatch.setattr('forgewright.tools.jobs._process_identity', lambda *a: 'replacement')
    monkeypatch.setattr(subprocess, 'run', lambda *a, **kw: pytest.fail('must not signal a reused PID'))
    assert not manager.kill('job-identity')
    assert manager._read('job-identity')['status'] == 'running'


def test_local_process_identity_is_stable():
    import os
    from forgewright.tools.jobs import _process_identity
    identity = _process_identity(os.getpid())
    assert identity and identity == _process_identity(os.getpid())


def test_cancel_signal_alone_does_not_mean_killed(monkeypatch):
    manager = JobManager()
    manager._save({'id': 'job-signal', 'pid': 123, 'host': 'local', 'status': 'running',
                   'process_identity': 'same'})
    monkeypatch.setattr('forgewright.tools.jobs._process_identity', lambda *a: 'same')
    monkeypatch.setattr('forgewright.tools.jobs.time.sleep', lambda *a: None)
    monkeypatch.setattr('forgewright.tools.jobs.os.killpg', lambda *a: None, raising=False)
    monkeypatch.setattr(subprocess, 'run', lambda *a, **kw: SimpleNamespace(returncode=0))
    assert not manager.kill('job-signal')
    assert manager._read('job-signal')['status'] == 'stopping'


def test_director_preserves_shared_interrupt_and_job_manager(tmp_path):
    import threading
    from forgewright.agents.base import Specialist
    from forgewright.agents.director import Director, Step
    from forgewright.tools.base import ToolRegistry
    manager = JobManager()
    event = threading.Event()
    manager.interrupt = event.is_set
    class Stage(Specialist):
        role = 'Stage'
        produces = 'model'
        def __init__(self, *, jobs, **kw):
            super().__init__(**kw)
            assert jobs is manager and jobs.interrupt.__self__ is event
        def system_prompt(self): return ''
        def tools(self): return ToolRegistry([])
        def run(self, inputs, goal): return ModelArtifact(uri='out', gate=Gate(True))
    result = Director(registry=Registry(tmp_path / 'r.jsonl')).run_recipe(
        'go', [Step(Stage, init_kwargs={'jobs': manager})])
    assert result.ok
