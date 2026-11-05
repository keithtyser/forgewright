"""Regression checks without model calls, uploads, SSH hosts or GPU work."""
from __future__ import annotations

import json
import shlex
import subprocess
import sys
import threading
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from types import SimpleNamespace

import pytest

from forgewright.agents.inputs import seed_artifact
from forgewright.agents.publisher import Publisher
from forgewright.brain.provider import AssistantTurn, Brain, BrainCancelled, BrainError, ToolCall, retry_delay
from forgewright.config import ProviderConfig
from forgewright.contracts import EvalArtifact, Gate, ModelArtifact
from forgewright.loop import Agent
from forgewright.permissions import PermissionPolicy
from forgewright.registry import Registry
from forgewright.tools.base import Tool, ToolRegistry, ToolResult
from forgewright.tools.forge import ForgeRunner
from forgewright.tools.sysconfig import EnvConfigTool


class HttpFailure(Exception):
    def __init__(self, status, headers=None):
        super().__init__(f"HTTP {status}")
        self.status_code = status
        self.response = SimpleNamespace(status_code=status, headers=headers or {})


def brain(**kwargs):
    return Brain(ProviderConfig(name="test", kind="vllm", model="test"), **kwargs)


@pytest.mark.parametrize("status", [400, 401, 402, 403, 404, 422])
def test_permanent_provider_errors_are_not_retried(monkeypatch, status):
    calls = []
    def completion(**kwargs):
        calls.append(kwargs)
        raise HttpFailure(status)
    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(completion=completion))
    monkeypatch.setattr("forgewright.brain.provider.time.sleep", lambda _: pytest.fail("unexpected retry"))
    with pytest.raises(BrainError, match="1 attempt"):
        brain().chat([])
    assert len(calls) == 1
    assert calls[0]["num_retries"] == calls[0]["max_retries"] == 0


def test_rate_limit_waits_then_recovers_without_nested_retries(monkeypatch):
    calls, sleeps = [], []
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])
    def completion(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise HttpFailure(429, {"retry-after": "0.25"})
        return response
    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(completion=completion))
    monkeypatch.setattr("forgewright.brain.provider.time.sleep", sleeps.append)
    assert brain().chat([]).content == "ok"
    assert len(calls) == 2 and sum(sleeps) == pytest.approx(.25)


def test_last_failed_attempt_does_not_sleep(monkeypatch):
    calls, sleeps = [], []
    def completion(**kwargs):
        calls.append(kwargs)
        raise HttpFailure(503, {"retry-after": "0.1"})
    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(completion=completion))
    monkeypatch.setattr("forgewright.brain.provider.time.sleep", sleeps.append)
    with pytest.raises(BrainError, match="3 attempt"):
        brain().chat([])
    assert len(calls) == 3 and sum(sleeps) == pytest.approx(.2)


def test_retry_after_date_and_excessive_delay():
    date = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=30), usegmt=True)
    assert 28 <= retry_delay(HttpFailure(429, {"Retry-After": date}), 0) <= 30
    assert retry_delay(HttpFailure(429, {"retry-after": "3600"}), 0) is None
    assert retry_delay(ValueError("bad config"), 0) is None
    assert retry_delay(ConnectionError("connection lost"), 0) is not None


def test_interrupt_during_retry_prevents_another_request(monkeypatch):
    event, calls = threading.Event(), []
    def completion(**kwargs):
        calls.append(kwargs)
        raise HttpFailure(429, {"retry-after": "1"})
    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(completion=completion))
    monkeypatch.setattr("forgewright.brain.provider.time.sleep", lambda _: event.set())
    with pytest.raises(BrainCancelled):
        brain(interrupt=event.is_set).chat([])
    assert len(calls) == 1


def test_interrupt_after_provider_return_prevents_tool_execution(monkeypatch):
    event = threading.Event()
    calls = []
    class Action(Tool):
        name = "action"
        risk = "read"
        def run(self, **kwargs):
            calls.append(kwargs)
            return ToolResult(True, "done")
    class Provider:
        def chat(self, *args, **kwargs):
            event.set()
            return AssistantTurn("", [ToolCall("a", "action", {})])
    result = Agent(Provider(), ToolRegistry([Action()]), interrupt=event.is_set).run("go")
    assert not result.done and result.final == "interrupted" and calls == []


def test_interrupt_during_approval_prevents_tool_execution():
    event, calls = threading.Event(), []
    class Action(Tool):
        name, risk = "action", "exec"
        parameters = {"type": "object"}
        def run(self, **kwargs):
            calls.append(kwargs)
            return ToolResult(True, "done")
    def approve(*_):
        event.set()
        return "yes"
    agent = Agent(SimpleNamespace(), ToolRegistry([Action()]), interrupt=event.is_set,
                  permissions=PermissionPolicy(ask_fn=approve))
    assert not agent._dispatch(ToolCall("a", "action", {}), []).ok
    assert calls == []


def test_registry_scope_rejects_global_and_ambiguous_selection(tmp_path):
    registry = Registry(tmp_path / "a.jsonl")
    old = registry.register(ModelArtifact(uri="old", meta={"family": "q"}, run_id="one"))
    registry.register(ModelArtifact(uri="unrelated", meta={"family": "other"}, run_id="two"))
    with pytest.raises(ValueError, match="global latest"):
        seed_artifact("model", {}, registry)
    assert seed_artifact("model", {"family": "q"}, registry).id == old.id
    registry.register(ModelArtifact(uri="new", meta={"family": "q"}, run_id="two"))
    with pytest.raises(ValueError, match="found 2"):
        seed_artifact("model", {"family": "q"}, registry)
    assert seed_artifact("model", {"model_artifact_id": old.id}, registry).id == old.id
    assert seed_artifact("model", {"artifact_run_id": "one"}, registry).id == old.id
    with pytest.raises(ValueError, match="not both"):
        seed_artifact("model", {"model_uri": "x", "model_artifact_id": old.id}, registry)


def test_failed_evaluation_excludes_producer_from_lookup(tmp_path):
    registry = Registry(tmp_path / "a.jsonl")
    model = registry.register(ModelArtifact(uri="m", gate=Gate(True)))
    registry.register(EvalArtifact(parents=[model.id], gate=Gate(False)))
    assert registry.latest("model") is None
    with pytest.raises(ValueError, match="found 0"):
        registry.select("model", artifact_id=model.id)
    registry.register(EvalArtifact(parents=[model.id], gate=Gate(True)))
    assert registry.select("model", artifact_id=model.id).id == model.id


def test_lineage_rejects_missing_parent_and_cycle(tmp_path):
    registry = Registry(tmp_path / "a.jsonl")
    a = registry.register(ModelArtifact(id="a", parents=["b"]))
    with pytest.raises(ValueError, match="missing b"):
        registry.lineage(a.id)
    registry.register(ModelArtifact(id="b", parents=["a"]))
    with pytest.raises(ValueError, match="cycle"):
        registry.lineage(a.id)
    assert len(registry.lineage(a.id, strict=False)) == 2


def test_publication_requires_complete_lineage(tmp_path):
    registry = Registry(tmp_path / "a.jsonl")
    model = registry.register(ModelArtifact(uri="m", parents=["missing"], gate=Gate(True)))
    registry.register(EvalArtifact(parents=[model.id], gate=Gate(True)))
    result = Publisher(registry=registry).run([model])
    assert not result.gate.passed and "missing" in result.gate.verdict


@pytest.mark.parametrize("row", [[], {"id": "x", "kind": "model", "parents": "bad"},
                                  {"id": "x", "kind": "model", "gate": {"passed": "false"}}])
def test_registry_rejects_invalid_record_shapes(tmp_path, row):
    path = tmp_path / "a.jsonl"
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match=":1"):
        Registry(path).all()


def test_registry_rejects_conflicting_old_records(tmp_path):
    registry = Registry(tmp_path / "a.jsonl")
    model = registry.register(ModelArtifact(uri="first"))
    model.uri = "second"
    with registry.path.open("a") as stream:
        stream.write(json.dumps(model.to_dict()) + "\n")
    with pytest.raises(ValueError, match=":2"):
        registry.get(model.id)


def test_registry_reader_waits_for_writer_to_finish_a_record(tmp_path):
    from concurrent.futures import ThreadPoolExecutor, TimeoutError
    registry = Registry(tmp_path / "a.jsonl")
    reader = Registry(registry.path)
    model = ModelArtifact(uri="checkpoint")
    encoded = json.dumps(model.to_dict()) + "\n"
    started = threading.Event()
    def read():
        started.set()
        return reader.all()
    with ThreadPoolExecutor(max_workers=1) as pool:
        with registry._lock:
            with registry.path.open("w") as stream:
                stream.write(encoded[:20])
                stream.flush()
                result = pool.submit(read)
                assert started.wait(timeout=5)
                with pytest.raises(TimeoutError):
                    result.result(timeout=.1)
                stream.write(encoded[20:])
        assert result.result(timeout=5)[0].id == model.id


def test_lineage_allows_shared_ancestors(tmp_path):
    registry = Registry(tmp_path / "a.jsonl")
    for artifact_id, parents in [("root", []), ("left", ["root"]), ("right", ["root"]),
                                  ("tip", ["left", "right"])]:
        registry.register(ModelArtifact(id=artifact_id, parents=parents))
    assert [a.id for a in registry.lineage("tip")] == ["tip", "left", "right", "root"]


def test_setup_quotes_remote_directory_and_keeps_command_in_container(tmp_path, monkeypatch):
    calls = []
    def run(args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")
    monkeypatch.setattr(subprocess, "run", run)
    command = "printf first && printf second > result"
    result = EnvConfigTool(ForgeRunner(tmp_path)).run(command, host="u@host:2222",
        cwd="~/project name; literal", in_container=True)
    assert result.ok
    args, _ = calls[0]
    assert "StrictHostKeyChecking=yes" in args and "2222" in args
    remote = shlex.split(args[-1])
    assert remote[:3] == ["cd", "$HOME/project name; literal", "&&"]
    assert remote[3:] == ["bash", "scripts/run_in_container.sh", "bash", "-lc", command]


def test_invalid_ssh_target_cannot_become_an_option(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: pytest.fail("invalid target reached SSH"))
    result = EnvConfigTool(ForgeRunner(tmp_path)).run("echo ok", host="-oProxyCommand=anything")
    assert not result.ok


@pytest.mark.parametrize("status,expected", [(401, 1), (503, 3)])
def test_real_litellm_request_count_against_local_http_server(monkeypatch, status, expected):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "true")
    monkeypatch.setenv("FORGEWRIGHT_TEST_KEY", "fixture-only")
    requests = []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            requests.append(self.path)
            body = json.dumps({"error": {"message": "fixture failure", "type": "server_error"}}).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Retry-After", "0")
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, *args):
            pass
    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            client = Brain(ProviderConfig(name="fixture", kind="vllm", model="fixture",
                api_base=f"http://127.0.0.1:{server.server_port}/v1", api_key_env="FORGEWRIGHT_TEST_KEY"), timeout=5)
            with pytest.raises(BrainError):
                client.chat([{"role": "user", "content": "fixture"}])
            assert requests == ["/v1/chat/completions"] * expected
        finally:
            server.shutdown()
            thread.join(timeout=5)
