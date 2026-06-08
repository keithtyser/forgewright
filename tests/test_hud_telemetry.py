"""Tests for the HUD telemetry + control-command backend: the structured metric parser/tap,
the command channel (route + run_command), and the /graph and /models event builders."""
from __future__ import annotations

from forgewright.brain.codex_oauth import parse_models_payload
from forgewright.cli import _graph_event, _models_event
from forgewright.config import parse_brain_arg
from forgewright.contracts import AdapterArtifact, DatasetArtifact, EvalArtifact, Gate
from forgewright.frontend.bridge import metric_tap, parse_metrics
from forgewright.frontend.server import BackendServer, route
from forgewright.registry import Registry


# --- metric parsing ---------------------------------------------------------
def test_parse_metrics_hf_dict():
    m = parse_metrics("{'loss': 0.31, 'grad_norm': 1.2, 'learning_rate': 9e-05, 'epoch': 0.5}")
    assert m == {"loss": 0.31, "grad_norm": 1.2, "lr": 9e-05, "epoch": 0.5}


def test_parse_metrics_tqdm_needs_training_signal():
    # a bare tqdm bar with NO training signal is not a step (avoids download/load false positives)
    assert parse_metrics("40%|####  | 40/120 [00:10<00:20]") == {}
    # but a training bar that carries loss in the same blob is a real step
    m = parse_metrics("{'loss': 0.31}\n 33%|###  | 40/120 [00:10<00:20, 1.9it/s]")
    assert m["loss"] == 0.31 and m["step"] == 40 and m["total"] == 120


def test_parse_metrics_explicit_step_and_kv():
    m = parse_metrics("step 40/120 loss: 0.31 reward=1.5 kl: 0.02")
    assert m["step"] == 40 and m["total"] == 120 and m["loss"] == 0.31 and m["reward"] == 1.5 and m["kl"] == 0.02


def test_parse_metrics_ignores_download_and_shard_bars():
    # the exact false positive from the field: a HF file-fetch bar must NOT become a step
    assert parse_metrics("Fetching 13 files:   0%|          | 0/13 [00:00<?, ?it/s]") == {}
    assert parse_metrics("Loading checkpoint shards:  50%|##   | 2/4 [00:01<00:01]") == {}


def test_parse_metrics_none():
    assert parse_metrics("just some prose, nothing numeric here") == {}
    assert parse_metrics("") == {}


def test_metric_tap_emits_metric_for_training_output():
    events = []
    tapped = metric_tap(lambda k, d: events.append((k, d)))
    tapped("progress", {"role": "SFTTrainer", "text": "step 10/100 loss 0.5"})
    kinds = [k for k, _ in events]
    assert kinds == ["progress", "metric"]
    metric = events[1][1]
    assert metric["role"] == "SFTTrainer" and metric["loss"] == 0.5 and metric["step"] == 10


def test_metric_tap_passthrough_when_no_metrics():
    events = []
    tapped = metric_tap(lambda k, d: events.append((k, d)))
    tapped("tool", {"tool": "discover", "output": "local models: a, b"})
    tapped("assistant", {"content": "hello, loss of generality"})  # not tool/progress -> never tapped
    assert [k for k, _ in events] == ["tool", "assistant"]


# --- command channel --------------------------------------------------------
def test_route_sends_command_to_turn_queue():
    import queue

    mq, aq = queue.Queue(), queue.Queue()
    route({"type": "command", "name": "graph"}, mq, aq)
    msg = mq.get_nowait()
    assert msg["type"] == "command" and msg["name"] == "graph"


def test_backend_run_command_dispatches_and_done():
    emitted = []
    seen = {}

    def handle_command(name, args, reporter, emit):
        seen["name"] = name
        emit({"type": "models", "available": ["gpt-5"]})

    srv = BackendServer(lambda *a: None, emitted.append, handle_command=handle_command)
    srv.run_command({"type": "command", "name": "models", "args": {}})
    assert seen["name"] == "models"
    assert emitted[-1] == {"type": "done", "ok": True}
    assert any(e.get("type") == "models" for e in emitted)


def test_backend_run_command_reports_errors():
    emitted = []

    def boom(*a):
        raise RuntimeError("nope")

    BackendServer(lambda *a: None, emitted.append, handle_command=boom).run_command({"name": "x"})
    assert emitted[-1]["type"] == "done" and emitted[-1]["ok"] is False and "nope" in emitted[-1]["error"]


# --- /graph + /models event builders ----------------------------------------
def test_graph_event_from_registry(tmp_path):
    reg = Registry(tmp_path / "artifacts.jsonl")
    ds = reg.register(DatasetArtifact(uri="d.jsonl"))
    ad = reg.register(AdapterArtifact(uri="a", produced_by="SFTTrainer", parents=[ds.id]))
    reg.register(EvalArtifact(uri="s.json", produced_by="Evaluator", parents=[ad.id],
                              gate=Gate(passed=True, metrics={"score": 0.94})))
    ev = _graph_event(reg)
    assert ev["type"] == "graph" and len(ev["nodes"]) == 3
    eval_node = next(n for n in ev["nodes"] if n["kind"] == "eval")
    assert eval_node["parents"] == [ad.id] and eval_node["passed"] is True and eval_node["score"] == 0.94


def test_models_event_non_codex_brain():
    ev = _models_event(parse_brain_arg("openrouter:deepseek/deepseek-v4-pro"))
    assert ev["type"] == "models" and ev["source"] == "brain"
    assert "Codex-only" in ev["note"]


# --- codex /models payload parsing ------------------------------------------
def test_parse_models_payload_shapes():
    assert parse_models_payload({"models": [{"slug": "gpt-5.5-codex"}, {"id": "gpt-5"}]}) == ["gpt-5.5-codex", "gpt-5"]
    assert parse_models_payload({"data": [{"id": "gpt-5-codex"}]}) == ["gpt-5-codex"]
    assert parse_models_payload(["gpt-5", "gpt-5", "gpt-4"]) == ["gpt-5", "gpt-4"]  # de-duped
    assert parse_models_payload({"nope": 1}) == []
