"""Tests for the DataCurator specialist (curation hygiene + curate_seed; no LLM/GPU)."""
from __future__ import annotations

import json
from pathlib import Path

from forgewright.agents.data_curator import (
    DataCurator,
    _closed_think_ok,
    _valid_messages,
    curate_messages_rows,
)
from forgewright.registry import Registry
from forgewright.tools.base import ToolResult


def _msg(u, a):
    return {"messages": [{"role": "user", "content": u}, {"role": "assistant", "content": a}]}


def test_closed_think_ok():
    assert _closed_think_ok("no block")
    assert _closed_think_ok("<think>reason</think> 41")
    assert not _closed_think_ok("</think> before <think>")   # wrong order
    assert not _closed_think_ok("<think>unclosed")            # missing close
    assert not _closed_think_ok("<think>a</think><think>b</think>")  # doubled


def test_valid_messages():
    assert _valid_messages(_msg("2+2?", "<think>add</think> 4"))
    assert not _valid_messages(_msg("hi", ""))                       # empty assistant
    assert not _valid_messages({"messages": [{"role": "user", "content": "lonely"}]})  # no assistant
    assert not _valid_messages({"messages": [{"role": "user", "content": "q"},
                                             {"role": "assistant", "content": "<think>oops"}]})  # bad think
    assert not _valid_messages({"text": "wrong schema"})


def test_curate_dedups_and_drops_invalid():
    rows = [
        _msg("2+2?", "4"),
        _msg("2+2?", "4"),                       # duplicate
        _msg("hi", ""),                          # invalid (empty)
        _msg("3+3?", "<think>6</think> 6"),
    ]
    kept, drops = curate_messages_rows(rows)
    assert len(kept) == 2
    assert drops == {"invalid": 1, "duplicate": 1}


class FakeForge:
    def __init__(self, repo): self.repo = repo
    def available(self): return True
    def run(self, args, timeout=0, **_): return ToolResult(True, "ok")


def test_curate_seed_produces_dataset_artifact(tmp_path):
    repo = tmp_path
    seed = repo / "datasets" / "seeds" / "s.jsonl"
    seed.parent.mkdir(parents=True)
    seed.write_text("\n".join(json.dumps(r) for r in [
        _msg("2+2?", "4"), _msg("2+2?", "4"), _msg("bad", "")]))
    reg = Registry(repo / "artifacts.jsonl")
    dc = DataCurator(runner=FakeForge(repo), registry=reg)
    art = dc.run([], "curate", mode="curate_seed", seed_paths=["datasets/seeds/s.jsonl"],
                 family="qwen35_0_8b", source="Qwen/Qwen3.5-0.8B", run_name="q08_curated",
                 holdout="datasets/rl/add_holdout.jsonl")

    assert art.kind == "dataset" and art.gate.passed
    assert art.meta["rows"] == 1 and art.meta["family"] == "qwen35_0_8b"
    assert art.meta["holdout"] == "datasets/rl/add_holdout.jsonl"
    # wrote the curated jsonl into the (tmp) repo, deduped+validated
    out = repo / art.uri
    assert out.exists()
    lines = [l for l in out.read_text().splitlines() if l.strip()]
    assert len(lines) == 1
    # discoverable for the next stage
    assert reg.latest("dataset", family="qwen35_0_8b").id == art.id


def test_curate_seed_fails_gate_when_no_valid_rows(tmp_path):
    repo = tmp_path
    (repo / "datasets").mkdir()
    (repo / "datasets" / "empty.jsonl").write_text(json.dumps(_msg("x", "")))  # all invalid
    dc = DataCurator(runner=FakeForge(repo), registry=Registry(repo / "r.jsonl"))
    art = dc.run([], "", seed_paths=["datasets/empty.jsonl"], family="q")
    assert not art.gate.passed and "no valid rows" in art.gate.verdict
