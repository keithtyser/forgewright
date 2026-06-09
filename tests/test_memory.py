"""Tests for the outcome/experience memory (the cross-run learning loop)."""
from __future__ import annotations

from forgewright.agents.memory import OutcomeMemory


def _mem(tmp_path):
    return OutcomeMemory(path=tmp_path / "outcomes.jsonl")


def test_record_and_recall(tmp_path):
    m = _mem(tmp_path)
    m.record(stage="Abliterator", family="q08", params={"strength": 3.0}, passed=False,
             verdict="capability regressed")
    m.record(stage="Abliterator", family="q08", params={"strength": 1.5}, passed=True)
    rows = m.recall(stage="Abliterator", family="q08")
    assert len(rows) == 2 and rows[0]["passed"] is True   # newest first
    assert m.recall(stage="Abliterator", passed=True)[0]["params"]["strength"] == 1.5


def test_best_params_returns_latest_passing(tmp_path):
    m = _mem(tmp_path)
    m.record(stage="Abliterator", family="q08", params={"strength": 2.0, "layer_skip_first": 5}, passed=True)
    m.record(stage="Abliterator", family="q08", params={"strength": 3.0}, passed=False)
    m.record(stage="Abliterator", family="q08", params={"strength": 1.2, "layer_skip_first": 6}, passed=True)
    best = m.best_params(stage="Abliterator", family="q08")
    assert best == {"strength": 1.2, "layer_skip_first": 6}
    assert m.best_params(stage="Abliterator", family="other") is None


def test_only_tunables_are_stored(tmp_path):
    m = _mem(tmp_path)
    m.record(stage="SFTTrainer", family="q08",
             params={"max_steps": 60, "holdout": "h.jsonl", "secret": "x"}, passed=True)
    stored = m.all()[0]["params"]
    assert stored == {"max_steps": 60}   # holdout/secret dropped, only tunables kept


def test_digest_empty_then_populated(tmp_path):
    m = _mem(tmp_path)
    assert m.digest() == ""
    m.record(stage="Abliterator", family="q08", params={"strength": 3.0}, passed=False,
             verdict="capability regressed")
    d = m.digest(family="q08")
    assert "Abliterator" in d and "FAIL" in d and "strength=3.0" in d


def test_garbled_line_is_skipped(tmp_path):
    m = _mem(tmp_path)
    m.record(stage="Quantizer", family="q08", params={"method": "nvfp4"}, passed=True)
    with m.path.open("a", encoding="utf-8") as f:
        f.write("{ broken json\n")
    assert len(m.all()) == 1   # tolerant read
