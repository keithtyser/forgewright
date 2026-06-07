"""Tests for the serving-opt decision logic (pure; no serving needed)."""
from __future__ import annotations

from forgewright.skills.serving_opt import (
    CandidateResult,
    _parse_metric,
    default_candidates,
    default_served_name,
    select_best,
)


def test_default_candidates():
    lat = {c.name: c for c in default_candidates("latency")}
    assert {"baseline", "ngram_spec", "eager_lowlat"} <= set(lat)
    assert "--speculative-config" in lat["ngram_spec"].extra_flags
    assert lat["baseline"].extra_flags == []
    thr = {c.name: c for c in default_candidates("throughput")}
    assert "batch_throughput" in thr


def test_select_best_latency_excludes_quality_regressions():
    rs = [
        CandidateResult("baseline", True, single_tps=31.0, quality_ok=True),
        CandidateResult("ngram", True, single_tps=45.0, quality_ok=True),
        CandidateResult("fast_but_bad", True, single_tps=60.0, quality_ok=False),  # excluded
        CandidateResult("crashed", False),
    ]
    best = select_best(rs, "latency")
    assert best is not None and best.name == "ngram"


def test_select_best_throughput():
    rs = [
        CandidateResult("baseline", True, aggregate_tps=200.0, quality_ok=True),
        CandidateResult("batch", True, aggregate_tps=900.0, quality_ok=True),
    ]
    assert select_best(rs, "throughput").name == "batch"


def test_select_best_none_eligible():
    rs = [CandidateResult("x", False), CandidateResult("y", True, single_tps=30.0, quality_ok=False)]
    assert select_best(rs, "latency") is None


def test_objective_score_uses_right_metric():
    r = CandidateResult("c", True, single_tps=30.0, aggregate_tps=500.0, quality_ok=True)
    assert r.objective_score("latency") == 30.0
    assert r.objective_score("throughput") == 500.0


def test_default_served_name():
    assert default_served_name("qwen35_9b", "base_nvfp4_modelopt") == "model-forge/qwen35-9b-base-nvfp4-modelopt"


def test_parse_metric_csv_and_table():
    assert _parse_metric("output_tokens_per_second,31.5,2", "output_tokens_per_second") == 31.5
    assert _parse_metric("normal_use_regression_pass_rate   0.81   32", "normal_use_regression_pass_rate") == 0.81
    assert _parse_metric("nothing here", "missing") is None
