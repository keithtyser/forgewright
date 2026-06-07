"""Tests for the serving-opt decision logic (pure; no serving needed)."""
from __future__ import annotations

from forgewright.skills.serving_opt import (
    CandidateResult,
    _parse_metric,
    _scores_csv_metric,
    default_candidates,
    default_served_name,
    select_best,
)


def test_default_candidates():
    lat = {c.name: c for c in default_candidates("latency")}
    assert {"baseline", "ngram_spec", "ngram_lowlat"} <= set(lat)
    # ngram candidate sets the serve-time env model-forge honors
    assert "ngram" in lat["ngram_spec"].env["VLLM_SPECULATIVE_CONFIG"]
    assert lat["baseline"].env == {}
    # low-latency candidate pins single-sequence batching on top of spec-decode
    assert "--max-num-seqs 1" in lat["ngram_lowlat"].env["VLLM_EXTRA_ARGS"]
    thr = {c.name: c for c in default_candidates("throughput")}
    assert "batch_throughput" in thr
    assert "--max-num-seqs" in thr["batch_throughput"].env["VLLM_EXTRA_ARGS"]


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


def test_scores_csv_metric_picks_right_bucket(tmp_path):
    # the metric name recurs across buckets — gate must read the capability bucket cell,
    # not the first/any match (mirrors a real model-forge scores.csv).
    p = tmp_path / "scores.csv"
    p.write_text(
        "bucket,metric,value,count,pass_count,fail_count,ci_low,ci_high,stddev\n"
        "normal_use_regression,normal_use_regression_pass_rate,1.0,3,3,0,0.43,1.0,0.0\n"
        "capability_preservation_challenge,normal_use_regression_pass_rate,0.8125,32,26,6,0.64,0.91,0.39\n"
        "agentic_code_debug,workflow_success,1.0,2,2,0,0.34,1.0,0.0\n"
    )
    assert _scores_csv_metric(p, "capability_preservation_challenge", "normal_use_regression_pass_rate") == 0.8125
    assert _scores_csv_metric(p, "normal_use_regression", "normal_use_regression_pass_rate") == 1.0
    assert _scores_csv_metric(p, "missing_bucket", "normal_use_regression_pass_rate") is None
    assert _scores_csv_metric(tmp_path / "nope.csv", "x", "y") is None
