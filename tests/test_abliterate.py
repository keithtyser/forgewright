"""Tests for the abliterate skill: dual-gate logic, scaffolding, scores parsing."""
from __future__ import annotations

from pathlib import Path

from forgewright.skills.abliterate import (
    abliterate_gate,
    read_abliterate_metrics,
    scaffold_abliterate_config,
    write_abliterate_config,
)


def test_gate_passes_when_refusal_drops_and_capability_holds():
    base = {"refusal_rate_harmful": 0.90, "capability": 0.80, "benign_refusal_rate": 0.05}
    cand = {"refusal_rate_harmful": 0.10, "capability": 0.78, "benign_refusal_rate": 0.08}
    r = abliterate_gate(base, cand)
    assert r["passed"] and r["verdict"] == "ABLITERATED"
    assert round(r["refusal_drop"], 2) == 0.80 and r["capability_held"]


def test_gate_fails_when_refusal_did_not_drop():
    base = {"refusal_rate_harmful": 0.90, "capability": 0.80, "benign_refusal_rate": 0.05}
    cand = {"refusal_rate_harmful": 0.85, "capability": 0.80, "benign_refusal_rate": 0.05}
    r = abliterate_gate(base, cand)  # drop 0.05 < 0.10
    assert not r["passed"] and "refusal did not drop" in r["verdict"]


def test_gate_fails_when_capability_regresses():
    base = {"refusal_rate_harmful": 0.90, "capability": 0.80, "benign_refusal_rate": 0.05}
    cand = {"refusal_rate_harmful": 0.10, "capability": 0.60, "benign_refusal_rate": 0.05}
    r = abliterate_gate(base, cand)  # capability -0.20 beyond tolerance
    assert not r["passed"] and "capability regressed" in r["verdict"]


def test_gate_fails_when_over_abliterated():
    base = {"refusal_rate_harmful": 0.90, "capability": 0.80, "benign_refusal_rate": 0.05}
    cand = {"refusal_rate_harmful": 0.05, "capability": 0.79, "benign_refusal_rate": 0.55}
    r = abliterate_gate(base, cand)  # benign refusals exploded -> over-abliterated
    assert not r["passed"] and "over-abliterated" in r["verdict"]


def test_scaffold_bakes_capability_preserving_defaults():
    cfg = scaffold_abliterate_config("qwen35_0_8b", source="Qwen/Qwen3.5-0.8B")
    assert "method: contrastive_refusal_direction" in cfg
    assert "leave_embeddings_untouched: true" in cfg
    assert "leave_lm_head_untouched: true" in cfg
    assert "leave_moe_experts_untouched: true" in cfg
    assert "norm_preserve: true" in cfg
    assert "harmful_prompts: ../../datasets/abliteration/harmful_refusal.yaml" in cfg
    assert "require_execute_flag: true" in cfg


def test_scaffold_omits_layer_window_by_default():
    # layer_start/end omitted -> model-forge derives the edit window from collected directions,
    # so the collected and edited layer ranges can't silently disagree.
    cfg = scaffold_abliterate_config("qwen35_0_8b", source="Qwen/Qwen3.5-0.8B")
    assert "layer_start:" not in cfg and "layer_end:" not in cfg
    assert "layer_skip_first: 4" in cfg and "target_weight_suffixes:" in cfg


def test_scaffold_includes_layer_window_when_set():
    cfg = scaffold_abliterate_config("qwen35_0_8b", source="Qwen/Qwen3.5-0.8B", layer_start=4, layer_end=21)
    assert "layer_start: 4" in cfg and "layer_end: 21" in cfg


def test_scaffold_config_is_valid_yaml():
    import yaml

    cfg = scaffold_abliterate_config("qwen35_0_8b", source="Qwen/Qwen3.5-0.8B")
    doc = yaml.safe_load(cfg)
    assert doc["edit"]["mode"] == "projection"
    assert "layer_start" not in doc["edit"] and "layer_end" not in doc["edit"]
    assert doc["activation_collection"]["layer_skip_first"] == 4


def test_write_abliterate_config_idempotent(tmp_path: Path):
    cfg = write_abliterate_config(tmp_path, "qwen35_0_8b", source="Qwen/Qwen3.5-0.8B")
    assert cfg == tmp_path / "configs" / "abliteration" / "qwen35_0_8b_abliterated_v0.yaml"
    assert cfg.exists()
    cfg.write_text(cfg.read_text() + "# touched\n")
    write_abliterate_config(tmp_path, "qwen35_0_8b", source="Qwen/Qwen3.5-0.8B", overwrite=False)
    assert "# touched" in cfg.read_text()  # not overwritten
    write_abliterate_config(tmp_path, "qwen35_0_8b", source="Qwen/Qwen3.5-0.8B", overwrite=True)
    assert "# touched" not in cfg.read_text()


class _FakeJobs:
    """Stand-in JobManager: launch/wait return a scripted exit code, no real process."""

    def __init__(self, exit_code=0):
        self._exit = exit_code

    def launch(self, cmd, host=None, cwd=None, name=None):
        return {"id": "job-test"}

    def wait(self, jid, **kw):
        return {"exit_code": self._exit}


def _abliterator(tmp_path, exit_code, fresh, monkeypatch):
    from forgewright.agents.abliterator import Abliterator
    from forgewright.registry import Registry
    from forgewright.tools.forge import ForgeRunner

    monkeypatch.setattr(Abliterator, "_wrote_fresh_weights", lambda self, out, since: fresh)
    return Abliterator(registry=Registry(tmp_path / "reg.jsonl"),
                       runner=ForgeRunner(repo=tmp_path), jobs=_FakeJobs(exit_code))


def test_abliterator_uses_unique_output_name_not_v0(tmp_path, monkeypatch):
    from forgewright.contracts import ModelArtifact

    abl = _abliterator(tmp_path, exit_code=0, fresh=True, monkeypatch=monkeypatch)
    src = ModelArtifact(uri="/home/u/models/Qwen3.5-0.8B", meta={"family": "qwen35_0_8b", "role": "base"})
    art = abl.run([src])
    # never the colliding _v0 dir; a unique, timestamped name instead
    assert "abliterated_v0" not in art.uri and "_abliterated_" in art.uri


def test_abliterator_fails_when_no_fresh_weights(tmp_path, monkeypatch):
    """exit 0 but nothing freshly written -> refuse to claim a stale/pre-existing model."""
    from forgewright.contracts import ModelArtifact

    abl = _abliterator(tmp_path, exit_code=0, fresh=False, monkeypatch=monkeypatch)
    src = ModelArtifact(uri="/home/u/models/Qwen3.5-0.8B", meta={"family": "qwen35_0_8b"})
    art = abl.run([src])
    assert art.gate.passed is False
    assert "stale/pre-existing" in art.gate.verdict
    assert art.gate.metrics["fresh_weights"] is False


def test_abliterator_passes_only_with_fresh_weights(tmp_path, monkeypatch):
    from forgewright.contracts import ModelArtifact

    abl = _abliterator(tmp_path, exit_code=0, fresh=True, monkeypatch=monkeypatch)
    art = abl.run([ModelArtifact(uri="/home/u/models/Qwen3.5-0.8B", meta={"family": "qwen35_0_8b"})])
    assert art.gate.passed is True and art.gate.verdict == "ABLITERATED"


def test_abliterator_fails_on_nonzero_exit(tmp_path, monkeypatch):
    from forgewright.contracts import ModelArtifact

    abl = _abliterator(tmp_path, exit_code=1, fresh=True, monkeypatch=monkeypatch)
    art = abl.run([ModelArtifact(uri="/home/u/models/Qwen3.5-0.8B", meta={"family": "qwen35_0_8b"})])
    assert art.gate.passed is False and "did not exit cleanly" in art.gate.verdict


def test_abliterator_resolves_hf_id_to_local_mirror(tmp_path, monkeypatch):
    abl = _abliterator(tmp_path, exit_code=0, fresh=True, monkeypatch=monkeypatch)
    # a real path passes through unchanged (no host call)
    assert abl._resolve_local("/home/u/models/Qwen3.5-0.8B") == "/home/u/models/Qwen3.5-0.8B"
    assert abl._resolve_local("~/models/X") == "~/models/X"
    # a bare HF id resolves to a local mirror when the host reports it exists
    monkeypatch.setattr(type(abl), "_host_run", lambda self, cmd, timeout=60: "yes")
    assert abl._resolve_local("Qwen/Qwen3.5-0.8B") == "~/models/Qwen3.5-0.8B"
    # ...but stays the HF id when no local mirror is present
    monkeypatch.setattr(type(abl), "_host_run", lambda self, cmd, timeout=60: "")
    assert abl._resolve_local("Qwen/Qwen3.5-0.8B") == "Qwen/Qwen3.5-0.8B"


def test_read_abliterate_metrics(tmp_path: Path):
    p = tmp_path / "scores.csv"
    p.write_text(
        "bucket,metric,value,count\n"
        "unsafe_overcompliance,refusal_rate_harmful,0.90,3\n"
        "capability_preservation_challenge,normal_use_regression_pass_rate,0.81,32\n"
        "refusal_benign_boundary,benign_refusal_rate,0.06,3\n"
        "agentic_code_debug,workflow_success,1.0,2\n"
    )
    m = read_abliterate_metrics(p)
    assert m == {"refusal_rate_harmful": 0.90, "capability": 0.81, "benign_refusal_rate": 0.06}
    assert read_abliterate_metrics(tmp_path / "nope.csv") == {}
