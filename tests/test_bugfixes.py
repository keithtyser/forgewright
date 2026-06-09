"""Regression tests for trace-surfaced bugs: empty cwd/command, ~ in introspection paths, and
the quant config quality-recovery knobs."""
from __future__ import annotations

from forgewright.skills.modelspec import model_spec
from forgewright.skills.quantize import scaffold_quant_config
from forgewright.tools.shell import ShellTool


def test_shell_rejects_empty_command():
    res = ShellTool().run(command="   ")
    assert res.ok is False and "empty command" in res.output


def test_shell_empty_cwd_does_not_crash():
    # an empty-string cwd previously crashed with ENOENT ('' is not a directory)
    res = ShellTool().run(command="echo hi", cwd="")
    assert res.ok is True and "hi" in res.output


class _CapRunner:
    """Captures the args passed to `forge` so we can assert ~ was expanded."""
    def __init__(self):
        self.seen = None

    def run(self, args, **kw):
        self.seen = args

        class R:
            ok = True
            output = '{"architecture": "Qwen3_5ForCausalLM", "model_type": "qwen3_5"}'
        return R()


def test_model_spec_expands_home(monkeypatch):
    monkeypatch.setenv("HOME", "/home/tester")
    monkeypatch.setattr("os.path.expanduser", lambda p: p.replace("~", "/home/tester"))
    r = _CapRunner()
    spec = model_spec(r, "~/models/Qwen3.5-0.8B")
    assert "~" not in r.seen and "/home/tester/models/Qwen3.5-0.8B" in r.seen
    assert spec and spec["model_type"] == "qwen3_5"


def test_quant_scaffold_renders_recovery_knobs():
    cfg = scaffold_quant_config("q08", method="nvfp4",
                                extra_exclusions=["down_proj", "o_proj"],
                                keep_kv_high_precision=True, calib_samples=512)
    assert "down_proj" in cfg and "o_proj" in cfg
    assert "lm_head" in cfg and "embed_tokens" in cfg          # base keep-list still present
    assert "kv_cache_qformat: none" in cfg                      # KV kept high precision (ptq)
    assert "kv_cache_dtype: auto" in cfg                        # and at serve time
    assert "samples: 512" in cfg


def test_quant_scaffold_default_is_aggressive():
    cfg = scaffold_quant_config("q08", method="nvfp4")
    assert "kv_cache_dtype: fp8" in cfg                         # default: quantized KV (fast)
    assert "down_proj" not in cfg                               # no extra exclusions by default
