"""Tests for the generate -> verify -> repair loop: repair policies + the Director retry loop."""
from __future__ import annotations

from forgewright.agents.base import Specialist
from forgewright.agents.director import Director, Step
from forgewright.agents.memory import OutcomeMemory
from forgewright.agents.repair import abliterate_repair, default_repair, finetune_repair, policy_for
from forgewright.contracts import Gate, ModelArtifact
from forgewright.registry import Registry
from forgewright.tools.base import ToolRegistry


# --- repair policies (pure) -------------------------------------------------------

def _failed(meta=None, verdict="capability regressed"):
    return ModelArtifact(uri="x", gate=Gate(passed=False, verdict=verdict), meta=meta or {"family": "q08"})


def test_default_repair_retries_same_params():
    assert default_repair(2, _failed(), {"max_steps": 60}) == {"max_steps": 60}


def test_abliterate_repair_backs_off_strength_and_layers():
    out = abliterate_repair(2, _failed(), {"strength": 3.0, "layer_skip_first": 4})
    assert out["strength"] == 2.1 and out["layer_skip_first"] == 5


def test_abliterate_repair_gives_up_when_too_weak():
    assert abliterate_repair(2, _failed(), {"strength": 0.6}) is None   # 0.42 < 0.5 floor


def test_abliterate_repair_seeds_from_memory(tmp_path):
    m = OutcomeMemory(path=tmp_path / "o.jsonl")
    m.record(stage="Abliterator", family="q08", params={"strength": 1.5, "layer_skip_first": 6}, passed=True)
    out = abliterate_repair(2, _failed(), {"strength": 3.0, "layer_skip_first": 4}, memory=m)
    assert out["strength"] == 1.5 and out["layer_skip_first"] == 6   # used the known-good config


def test_finetune_repair_shortens_then_gives_up():
    assert finetune_repair(2, _failed(), {"max_steps": 60})["max_steps"] == 36
    assert finetune_repair(2, _failed(), {"max_steps": 10}) is None   # 6 < 10 floor


def test_policy_for_maps_roles():
    assert policy_for("Abliterator") is abliterate_repair
    assert policy_for("SFTTrainer") is finetune_repair
    assert policy_for("Unknown") is default_repair


# --- Director repair loop (integration) -------------------------------------------

class FlakyAbliterator(Specialist):
    """Fails the gate until strength drops to/below a threshold, then passes — exercising the
    Director's repair loop end to end."""
    role = "Abliterator"
    accepts = ("model",)
    produces = "model"

    pass_at = 1.5   # passes once strength <= this

    def system_prompt(self): return "ablate"
    def tools(self): return ToolRegistry([])

    def run(self, inputs, goal="", *, strength=3.0, layer_skip_first=4, **kw):
        ok = strength <= self.pass_at
        art = ModelArtifact(uri=f"~/m-{strength}", produced_by=self.role, parents=[inputs[0].id],
                            gate=Gate(passed=ok, metrics={"strength": strength},
                                      verdict="ABLITERATED" if ok else "capability regressed"),
                            meta={"family": "q08", "role": "abliterated"})
        self.registry.register(art)
        return art


def test_director_repairs_until_gate_passes(tmp_path):
    reg = Registry(tmp_path / "a.jsonl")
    mem = OutcomeMemory(path=tmp_path / "o.jsonl")
    seed = reg.register(ModelArtifact(uri="~/base", meta={"family": "q08", "role": "base"}))
    events = []
    director = Director(registry=reg, reporter=lambda k, d: events.append((k, d)), memory=mem)
    # strength 3.0 -> 2.1 -> 1.47(<=1.5) passes on the 3rd attempt
    res = director.run_recipe("abliterate", [Step(FlakyAbliterator, run_kwargs={"strength": 3.0},
                                                 max_attempts=3)], seed_inputs=[seed])
    assert res.ok is True
    repairs = [d for k, d in events if k == "repair"]
    assert len(repairs) == 2 and "strength" in repairs[0]["changes"]
    # every attempt got recorded to the learning loop (2 fails + 1 pass)
    rows = mem.recall(stage="Abliterator", family="q08")
    assert len(rows) == 3 and rows[0]["passed"] is True


def test_director_halts_when_repair_exhausted(tmp_path):
    reg = Registry(tmp_path / "a.jsonl")
    mem = OutcomeMemory(path=tmp_path / "o.jsonl")
    seed = reg.register(ModelArtifact(uri="~/base", meta={"family": "q08", "role": "base"}))
    director = Director(registry=reg, reporter=lambda k, d: None, memory=mem)
    # only 2 attempts: 3.0 -> 2.1, neither <= 1.5, so it halts
    res = director.run_recipe("abliterate", [Step(FlakyAbliterator, run_kwargs={"strength": 3.0},
                                                 max_attempts=2)], seed_inputs=[seed])
    assert res.ok is False and res.failed_at == "Abliterator"


def test_no_repair_when_max_attempts_one(tmp_path):
    reg = Registry(tmp_path / "a.jsonl")
    seed = reg.register(ModelArtifact(uri="~/base", meta={"family": "q08", "role": "base"}))
    events = []
    director = Director(registry=reg, reporter=lambda k, d: events.append((k, d)),
                       memory=OutcomeMemory(path=tmp_path / "o.jsonl"))
    res = director.run_recipe("abliterate", [Step(FlakyAbliterator, run_kwargs={"strength": 3.0})],
                             seed_inputs=[seed])
    assert res.ok is False and not [d for k, d in events if k == "repair"]   # halted immediately
