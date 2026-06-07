"""Eval-gate: did the fine-tune help without regressing the base?

A self-contained, held-out verifiable eval that needs no serving/family-config: it
generates base-vs-adapter completions on a held-out set and scores them with the same
verifiable reward used in training (numeric correctness), then makes a pass/fail gate
decision. Runs in the posttrain container (same vehicle as training). Pure decision +
reward logic is unit-tested; the generated script reuses the reward source verbatim.

This is the right gate for *task* (verifiable) fine-tunes. For *uplift* capability-
regression checks, serve the adapter (model-forge serve honors VLLM_ENABLE_LORA +
MODEL_FORGE_LORA_MODULES) and reuse the model-forge internal eval / scores.csv gate.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from forgewright.trainers.rl import _REWARD_SOURCE


def gate_pass(base_score: float, candidate_score: float, *, tolerance: float = 0.0) -> bool:
    """The candidate passes if it does not regress the base beyond `tolerance`
    (tolerance absorbs eval noise; set 0.0 to require strict non-regression)."""
    return candidate_score >= base_score - tolerance


def gate_report(base_score: float, candidate_score: float, *, tolerance: float = 0.0) -> dict:
    delta = candidate_score - base_score
    return {
        "base_score": base_score,
        "candidate_score": candidate_score,
        "delta": delta,
        "tolerance": tolerance,
        "passed": gate_pass(base_score, candidate_score, tolerance=tolerance),
        "verdict": "PASS" if gate_pass(base_score, candidate_score, tolerance=tolerance)
        else "REGRESSION",
    }


_EVAL_GATE_TEMPLATE = '''\
"""Forgewright held-out verifiable eval-gate (generated; runs in the posttrain container)."""
import json, os, re, sys
from pathlib import Path
from typing import Optional

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

cfg = json.loads(Path(sys.argv[sys.argv.index("--config") + 1]).read_text())

{reward_source}

tok = AutoTokenizer.from_pretrained(cfg["base"])
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
ds = load_dataset("json", data_files=cfg["holdout"], split="train")
adapter = os.path.expanduser(cfg["adapter"])

def _lora_b_sum(m):
    return sum(float(p.abs().sum()) for n, p in m.named_parameters() if "lora_B" in n)

def _load_base(loader):
    return loader.from_pretrained(cfg["base"], dtype=torch.bfloat16).to("cuda")

# Different trainers load this model as different classes (TRL -> ForConditionalGeneration
# with the text tower under .language_model; model-forge SFT -> flat ForCausalLM). The
# adapter keys only match one of them, so pick the loader where the adapter ACTUALLY
# applies (non-zero lora_B). Fail loudly if none does — never report a silent no-op gate.
def load_pair():
    for loader in (AutoModelForImageTextToText, AutoModelForCausalLM):
        try:
            base = _load_base(loader)
        except Exception:
            continue
        ft = PeftModel.from_pretrained(base, adapter)
        if _lora_b_sum(ft) > 0:
            return _load_base(loader), ft  # fresh base (clean) + adapter-applied model
    raise RuntimeError(
        "adapter did not apply with any loader (lora_B stayed 0) — "
        "base/adapter architecture mismatch; the gate would be meaningless")

def generate(model, prompt):
    msgs = [{{"role": "user", "content": prompt}}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").to(model.device)
    out = model.generate(**ids, max_new_tokens=cfg.get("max_new_tokens", 256), do_sample=False)
    return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)

def score(model):
    correct = 0
    for ex in ds:
        comp = generate(model, ex["prompt"])
        correct += numeric_correctness_reward(comp, ex["answer"])
    return correct / max(len(ds), 1)

base, ft = load_pair()
base_score = score(base)
cand_score = score(ft)

delta = cand_score - base_score
passed = cand_score >= base_score - cfg.get("tolerance", 0.0)
result = {{"base_score": base_score, "candidate_score": cand_score, "delta": delta,
          "passed": bool(passed), "verdict": "PASS" if passed else "REGRESSION", "n": len(ds)}}
Path(cfg["out"]).write_text(json.dumps(result, indent=2) + "\\n")
print("EVAL_GATE", json.dumps(result))
'''


def render_eval_gate() -> str:
    """The self-contained held-out eval-gate script (reward logic injected)."""
    return _EVAL_GATE_TEMPLATE.format(reward_source=_REWARD_SOURCE)


def write_eval_gate(
    repo: Path,
    name: str,
    *,
    base: str,
    adapter: str,
    holdout: str,
    tolerance: float = 0.0,
    max_new_tokens: int = 256,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Write the eval-gate config + script under runs/eval_gate/<name>/. Returns both."""
    run_dir = Path(repo) / "runs" / "eval_gate" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = run_dir / "config.json"
    script_path = run_dir / "eval_gate.py"
    if overwrite or not cfg_path.exists():
        cfg = {
            "base": base,
            "adapter": adapter,
            "holdout": holdout,
            "tolerance": tolerance,
            "max_new_tokens": max_new_tokens,
            "out": str(run_dir / "result.json"),
        }
        cfg_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    if overwrite or not script_path.exists():
        script_path.write_text(render_eval_gate(), encoding="utf-8")
    return cfg_path, script_path


def build_eval_gate_command(
    name: str,
    *,
    repo: str = "$HOME/projects/model-forge",
    models_dir: str = "$HOME/models",
    hf_home: str = "$HOME/.forgewright/hf_home",
    image: str = "model-forge-posttrain-tf5:latest",
    gpus: str = "all",
) -> str:
    """`docker run` to score base-vs-adapter in the posttrain container (launch via
    launch_job). The verdict + scores land in runs/eval_gate/<name>/result.json."""
    run_dir = f"runs/eval_gate/{name}"
    return (
        f"mkdir -p {hf_home} && docker run --rm --gpus {gpus} "
        f'--user "$(id -u):$(id -g)" -e HOME="$HOME" --shm-size=16g '
        f"-e HF_HOME={hf_home} -e HF_DATASETS_CACHE={hf_home}/datasets -e HF_HUB_DISABLE_XET=1 "
        f"-e XDG_CACHE_HOME={hf_home}/cache -e TRITON_CACHE_DIR={hf_home}/triton "
        f"-e TORCHINDUCTOR_CACHE_DIR={hf_home}/inductor "
        f"-v {repo}:{repo} -v {models_dir}:{models_dir} -v {hf_home}:{hf_home} -w {repo} "
        f"--entrypoint python3 {image} {run_dir}/eval_gate.py --config {run_dir}/config.json"
    )
