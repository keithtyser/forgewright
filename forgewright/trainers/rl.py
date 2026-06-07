"""GRPO/RLVR trainer — the fine-tune skill's *task* mode (verifiable-reward RL).

model-forge has no RL, so this is genuinely new. It builds a self-contained TRL
``GRPOTrainer`` script that runs in the posttrain GPU container (same vehicle as
uplift SFT), with the user's hard-won RL scars baked into the defaults:

- **KL anchor** (``beta`` > 0) — round-1 GRPO collapsed to 0.00 with an unbounded
  negative-advantage term; the KL anchor to the reference keeps it bounded.
- **PPO clip** (``epsilon`` / ``epsilon_high``) — clip the importance ratio.
- **dynamic sampling / overlong masking** (``mask_truncated_completions`` + reward
  scaling) — DAPO-style; don't learn from truncated or zero-variance groups.

The reward logic (verifiable correctness + ``<think>`` format) is pure + unit-tested
here, then injected verbatim into the generated trainer (single source of truth, while
the trainer stays self-contained so it needs nothing but the container's libs).
"""
from __future__ import annotations

import inspect
import json
import re
from pathlib import Path
from typing import Optional

# --- pure, unit-tested reward logic (also injected into the generated trainer) ------


def extract_final_number(text: str) -> Optional[str]:
    """The model's final numeric answer: prefer \\boxed{..}, else the last number."""
    boxed = re.findall(r"\\boxed\{\s*(-?\d+(?:\.\d+)?)\s*\}", text)
    if boxed:
        return boxed[-1]
    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return nums[-1] if nums else None


def numeric_correctness_reward(completion: str, answer: str) -> float:
    """1.0 if the completion's final number equals the gold answer, else 0.0."""
    pred, gold = extract_final_number(completion), extract_final_number(answer)
    if pred is None or gold is None:
        return 0.0
    try:
        return 1.0 if abs(float(pred) - float(gold)) < 1e-6 else 0.0
    except ValueError:
        return 0.0


def has_closed_think(text: str) -> bool:
    """A single, properly-ordered, closed ``<think>`` block."""
    o, c = text.find("<think>"), text.find("</think>")
    return o != -1 and c != -1 and o < c


def think_format_reward(completion: str) -> float:
    """Small shaping reward for the strict ``<think>`` format (don't swamp correctness)."""
    return 0.2 if has_closed_think(completion) else 0.0


# Source of the three helpers, injected into the generated trainer so it is standalone.
_REWARD_SOURCE = "\n\n".join(
    inspect.getsource(fn)
    for fn in (extract_final_number, numeric_correctness_reward, has_closed_think, think_format_reward)
)


# --- scar defaults -----------------------------------------------------------------


def grpo_scar_defaults() -> dict:
    """The KL-anchored, PPO-clipped, overlong-masked defaults (the user's RL scars)."""
    return {
        "beta": 0.04,                     # KL anchor to the reference (was the 0.00-collapse fix)
        "epsilon": 0.2,                   # PPO clip low
        "epsilon_high": 0.28,             # DAPO clip-higher
        "scale_rewards": "group",         # normalize advantages within a group
        "loss_type": "dapo",              # length-unbiased DAPO loss
        "mask_truncated_completions": True,  # don't learn from overlong/truncated rollouts
        "num_generations": 4,
        "num_iterations": 1,
        "temperature": 1.0,
        "top_p": 1.0,
        "learning_rate": 1e-6,            # RL LR is far lower than SFT
        "max_completion_length": 256,
    }


# --- the generated, self-contained TRL GRPO trainer --------------------------------

_GRPO_TRAINER_TEMPLATE = '''\
"""Forgewright GRPO/RLVR trainer (generated; runs inside model-forge-posttrain-tf5)."""
import json, os, re, sys
from pathlib import Path
from typing import Optional

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

plan = json.loads(Path(sys.argv[sys.argv.index("--plan") + 1]).read_text())
out_dir = os.path.expanduser(plan["model"]["output_dir"])  # ~ is not expanded by TRL

{reward_source}

tok = AutoTokenizer.from_pretrained(plan["model"]["source"])
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

def _format(ex):
    msgs = [{{"role": "user", "content": ex["prompt"]}}]
    return {{"prompt": tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True),
             "answer": ex["answer"]}}

ds = load_dataset("json", data_files=plan["data"]["path"], split="train").map(_format)

def reward_correct(completions, answer, **kwargs):
    return [numeric_correctness_reward(c, a) for c, a in zip(completions, answer)]

def reward_format(completions, **kwargs):
    return [think_format_reward(c) for c in completions]

h = plan["grpo"]
args = GRPOConfig(
    output_dir=out_dir,
    per_device_train_batch_size=h["num_generations"],
    gradient_accumulation_steps=1,
    max_steps=h["max_steps"],
    logging_steps=1, save_steps=h["max_steps"], save_total_limit=1,
    bf16=True, report_to="none", seed=3407,
    beta=h["beta"], epsilon=h["epsilon"], epsilon_high=h["epsilon_high"],
    scale_rewards=h["scale_rewards"], loss_type=h["loss_type"],
    mask_truncated_completions=h["mask_truncated_completions"],
    num_generations=h["num_generations"], num_iterations=h["num_iterations"],
    temperature=h["temperature"], top_p=h["top_p"],
    learning_rate=h["learning_rate"], max_completion_length=h["max_completion_length"],
    use_vllm=False,
)
peft = LoraConfig(r=plan["lora"]["r"], lora_alpha=plan["lora"]["alpha"], lora_dropout=0.0,
                  target_modules=plan["lora"]["target_modules"], task_type="CAUSAL_LM")
trainer = GRPOTrainer(model=plan["model"]["source"], reward_funcs=[reward_correct, reward_format],
                      args=args, train_dataset=ds, peft_config=peft)
trainer.train()
trainer.save_model(out_dir)
print("GRPO DONE ->", out_dir)
'''


def render_grpo_trainer() -> str:
    """The self-contained GRPO trainer script (reward helpers injected)."""
    return _GRPO_TRAINER_TEMPLATE.format(reward_source=_REWARD_SOURCE)


def build_grpo_plan(
    name: str,
    *,
    source: str,
    data_path: str,
    output_dir: Optional[str] = None,
    max_steps: int = 30,
    lora_r: int = 16,
    lora_alpha: int = 32,
    **overrides,
) -> dict:
    """Assemble the GRPO plan dict (scar defaults + any overrides)."""
    grpo = grpo_scar_defaults()
    grpo["max_steps"] = max_steps
    grpo.update({k: v for k, v in overrides.items() if k in grpo})
    stem = source.rstrip("/").split("/")[-1]
    return {
        "name": name,
        "model": {"source": source, "output_dir": output_dir or f"~/models/{stem}-{name}"},
        "data": {"path": data_path},
        "lora": {"r": lora_r, "alpha": lora_alpha,
                 "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                                    "gate_proj", "up_proj", "down_proj"]},
        "grpo": grpo,
    }


def write_grpo_run(
    repo: Path,
    name: str,
    *,
    source: str,
    data_path: str,
    overwrite: bool = False,
    **kwargs,
) -> tuple[Path, Path]:
    """Write the GRPO plan.json + trainer script under runs/rl/<name>/. Returns both."""
    run_dir = Path(repo) / "runs" / "rl" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    plan_path = run_dir / "plan.json"
    trainer_path = run_dir / "train_grpo.py"
    if overwrite or not plan_path.exists():
        plan = build_grpo_plan(name, source=source, data_path=data_path, **kwargs)
        plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    if overwrite or not trainer_path.exists():
        trainer_path.write_text(render_grpo_trainer(), encoding="utf-8")
    return plan_path, trainer_path


def build_grpo_train_command(
    name: str,
    *,
    repo: str = "$HOME/projects/model-forge",
    models_dir: str = "$HOME/models",
    hf_home: str = "$HOME/.forgewright/hf_home",
    image: str = "model-forge-posttrain-tf5:latest",
    gpus: str = "all",
) -> str:
    """`docker run` to train the GRPO run inside the posttrain GPU container (detached
    via launch_job), same vehicle/mounts/HF-cache fix as uplift SFT."""
    run_dir = f"runs/rl/{name}"
    return (
        f"mkdir -p {hf_home} && docker run --rm --gpus {gpus} "
        f'--user "$(id -u):$(id -g)" -e HOME="$HOME" --shm-size=16g '
        f"-e HF_HOME={hf_home} -e HF_DATASETS_CACHE={hf_home}/datasets -e HF_HUB_DISABLE_XET=1 "
        f"-v {repo}:{repo} -v {models_dir}:{models_dir} -v {hf_home}:{hf_home} -w {repo} "
        f"--entrypoint python3 {image} {run_dir}/train_grpo.py --plan {run_dir}/plan.json"
    )
