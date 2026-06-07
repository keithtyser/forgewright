"""Fine-tune skill: the core capability, in two modes.

- **uplift** (Jackrong pattern): teacher-distillation LoRA SFT for broad reasoning +
  formatting/style uplift. Assistant-only loss (train_on_responses_only), strict
  ``<think>`` formatting, conservative LR + holdout hygiene. Driven through
  model-forge's finetune engine (``forge finetune``), which already supplies the data
  hygiene we want (assistant_only_loss masking, reject_unclosed_think, eval-overlap
  rejection, dedup, a ResourceGuard) — Forgewright supplies the mode selection, the
  scar-baked config/manifest scaffolding, and the eval gate.
- **task** (ml-intern pattern): research -> synth data -> GRPO/RLVR -> eval ->
  self-correct, for a verifiable target metric. Uses Forgewright's own RL trainer
  (``trainers/rl.py``: KL-anchor + PPO-clip + dynamic sampling) since model-forge has
  no RL — built/proven separately.

Mode selection + config scaffolding (scar defaults) are pure + unit-tested. The actual
training is launched as a DETACHED job on the GPU box and eval-gated vs the base model.

Training scars baked into the defaults (the user's hard-won lessons):
- conservative LR (repetition/format "gravity" collapse mid-train at higher LR),
- assistant-only loss + strict ``<think>`` closure (format-degeneration guard),
- holdout/eval-overlap rejection (no train/test contamination).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

from forgewright.tools.base import Tool, ToolResult
from forgewright.tools.forge import ForgeRunner

Mode = Literal["uplift", "task"]

# Goal phrasing that implies a *verifiable, measurable* target -> RL/RLVR (task mode).
_TASK_SIGNALS = re.compile(
    r"\b(accuracy|pass@\d|pass rate|score|benchmark|solve|verifiab\w+|reward|"
    r"rlvr|grpo|rl\b|exact match|unit test|metric|beat \d|to \d+%|achieve \d)",
    re.IGNORECASE,
)
# Goal phrasing that implies broad capability/style uplift -> distillation SFT (uplift mode).
_UPLIFT_SIGNALS = re.compile(
    r"\b(distill\w*|teacher|uplift|general\w*|reasoning|style|format\w*|broad\w*|"
    r"sft|instruct\w*|chat quality|persona|tone)\b",
    re.IGNORECASE,
)


def select_mode(goal: str) -> Mode:
    """Pick uplift (broad distillation SFT) vs task (verifiable-reward RL).

    A verifiable/measurable target wins (RL is the lever when there's a checkable
    signal); otherwise broad uplift via distillation SFT, which is the safe default."""
    task = bool(_TASK_SIGNALS.search(goal))
    uplift = bool(_UPLIFT_SIGNALS.search(goal))
    if task and not uplift:
        return "task"
    if uplift and not task:
        return "uplift"
    # both or neither: a concrete verifiable target is the stronger signal for RL
    return "task" if task else "uplift"


# --- uplift (distillation SFT) scaffolding -----------------------------------------
# Only keys model-forge's finetune engine reads. backend=hf_causal_lm because the GB10
# (ARM + Blackwell sm_121) has no Unsloth; the HF/PEFT path is the portable trainer.

UPLIFT_CONFIG_TEMPLATE = """\
name: {name}
family: {family}
run_dir: runs/finetune/{name}
dry_run_only: {dry_run_only}

model:
  source: {source}
  local_dir: {local_dir}
  output_dir: {output_dir}
  served_model_name: local/{name}
  trust_remote_code: {trust_remote_code}
  max_seq_length: {max_seq_length}

trainer:
  backend: hf_causal_lm
  method: qlora
  device_map: single_gpu
  load_in_4bit: true
  attn_implementation: eager
  bf16: true
  gradient_checkpointing: true
  group_by_length: true
  assistant_only_loss: true        # train_on_responses_only — mask the prompt
  per_device_train_batch_size: {batch_size}
  gradient_accumulation_steps: {grad_accum}
  learning_rate: {learning_rate}   # conservative — guards the repetition/format gravity collapse
  num_train_epochs: {epochs}
  max_steps: {max_steps}
  warmup_ratio: 0.03
  lr_scheduler_type: cosine
  optim: paged_adamw_8bit
  weight_decay: 0.001
  logging_steps: 1
  save_steps: {save_steps}
  save_total_limit: 1
  seed: 3407
  report_to: none

lora:
  r: {lora_r}
  alpha: {lora_alpha}
  dropout: 0.0
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
  modules_to_save: []

data:
  manifest: datasets/finetuning/{name}.yaml

baseline:
  target_to_beat: {target_to_beat}
  strategy_delta:
    - Distillation SFT for broad reasoning + <think> formatting uplift.
    - Eval-gate the adapter vs the base model; capability must not regress.
"""

# Manifest: messages format with the <think>/holdout hygiene gates ON.
UPLIFT_MANIFEST_TEMPLATE = """\
name: {name}
description: "Forgewright uplift (distillation SFT) data for {family}."
format: messages
chat_template: auto
max_context_window: {max_seq_length}
random_seed: 3407

quality_gates:
  min_turn_chars: 2
  require_assistant_content: true
  reject_unclosed_think: true          # strict <think> formatting (Jackrong)
  dedupe_by_conversation_hash: true
  max_tokens: {max_seq_length}
  reject_eval_prompt_overlap: true     # no train/eval contamination

sources:
  - id: {name}_distill
    name: {name}_distill
    path: {data_path}
    target_samples: {target_samples}
    role: teacher_distillation

holdouts:
  - evals/prompts/capability_preservation_challenge.yaml
  - evals/prompts/normal_use_regression.yaml
  - evals/prompts/reasoning_style_stability.yaml
"""

FINETUNE_RUNBOOK = """\
Fine-tune runbook (uplift / distillation SFT; DGX Spark / GB10):
 1. select_mode(goal): verifiable target metric -> task (RL); broad reasoning/style -> uplift (SFT).
 2. uplift: scaffold_finetune_config <family> --source <hf_model> --data-path <distill.jsonl>
      (writes configs/finetuning/<name>.yaml + datasets/finetuning/<name>.yaml with scar defaults:
       assistant_only_loss, conservative LR, <think> + holdout hygiene; backend hf_causal_lm — no Unsloth on GB10).
 3. forge: finetune --config configs/finetuning/<name>.yaml plan        (dry sanity check of the plan).
 4. forge: finetune --config configs/finetuning/<name>.yaml prepare     (builds + hygiene-filters the dataset).
 5. launch_job: bash forge finetune --config configs/finetuning/<name>.yaml run   (DETACHED; poll to done).
 6. Eval-gate: serve the merged/adaptered model -> forge eval <family> <variant> --internal, and compare the
      capability_preservation_challenge pass rate vs the BASE model. Capability must not regress; report the delta.
 7. Watch the loss/grad logs for repetition/format degeneration (the gravity collapse); if it degenerates,
      lower LR and re-run from the last good checkpoint.
Teacher-distillation data: generate with model_forge.data.factory (teacher = a strong model), format each
turn with a closed <think> reasoning block, then point the manifest source `path` at the JSONL.
For a verifiable target (task mode): build the GRPO run via trainers/rl.py (KL-anchor + PPO-clip + dynamic
sampling) — do NOT use a plain SFT loop for an RL objective.
"""


def scaffold_uplift_config(
    family: str,
    *,
    name: str | None = None,
    source: str = "Qwen/Qwen3.5-0.8B",
    local_dir: str | None = None,
    output_dir: str | None = None,
    max_seq_length: int = 4096,
    batch_size: int = 1,
    grad_accum: int = 8,
    learning_rate: float = 8e-5,
    epochs: int = 1,
    max_steps: int = 60,
    save_steps: int = 60,
    lora_r: int = 16,
    lora_alpha: int = 32,
    trust_remote_code: bool = True,
    dry_run_only: bool = False,
    target_to_beat: str = "self/base",
) -> str:
    """Render a model-forge finetune config for an uplift (distillation SFT) run."""
    name = name or f"{family}_uplift_v0"
    stem = source.rstrip("/").split("/")[-1]
    return UPLIFT_CONFIG_TEMPLATE.format(
        name=name,
        family=family,
        source=source,
        local_dir=local_dir or f"~/models/{stem}",
        output_dir=output_dir or f"~/models/{stem}-{name}",
        max_seq_length=max_seq_length,
        batch_size=batch_size,
        grad_accum=grad_accum,
        learning_rate=learning_rate,
        epochs=epochs,
        max_steps=max_steps,
        save_steps=save_steps,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        trust_remote_code=str(trust_remote_code).lower(),
        dry_run_only=str(dry_run_only).lower(),
        target_to_beat=target_to_beat,
    )


def scaffold_uplift_manifest(
    family: str,
    *,
    name: str | None = None,
    data_path: str = "datasets/finetuning/distill_smoke.jsonl",
    target_samples: int = 64,
    max_seq_length: int = 4096,
) -> str:
    """Render the dataset manifest (messages format, <think>/holdout hygiene on)."""
    name = name or f"{family}_uplift_v0"
    return UPLIFT_MANIFEST_TEMPLATE.format(
        name=name,
        family=family,
        data_path=data_path,
        target_samples=target_samples,
        max_seq_length=max_seq_length,
    )


def write_finetune_config(
    repo: Path,
    family: str,
    *,
    name: str | None = None,
    overwrite: bool = False,
    data_path: str = "datasets/finetuning/distill_smoke.jsonl",
    target_samples: int = 64,
    **kwargs,
) -> tuple[Path, Path]:
    """Write config + dataset manifest into model-forge (idempotent). Returns both paths."""
    name = name or f"{family}_uplift_v0"
    max_seq_length = int(kwargs.get("max_seq_length", 4096))
    cfg = Path(repo) / "configs" / "finetuning" / f"{name}.yaml"
    man = Path(repo) / "datasets" / "finetuning" / f"{name}.yaml"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    man.parent.mkdir(parents=True, exist_ok=True)
    if overwrite or not cfg.exists():
        cfg.write_text(scaffold_uplift_config(family, name=name, **kwargs), encoding="utf-8")
    if overwrite or not man.exists():
        man.write_text(
            scaffold_uplift_manifest(
                family, name=name, data_path=data_path,
                target_samples=target_samples, max_seq_length=max_seq_length,
            ),
            encoding="utf-8",
        )
    return cfg, man


class ScaffoldFinetuneConfigTool(Tool):
    name = "scaffold_finetune_config"
    description = (
        "Generate a model-forge fine-tune config + dataset manifest for an UPLIFT (distillation SFT) "
        "run, with Forgewright's scar defaults baked in (assistant_only_loss/train_on_responses_only, "
        "conservative LR, strict <think> + holdout-overlap hygiene, backend hf_causal_lm since the GB10 "
        "has no Unsloth). Writes configs/finetuning/<name>.yaml and datasets/finetuning/<name>.yaml. "
        "Point data_path at a teacher-distillation JSONL (messages format). Then drive it with the "
        "`forge` tool (finetune --config <cfg> plan|prepare) and launch_job for the run."
    )
    risk = "write"
    parameters = {
        "type": "object",
        "properties": {
            "family": {"type": "string", "description": "model family id, e.g. qwen35_0_8b"},
            "source": {"type": "string", "description": "HF model id (default Qwen/Qwen3.5-0.8B)"},
            "name": {"type": "string", "description": "run name (default <family>_uplift_v0)"},
            "data_path": {"type": "string", "description": "distillation JSONL path (messages format)"},
            "max_steps": {"type": "integer", "description": "training steps (default 60)"},
            "learning_rate": {"type": "number", "description": "default 8e-5 (conservative)"},
            "dry_run_only": {"type": "boolean"},
            "overwrite": {"type": "boolean"},
        },
        "required": ["family"],
    }

    def __init__(self, runner: ForgeRunner | None = None) -> None:
        self.runner = runner or ForgeRunner()

    def run(self, family: str, overwrite: bool = False, **kwargs: Any) -> ToolResult:
        if not self.runner.available():
            return ToolResult(False, f"model-forge not found at {self.runner.repo}")
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        data_path = kwargs.pop("data_path", "datasets/finetuning/distill_smoke.jsonl")
        try:
            cfg, man = write_finetune_config(
                self.runner.repo, family, overwrite=overwrite, data_path=data_path, **kwargs
            )
        except Exception as e:  # noqa: BLE001
            return ToolResult(False, f"scaffold failed: {e}")
        mode = select_mode(kwargs.get("name", "") or family)
        return ToolResult(
            True,
            f"uplift finetune config: {cfg}\ndataset manifest: {man}\n"
            f"next: forge finetune --config configs/finetuning/{cfg.stem}.yaml plan",
            {"config": str(cfg), "manifest": str(man), "mode": mode},
        )
