# Forgewright

> An agentic CLI harness for **autonomous LLM post-training** — in the lineage of Claude Code / Codex, but its domain is fine-tuning, abliteration, quantization, and serving optimization.

Point it at a model and a goal, give it GPUs (local or over SSH), and it plans and
executes the whole job autonomously — writing code, launching jobs, recovering from
OOM/divergence, evaluating, iterating, and publishing — without babysitting.

```bash
forgewright "Take Qwen3.5-9B, abliterate it, NVFP4-quantize for my 5090, \
  maximize tok/s without losing quality, eval, and publish to HF" \
  --hardware ssh://gpu-box-1,local  --brain local:vllm/qwen3.5-coder-30b
```

## Capabilities (target)

- **Fine-tune** — *uplift* (teacher-distillation SFT, Jackrong-style: Unsloth+TRL,
  `train_on_responses_only`, `<think>` formatting) and *task-specific* (research →
  synth data → GRPO/RLVR → eval → self-correct).
- **Abliterate** — refusal-direction ablation + DPO healing, capability-gated.
- **Quantize** — NVFP4 / FP8 / AWQ / GGUF (Blackwell SM120/121 native).
- **Serving-opt** — auto-sweep engines + speculative decoding → tok/s × quality Pareto.
- **Eval & publish** — capability + refusal + tok/s gates; weights + datasets + card.

## Architecture

A light, cross-platform **control plane** (agent loop + [LiteLLM](https://github.com/BerriAI/litellm)
brain + tools + SSH) that dispatches heavy work to GPU boxes, calling
[`model-forge`](https://github.com/keithtyser/model-forge) as a library for the mature
ML-ops primitives. See the design plan in `docs/` for the full breakdown.

## Install

```bash
# Control plane only (runs anywhere you type — incl. Windows):
pip install -e .

# On a Linux GPU box, to run training/quant locally too:
pip install -e ".[local-train]"
# unsloth: install separately with the Blackwell-appropriate wheel (see docs).
```

## Status

🚧 Early build. **Slice 0** (core agent: loop, brain, tools, SSH, job lifecycle,
ledger, permissions) in progress.
