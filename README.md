# Forgewright

> A multi-agent swarm for **autonomous LLM post-training**. You talk to one
> Claude-Code-style conversational CLI; behind it, a swarm of specialist agents
> (data curation, fine-tuning, abliteration, quantization, serving optimization,
> evaluation, publishing) hands typed artifacts to each other and does the work.

Point it at a model and a goal, give it GPUs (local or over SSH), and it plans and runs
the whole post-training pipeline. The swarm is invisible: you chat, watch progress stream
in, and approve outward actions at a single prompt, just like a coding agent.

```bash
forgewright "Curate a reasoning dataset, fine-tune Qwen3.5-0.8B on it,
  NVFP4-quantize for my GB10, optimize tok/s without losing quality, eval, and publish"
```

## The swarm

Post-training is a chain of typed handoffs, not one blob:

```
goal -> dataset -> adapter -> (abliterated) model -> quantized model
     -> tuned serving config -> eval report -> published repo
```

Each stage is its own specialist agent with a focused prompt, a small tool set, and one
gate it is responsible for. A **Director** plans the recipe, dispatches specialists,
passes each produced artifact to the next, enforces the global gate (a regression never
flows downstream), and surfaces everything in one transcript.

| Specialist | consumes -> produces |
|---|---|
| DataCurator | goal -> dataset (curate seeds, or teacher-distillation) |
| SFTTrainer | dataset -> adapter (uplift / distillation SFT) |
| RLTrainer | task spec -> adapter (GRPO / RLVR, KL-anchored) |
| Abliterator | model -> model (refusal-direction ablation) |
| Quantizer | model -> model (NVFP4 / FP8) |
| ServingOptimizer | model -> tuned serving config |
| Evaluator | adapter / model -> eval report (the gate authority) |
| Publisher | any -> published HF repo (human-gated) |

Specialists never call each other directly. A producer registers a typed `Artifact` (a
handle to the bytes plus provenance and its gate result); the consumer pulls it from the
**registry**, which doubles as the provenance graph (`register / get / latest / lineage`).

## Capabilities (proven on a DGX Spark GB10, sm_121)

- **Quantize**: NVFP4 (NVIDIA ModelOpt) with a speedup-based gate (no static floor).
- **Fine-tune**: uplift distillation SFT (assistant-only loss, conservative LR, strict
  `<think>` and holdout hygiene) and task-mode GRPO/RLVR with the user's RL scars baked in
  (KL anchor, PPO clip, DAPO overlong masking).
- **Abliterate**: contrastive refusal-direction projection with a dual gate (refusal must
  drop and capability must hold).
- **Serving-opt**: latency or throughput objective, speculative decoding plus batching,
  quality-gated against the source.
- **Eval-gate**: held-out verifiable scoring with a robust adapter loader that fails loud
  if an adapter does not actually apply.

## Architecture

A light, cross-platform shared runtime (agent loop, [LiteLLM](https://github.com/BerriAI/litellm)
brain, tools, SSH, job manager, ledger, permissions) plus the swarm layer
(`contracts`, `registry`, `agents`, the Director). It calls
[`model-forge`](https://github.com/keithtyser/model-forge) for the mature ML-ops
primitives and runs heavy GPU stages inside model-forge's posttrain container, so the same
harness drives a Blackwell fleet over SSH.

### Frontend

One conversational CLI. The default rich UI is a [terminal-kit](https://www.terminal-kit.com/)
Node TUI (see Credits) talking to the Python backend over a newline-delimited JSON event
stream, so the swarm stays entirely on the backend. A built-in Python REPL is the headless
fallback and is used in tests.

## Install

```bash
# Control plane (runs anywhere you type, including Windows):
pip install -e .
```

Heavy training, quantization, and abliteration run inside model-forge's GPU container on a
Linux Blackwell box; Forgewright orchestrates them locally or over SSH.

## Status

The four core capabilities and the swarm backend are built and proven end to end on real
hardware. The full three-stage swarm (DataCurator, SFTTrainer, Evaluator) runs through the
Director with a single provenance chain and one role-tagged transcript. The terminal-kit
Node TUI is the remaining frontend piece. See `docs/` and the plan file for the slice
breakdown.

## Credits

- [terminal-kit](https://www.terminal-kit.com/) by Cedric Ronvel (MIT), the terminal UI
  library used for the frontend.
- [model-forge](https://github.com/keithtyser/model-forge), the ML-ops engine.
- [LiteLLM](https://github.com/BerriAI/litellm), the model-agnostic brain.
