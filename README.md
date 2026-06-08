# Forgewright

> A multi-agent swarm for **autonomous LLM post-training**, behind one conversational CLI.

Point it at a model and a goal, give it GPUs (local or over SSH), and it plans and runs the
whole pipeline: data curation, fine-tuning, abliteration, quantization, serving optimization,
eval, and publishing. The swarm is invisible: you chat, watch progress stream in, and approve
outward actions at a single prompt, like a coding agent.

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

The Director runs named **recipes** (chains over the roster, e.g. `uplift`, `task_grpo`,
`quantize_serve`, `uplift_publish`, `abliterate`). If a stage's gate fails, the Director
halts and runs **saga compensations** in reverse, so a regression never flows downstream
and side effects (like a live served endpoint) get torn down.

From the chat, the conversational agent dispatches the swarm through the `run_recipe` tool:
you state a goal, the agent picks the recipe + params and hands it to the Director, and the
specialist progress plus any approval surface in the one transcript.

## Capabilities

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

## Install and run

```bash
# 1. Python backend (on the GPU box, in a venv):
pip install -e .

# 2. The `forgewright` command (the Node TUI):
cd frontend && npm install && npm install -g .   # use a user npm prefix if global is root-owned

# 3. Launch the chat:
forgewright
```

On first launch `forgewright` runs a short setup wizard: pick **OpenRouter** (paste an API
key) or **Codex** (ChatGPT login), and the choice is saved to
`~/.forgewright/credentials.json` so you are never asked again. It then starts the
conversational TUI and spawns the Python backend (`python -m forgewright serve`) behind it.
If the backend venv is not your default `python3`, point the TUI at it:
`export FORGEWRIGHT_PYTHON=/path/to/venv/bin/python`.

In-chat slash commands:

| command | what it does |
|---|---|
| `/graph` | draw the session's provenance DAG (dataset → adapter → model → eval), each node colored by its producer with its gate score |
| `/models` | list the models the current brain can reach (Codex probes the live token; falls back to a curated list) |
| `/login` | reconfigure or refresh your brain (new OpenRouter key / re-run Codex login); restarts the backend with the new credentials |
| `/brain` | show the brain currently in use |
| `/help` | list commands |
| `/quit` | exit (or Ctrl-C) |

Approvals use a vertical menu: **up/down to choose, enter to confirm** (approve once / approve
all of this tool / YOLO bypass-all / deny).

The agent can also **configure your environment** to unblock work -- install or upgrade
packages, drivers, and dependencies (pip/uv/conda/apt, CUDA, docker) via the `configure_env`
tool. These are system-modifying, so they carry a `system` risk tier and are approval-gated:
you see the exact command and where it runs (local / a host / the container) before anything
changes. The agent prefers the least-invasive scoped change and retries the blocked step after
a fix, rather than giving up.

### Brains

You can also pick a brain explicitly with `--brain` (this overrides the saved setup). Any
LiteLLM-supported backend works, plus a Codex (ChatGPT login) tap:

```bash
forgewright --brain openrouter:deepseek/deepseek-v4-pro   # hosted, key in OPENROUTER_API_KEY
forgewright --brain anthropic:claude-opus-4-8             # key in ANTHROPIC_API_KEY
forgewright --brain openai:gpt-5.1                        # key in OPENAI_API_KEY
forgewright --brain vllm:qwen3.5-coder-30b@http://host:8000/v1   # local / self-hosted
forgewright --brain oauth-codex:gpt-5.5-codex             # ChatGPT-login Codex, explicit model
```

`oauth-codex` reuses the credentials the official Codex CLI writes to `~/.codex/auth.json`,
so run `codex login` (ChatGPT account) once first. Forgewright refreshes the access token
automatically and talks to OpenAI's Codex Responses API. When you pick Codex in the setup
wizard it offers a model menu (gpt-5.5-codex, gpt-5.5, gpt-5.1-codex-max, gpt-5.1-codex,
gpt-5.1, gpt-5-codex, gpt-5, gpt-5-mini) plus a free-text entry for any newer id; or pass
`--brain oauth-codex:<model>` directly. The Claude subscription tap is intentionally not
supported (use an `anthropic:` API key instead).

Backend-only (no Node UI):

```bash
forgewright            # python entry: built-in REPL (also: `python -m forgewright`)
forgewright "<goal>" --yes    # one-shot, unattended
python -m forgewright serve   # the newline-JSON event server the TUI spawns
```

Heavy training, quantization, and abliteration run inside model-forge's GPU container on a
Linux Blackwell box; Forgewright orchestrates them locally or over SSH.

## Live HUD

While the swarm works, a panel hovers above the prompt showing the pipeline (each
specialist's stage state), the active specialist, training step/loss/reward with sparklines,
a guardrails gauge (the governor's step/GPU-hour/cost/wall caps with a live step bar),
elapsed, and tokens. Training telemetry is structured: the backend taps trainer output into
typed `metric` events the HUD consumes directly. Produced artifacts stream into the transcript
as lineage badges (`◇ eval#def456 ← adapter#xyz789  score 0.94 ✓`), and `/graph` draws the
full session provenance DAG. Set `FORGEWRIGHT_PLAIN=1` for a minimal one-line status instead.

## Traces

Every session is recorded two ways under `$FORGEWRIGHT_HOME` (default `~/.forgewright`):

- `ledger/<run_id>.jsonl` — the agent-loop view (goal, each assistant turn + tool calls/
  results, token usage), written by the loop.
- `transcripts/<run_id>.jsonl` — the complete, replayable session: a `meta` header (run id,
  brain, ledger path), then every inbound message (your turns, approval decisions, slash
  commands) and every outbound event (assistant text, tool calls, pipeline/stage/artifact/
  metric, approvals), each tagged with a wall-clock time and direction (`in`/`out`).

The transcript is a full agentic trajectory (goal to outcome), so it doubles as review
material for improving the harness and as a source for distillation/RL training data later.

## Credits

- [terminal-kit](https://www.terminal-kit.com/) by Cedric Ronvel (MIT), the terminal UI
  library used for the frontend.
- [model-forge](https://github.com/keithtyser/model-forge), the ML-ops engine.
- [LiteLLM](https://github.com/BerriAI/litellm), the model-agnostic brain.
