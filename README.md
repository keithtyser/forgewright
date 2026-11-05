# Forgewright

Forgewright plans and runs LLM post-training recipes through a conversational CLI. Python handles the agent loop, approvals, job records, and artifacts. A Node terminal UI displays progress. [model-forge](https://github.com/keithtyser/model-forge) runs the model operations.

This is an experimental project. The control plane has CPU tests on Windows and Linux. GPU recipes need a configured Linux host, model snapshots, and compatible containers. Remote recipes currently stop before launch because their file and execution paths are not yet consistent. Run the application on the GPU host instead.

## Install

Use Python 3.11–3.13 and Node 24. From a checkout:

```bash
python -m venv .venv
# Activate the environment, then:
python -m pip install --require-hashes -r requirements-dev.lock
python -m pip install --no-deps -e .
cd frontend
npm ci --ignore-scripts
npm install -g .
```

Python owns `forgewright`. The UI uses `forgewright-tui`, so the installations do not replace each other's commands. If the UI selects the wrong interpreter, set `FORGEWRIGHT_PYTHON` to the environment's Python executable.

Installing the control plane does not prepare model-forge or GPU dependencies. Set `FORGE_REPO` to a separate checkout at commit `324a5b487d578c1da837719bae3d1399eeb9f55b`. Prepare that checkout and its containers on the GPU host. The optional `local-train` package extra does not replace this setup.

## Run

```bash
forgewright-tui
python -m forgewright run "Inspect my local model and propose a quantization plan" --max-steps 20
```

Set `--brain` to a provider name from `providers.yaml` or a shorthand such as `vllm:model@http://localhost:8000/v1`. Copy [the example config](configs/providers.example.yaml) to `$FORGEWRIGHT_HOME/providers.yaml` to define providers.

Provider requests use at most three attempts by default. Authentication, permission and invalid-request errors stop at once. Transient errors use backoff and `Retry-After`; a requested wait over 60 seconds stops the turn. Cancellation interrupts retry waits and prevents later tool calls. An in-flight synchronous request still waits for its response or timeout. Wrapped errors from the experimental OAuth adapter stop without an automatic retry.

The experimental Codex adapter reads `$CODEX_HOME/auth.json`, falling back to `~/.codex/auth.json`. Log in with the official CLI first. See [SECURITY.md](SECURITY.md) for the direct OAuth adapter's limits.

Shell, job, system, and publishing tools ask for approval by default. “All” remembers the same tool and arguments for the session. `--yes` and the UI's YOLO choice bypass approval. They do not create a sandbox or limit spending.

## Recipes and checks

| Recipe | Stages |
|---|---|
| `uplift` | Curate → SFT → evaluate adapter |
| `task_grpo` | Train on verifiable tasks → evaluate adapter |
| `quantize_serve` | Quantize → evaluate model → tune serving |
| `abliterate` | Edit refusal directions → evaluate model |
| `uplift_publish` | Curate → SFT → evaluate → merge → evaluate → approved publication |
| `full` | Curate → SFT → evaluate → merge → evaluate → quantize → evaluate → tune serving → evaluate endpoint |

`full` adds abliteration only with `include_abliteration=true`. Quantization requires an explicit method or an inspected capability list. A method request alone is an explicit choice, not hardware validation.

Each transform returns an artifact with its source and gate result. Evaluation is a separate step. Missing baseline scores, failed evaluation jobs, stale files, or invalid results fail the gate. Serving optimization measures its baseline configuration, checks candidate quality, and restarts the winner. Publication supports standalone models and requires evaluation of that artifact, approval, and a completed upload receipt.

Recipe inputs can use `model_uri` or `dataset_uri`, or a registered `model_artifact_id` or `dataset_artifact_id`. Without an explicit input, supply `family` or `artifact_run_id`; exactly one accepted artifact must match. A failed latest evaluation excludes its model from lookup. Publication rejects missing lineage records or cycles.

Standalone model evaluation hashes the candidate's weights, config, tokenizer, and other files before and after evaluation. Publication checks that manifest again before approval, after approval, and after upload, and pins `--artifact-path` to the checked directory. Older evaluations without a manifest require a new evaluation. Use a stable local directory with at most 10,000 files; symlinks and Windows junctions are unsupported. Copy linked Hub snapshots to a regular directory first. Each check reads all files in chunks, so large checkpoints add disk I/O.

These checks detect persistent file changes; they do not lock files or create an immutable upload snapshot. If files change during upload, some files may already be remote. The result keeps the upload receipt and reports `UNVERIFIED`. Inspect the remote repository before trying again. Adapter evaluation, baseline files, and remote upload contents still need separate byte-level proof.

Adapter evaluation supports numeric `{prompt, answer}` JSONL tasks, with at least 32 evaluation examples. It does not measure general chat quality. Model evaluation uses model-forge's internal capability suite. A validation set used to guide repair is no longer an untouched final test; reserve a separate final test for release claims.

Custom holdout prompts are excluded during curation and checked before training. Training records data hashes. Licenses default to `unknown`; verify the actual source rights before publication. Distillation requires an explicit provider. Template output is simulation and cannot enter training.

## Execution limits

The agent step limit follows config (80 by default). `--max-steps 0` explicitly removes that limit. Wall time is checked between agent calls and does not stop detached GPU jobs. Legacy dollar and GPU-hour settings now raise an error because they were never enforced. Set those limits at the provider or host scheduler.

Generated training and adapter-evaluation commands use `FORGE_REPO`, `MODEL_FORGE_POSTTRAIN_IMAGE`, `MODEL_FORGE_MODELS_DIR`, `MODEL_FORGE_HF_HOME`, and `MODEL_FORGE_GPUS`. Container defaults remain development defaults. Pin an image digest and inspect device allocation before a real run. Quantization and serving use separate upstream launchers; a training device setting is not proof of isolation for those stages.

Serving cleanup checks the ownership label and uses the immutable Docker ID. An occupied port causes an error. This integration uses model-forge's Spark naming and label hooks; validate a different launcher before using it.

Job waits support interruption and attempt cancellation on timeout. SSH tools require a known host key. Check jobs and containers after a timeout or lost connection: an unreachable host or independent container may continue working.

## Recovery and records

Records live under `$FORGEWRIGHT_HOME` (default `~/.forgewright`):

- `ledger/` holds agent events and recipe receipts; `checkpoints/` holds conversation state.
- `registry/artifacts.jsonl` holds immutable artifact records and parent IDs. `runs/jobs/` holds job commands, logs, and exit status.
- `transcripts/` holds UI events and user messages. Common secrets and known environment credentials are redacted; inspect files before sharing them.

Resume with `forgewright run --resume RUN_ID "goal"`. A recipe receipt marked `active`, `failed`, or `interrupted` blocks automatic replay. Inspect the stage, jobs, outputs, and containers before choosing a fresh run. Do not delete the receipt merely to bypass the check. Automatic reconciliation of unfinished jobs remains future work.

A completed recipe can reuse its result only when its saved settings and local file proofs still match. Missing, changed, unreadable, or unrecorded files block reuse without rerunning stages. Older receipts also need review. Saved endpoint and publication receipts require a fresh remote check; a past receipt cannot prove their current state.

## Terminal controls

Use `/graph`, `/models`, `/brain`, `/login`, `/help`, and `/quit`. Messages sent during a turn queue for the next turn. Ctrl-C requests an interrupt; it cannot cancel every external operation immediately.

Set `NO_COLOR=1` for plain color output, `FORGEWRIGHT_REDUCED_MOTION=1` for slower animation, or `FORGEWRIGHT_PLAIN=1` for compact status. Full-screen mode is optional with `--fullscreen`. Approval messages show full arguments and use request IDs.

## Development and releases

See [CONTRIBUTING.md](CONTRIBUTING.md) for checks, [CHANGELOG.md](CHANGELOG.md) for changes, and [docs/maintenance.md](docs/maintenance.md) for remaining work. Mocked tests do not establish GPU correctness. Record a GPU smoke on each claimed configuration before releasing it.

## Credits

- [terminal-kit](https://www.terminal-kit.com/) by Cedric Ronvel (MIT), used by the terminal UI.
- [model-forge](https://github.com/keithtyser/model-forge), the model operations engine.
- [LiteLLM](https://github.com/BerriAI/litellm), the provider interface.
