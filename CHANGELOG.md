# Changelog

## Unreleased

- Bind quantization to its input checkpoint and require fresh output weights.
- Require current evaluation results, valid baselines, and evaluation after model transforms.
- Evaluate the selected endpoint in place; restart and verify the winning serving configuration.
- Restrict cleanup to owned containers and prevent replacement of an occupied serving port.
- Require publication evidence and a completed upload receipt. Adapter publication requires a merge first.
- Filter custom holdouts, recheck training overlap, record data hashes, and use the full curated dataset by default.
- Correct numeric fraction scoring and reject empty or zero-signal evaluation passes.
- Record recipe stages before execution and stop replay of interrupted work pending reconciliation.
- Harden SSH status, cancellation, approval messages, credentials, context limits, and terminal output.
- Add hashed dependency locks, CI, maintenance instructions, and vulnerability checks.
- Keep `marked-terminal` 7.3.1 and `marked` 12.0.2: the tested major upgrade introduced vulnerable transitive dependencies.

Compatibility changes: remote recipes now stop before launch; run on the GPU host. Quantization needs an explicit method or capability list. Abliteration in `full` requires `include_abliteration=true`. The TUI command is `forgewright-tui`; Python owns `forgewright`. Legacy dollar/GPU-hour cap settings now raise an error because they were never enforced. The default agent step limit now follows the config. Use `--max-steps 0` only for an explicit unlimited run.
