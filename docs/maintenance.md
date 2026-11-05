# Maintenance work

This checklist follows the 31 findings from the November 5, 2026 review of
`79add220a047950a3f633a5ea14ed4768bfa4483`. It separates code fixes from live proof.
CPU tests use fake services. They do not establish that a GPU recipe works.

## Work in this change

“Fixed” means the reported defect has a code fix and a local check. “Partial” means
the change reduces the risk but leaves part of the finding open. “Blocked” means
the unsupported path now stops before launch.

| ID | Finding | Status and evidence |
|---|---|---|
| 01 | Wrong quantization source | Fixed: register the input URI, retain variant settings, reject conflicting source variants, use unique targets, and record the config hash. |
| 02 | Requested quantization becomes a no-op | Fixed: reject unknown or incompatible methods; require a choice or inspected capabilities. |
| 03 | Missing quality evidence passes | Fixed: require valid baseline and candidate scores; reject empty or zero-signal results. |
| 04 | Old evaluation results pass a failed run | Fixed: unique adapter run paths; require successful exit and fresh, explicitly named model scores. |
| 05 | Selected endpoint is stopped | Fixed in mocked tests: restart the winner and evaluate that live endpoint. GPU proof remains open. |
| 06 | Dry run reported as published | Fixed: require a passing evaluation of the target and a matching completed upload receipt. No upload was performed. |
| 07 | Plans omit evaluation | Fixed: validate evaluation after every model transform; publication must follow evaluation. |
| 08 | Remote target only controls some stages | Blocked: reject remote recipes and mixed targets before launch. A shared host/filesystem abstraction remains open. |
| 09 | Resource limits and GPU selection | Partial: reject unsupported dollar/GPU-hour caps, honor step limits, check wall time between calls, and expose training container settings. Hard limits and device isolation remain open. |
| 10 | Cleanup deletes shared serving container | Partial: unique names, ownership labels, immutable Docker IDs, and occupied-port checks. Prove every supported launcher's cleanup and failure behavior. |
| 11 | Unsafe recipe replay after restart | Partial: durable stage receipts block replay of unfinished work; completed endpoint receipts require revalidation. Automatic job/output reconciliation remains open. |
| 12 | Custom holdout leaks into training | Fixed for normalized prompt overlap: filter during curation, verify before SFT/GRPO, and record file hashes. Semantic duplicates and final-test policy remain open. |
| 13 | Vulnerable dependencies | Fixed for the audited control-plane lock: refreshed Python dependencies; retained compatible, pinned terminal renderer versions; audits are clear. GPU images need separate scans. |
| 14 | Unrepeatable installs | Partial: hashed Python lock, npm lock, pinned model-forge revision and CI actions. GPU image digests and training dependency locks remain open. |
| 15 | False job states and unsafe cancellation | Partial: quote remote paths, record before launch, write atomically, check process identity before cancellation, and distinguish a stop request from confirmed termination. Detached container ownership and host-loss recovery remain open. |
| 16 | Approval relies on prompts | Partial: exact tool/argument scope, full arguments, correlated replies, and a publication check in the generic Forge tool. Shell access remains trusted execution, not a sandbox. |
| 17 | SSH and shell handling | Partial: known-host enforcement, concurrent output reads, quoted shared path helpers and remote working directory. Audit remaining system-tool command builders and test a real SSH host. |
| 18 | Codex authentication and streaming | Partial: honor CODEX_HOME, reject incomplete streams, lock refreshes and replace credentials atomically. The direct OAuth adapter remains experimental; supported integration and cross-client coordination remain open. |
| 19 | Secret storage | Partial: restrict credential files before writing and redact common secrets from durable events. Plaintext storage, child-process environment scope and broader secret detection remain open. |
| 20 | Artifact identity and scope | Partial: immutable records, locked writes, corruption errors, registered seeds and failed-artifact filtering. Weight manifests, scoped discovery and complete lineage validation remain open. |
| 21 | Repair recalls rejected work as success | Fixed: quality stays pending after a transform; evaluation updates that artifact's outcome; repair uses its latest result and keeps GRPO's method metadata. |
| 22 | Smoke defaults enter real training | Partial: SFT uses all curated rows unless bounded explicitly; distillation requires an explicit provider and rejects template simulation for training. Training schedules still need workload-specific validation. |
| 23 | Data contracts and provenance | Partial: strict JSONL reads, safe message validation, overlap checks, counts and hashes; unknown licenses stay unknown. Source-level evidence and streaming curation remain open. |
| 24 | Weak numeric scoring | Partial: fractions and scientific notation, minimum sample count, nonzero quality, strict think-block format, and one model with adapters disabled for baseline. Units, task-specific verifiers, paired uncertainty and GPU memory proof remain open. |
| 25 | Misleading status and hung approval | Fixed for reported cases: preserve loop outcome, correlate approvals, deny on EOF and reject non-object messages. Protocol versioning and queue bounds remain open. |
| 26 | Conflicting command names | Fixed: Python owns forgewright; Node owns forgewright-tui; unsupported UI arguments fail; named providers resolve through config. |
| 27 | Missing CI and handoff tests | Partial: Windows/Linux Python matrix, Node tests, builds, types, lint and audits. GPU, real SSH, PTY and upload-canary tests remain open; branch protection is unchanged. |
| 28 | Packaging and contributor upkeep | Partial: explicit tool versions/rules, clean production lint/types, contributor and security guides, changelog and lockfiles. Release/tag work awaits integration proof. |
| 29 | Terminal usability | Partial: full approval arguments, control-sequence stripping, cumulative tokens, retry state, NO_COLOR precedence and reduced motion. Keyboard-only, screen-reader and resize tests remain open. |
| 30 | Context and provider limits | Partial: bound oversized recent context and validate tool arguments. Exact token accounting, retry classification, cancellation during requests and cost accounting remain open. |
| 31 | Claims exceed proof | Partial: README now states supported paths and limits; abliteration is optional. A dated small-model walkthrough and evidence bundle remain open. |

## Next work, in order

1. Prove one local Linux GPU recipe. Choose the host and device first. Pin the model,
   data and image; use a small model and a short runtime. Check the source checkpoint,
   fresh output, paired evaluation and surviving endpoint. Keep the full receipt.
2. Tie each job and container to one run. Add device checks and scheduler-enforced
   runtime limits, then reconcile an interrupted run without duplicate training.
3. Add a shared remote execution and file-access layer. Test paths with spaces,
   disconnection, process exit and cleanup before restoring remote recipes.
4. Separate repair validation from final release testing. Save per-example results,
   score uncertainty, weight manifests and source-license evidence.
5. Test the terminal in a PTY, replace the experimental authentication path, and
   classify provider retries. Add a non-public upload canary only with a chosen target.

## Routine upkeep

- Review monthly dependency PRs. Keep both lockfiles, run audits, and inspect major
  upgrades before accepting them. A newer version can introduce more dependencies.
- Require passing CI before merging. Run a recorded GPU smoke after model-forge,
  GPU image, trainer, serving or evaluation changes.
- Before each release, check version parity, package contents, documentation commands,
  known limits and the evidence for each supported host. Follow CONTRIBUTING.md.
- Review retained logs, transcripts and model/data storage. Define retention before
  adding automatic cleanup; incomplete jobs and unique artifacts need protection.
