"""Abliterator specialist — remove refusals. ModelArtifact -> ModelArtifact(abliterated).

Wraps `skills/abliterate.py` + model-forge's pipeline run in the posttrain container via the
canonical `scripts/run_in_container.sh`. Scaffolds the config (capability-preserving scar
defaults), runs collect then export, and registers the abliterated standalone model. The
dual gate (refusal must DROP and capability HOLD) is enforced downstream by the Evaluator.
"""
from __future__ import annotations

import subprocess
import time
from typing import Optional, Sequence

from forgewright.agents.base import Specialist
from forgewright.contracts import Artifact, Gate, ModelArtifact
from forgewright.skills.abliterate import write_abliterate_config
from forgewright.tools.base import ToolRegistry
from forgewright.tools.forge import ForgeRunner, ForgeTool
from forgewright.tools.jobs import JobManager, LaunchJobTool, MonitorJobTool, TailLogsTool

_PROMPT = """\
You are the Abliterator specialist: remove refusal behavior via contrastive refusal-direction
projection while preserving capability. Project mid layers only; leave embeddings/lm_head/MoE
experts untouched; keep strength conservative (over-abliteration breaks benign answers). You
never publish; hand the abliterated model on for eval (refusal must drop AND capability hold).

Produce a GENUINELY FRESH result. Collect refusal directions from THIS run and write new
weights to a fresh output dir. NEVER reuse a prior run's collected directions, and NEVER pass
off a pre-existing abliterated model as your output. If activation collection cannot run here
(e.g. transformers cannot load the model architecture), STOP and report the blocker: a truthful
failure is correct; a fabricated success is a serious error.
"""

_IMG_RUN = "bash scripts/run_in_container.sh python3 -m model_forge.pipelines.abliterate"


class Abliterator(Specialist):
    role = "Abliterator"
    accepts = ("model",)
    produces = "model"
    description = "Refusal-direction abliteration of a ModelArtifact -> abliterated ModelArtifact."

    def __init__(self, *, runner: Optional[ForgeRunner] = None, jobs: Optional[JobManager] = None,
                 host: Optional[str] = None, **kw) -> None:
        super().__init__(**kw)
        self.forge = runner or ForgeRunner()
        self.jobs = jobs or JobManager()
        self.host = host

    def system_prompt(self) -> str:
        return _PROMPT

    def tools(self) -> ToolRegistry:
        return ToolRegistry([
            ForgeTool(self.forge), LaunchJobTool(self.jobs),
            MonitorJobTool(self.jobs), TailLogsTool(self.jobs),
        ])

    def run(self, inputs: Sequence[Artifact], goal: str = "", *, strength: float = 3.0,
            layer_skip_first: int = 4, layer_skip_last: int = 2, layer_start: int = 4,
            layer_end: int = 24) -> Artifact:
        self.validate_inputs(inputs)
        model = inputs[0]
        family = model.meta.get("family") or "model"
        # collect/export need LOCAL weights; if handed a bare HF id, prefer a local mirror so we
        # run in-container (where transformers knows the arch) instead of failing on a remote id.
        source = self._resolve_local(model.uri)
        # A UNIQUE, timestamped output name so we can never collide with (or silently inherit)
        # a pre-existing abliterated model directory. If this run does not write weights here,
        # the dir simply will not exist -> the gate fails honestly.
        name = f"{family}_abliterated_{time.strftime('%Y%m%d-%H%M%S')}"
        self._emit("assistant", content=f"abliterate {model.id} (family {family}, strength {strength})")
        write_abliterate_config(
            self.forge.repo, family, name=name, source=source, local_dir=source,
            strength=strength, layer_skip_first=layer_skip_first, layer_skip_last=layer_skip_last,
            layer_start=layer_start, layer_end=layer_end, overwrite=True,
        )
        rel = f"configs/abliteration/{name}.yaml"
        stem = source.rstrip("/").split("/")[-1]
        out = f"~/models/{stem}-{name}"
        # collect refusal directions, then project + export the standalone model
        cmd = f"{_IMG_RUN} --config {rel} collect --execute && {_IMG_RUN} --config {rel} export --execute"
        since = time.time()
        rec = self.jobs.launch(cmd, host=self.host, cwd=str(self.forge.repo), name=f"ablate-{family}")
        self._emit("tool", tool="launch_job", ok=True, output=f"job {rec['id']} abliterating {family}")
        final = self.jobs.wait(rec["id"])
        exit_ok = bool(final and final.get("exit_code") == 0)
        # Provenance honesty: only claim success if THIS run actually wrote fresh weights into
        # the (uniquely-named) output dir. Catches no-ops, partial failures, and any attempt to
        # reuse stale/pre-existing weights.
        fresh = self._wrote_fresh_weights(out, since) if exit_ok else False
        ok = exit_ok and fresh
        verdict = (
            "ABLITERATED" if ok
            else "FAIL: abliteration did not exit cleanly" if not exit_ok
            else "FAIL: no freshly-written weights in the output dir (refusing to claim a stale/pre-existing model)"
        )
        art = ModelArtifact(
            uri=out, produced_by=self.role, parents=[model.id],
            config_hash="", run_id=(self.ledger.run_id if self.ledger else ""),
            gate=Gate(passed=ok, metrics={"exit_code": (final or {}).get("exit_code"),
                                          "strength": strength, "fresh_weights": fresh},
                      verdict=verdict),
            meta={"role": "abliterated", "family": family, "base": source},
        )
        self.registry.register(art)
        self._emit("tool", tool="register_artifact", ok=ok, output=f"abliterated ModelArtifact {art.id}")
        return art

    def _resolve_local(self, uri: str) -> str:
        """Prefer a local checkpoint over a bare HF id. Paths (~, /, .) pass through; a bare id
        like 'Qwen/Qwen3.5-0.8B' resolves to '~/models/Qwen3.5-0.8B' when the host has it."""
        if uri.startswith(("~", "/", ".")):
            return uri
        stem = uri.rstrip("/").split("/")[-1]
        cand = f"~/models/{stem}"
        return cand if self._host_run(f"test -d {cand} && echo yes") else uri

    def _host_run(self, cmd: str, timeout: int = 60) -> str:
        """Run a quick shell command on this specialist's host (ssh) or locally; return stdout."""
        argv = ["ssh", self.host, cmd] if self.host else ["bash", "-lc", cmd]
        try:
            r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
            return (r.stdout or "").strip()
        except Exception:  # noqa: BLE001 - verification is best-effort; treat as "not verified"
            return ""

    def _wrote_fresh_weights(self, out_dir: str, since: float) -> bool:
        """True iff the output dir contains model weights modified at/after `since` (i.e. written
        by THIS run). `~` is expanded by the shell (local bash -lc or the remote ssh shell)."""
        cmd = (
            f'find {out_dir} -type f \\( -name "*.safetensors" -o -name "*.bin" \\) '
            f"-newermt @{int(since)} 2>/dev/null | head -1"
        )
        return bool(self._host_run(cmd))
