"""DataCurator specialist — the swarm's data origin. Produces a DatasetArtifact.

Two modes:
- ``curate_seed`` (no LLM): assemble seed/source JSONL(s), apply real curation hygiene
  (dedup by content hash, drop malformed / unclosed-<think> / empty rows), write a clean
  messages JSONL, and register a DatasetArtifact. Enough to seed the swarm today.
- ``distill`` (Jackrong teacher-distillation): drive model-forge's data factory
  (`forge data generate -> judge -> verify -> filter -> pack`) against a teacher endpoint
  (OpenRouter / local vLLM). Wired here; exercised once a teacher endpoint is configured.

The curation hygiene is pure + unit-tested; the heavy distill path shells out to `forge data`.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional, Sequence

from forgewright.agents.base import Specialist
from forgewright.contracts import Artifact, DatasetArtifact, Gate
from forgewright.tools.base import ToolRegistry
from forgewright.tools.forge import ForgeRunner, ForgeTool
from forgewright.tools.jobs import JobManager, LaunchJobTool, TailLogsTool

_PROMPT = """\
You are the DataCurator specialist in a post-training swarm. You build the training dataset
and hand it to the SFTTrainer. Curate for quality: dedup, drop malformed/unclosed-<think>
rows, keep assistant-only-loss-friendly messages, and never let held-out eval prompts leak
into training. For teacher distillation, generate <think>-formatted CoT from a strong teacher,
judge + verify + filter, then pack. You never train, abliterate, or publish.
"""


# --- pure curation hygiene (unit-tested) -------------------------------------------

def _closed_think_ok(text: str) -> bool:
    """A <think> block, if present, must be opened-then-closed exactly once, in order."""
    o, c = text.count("<think>"), text.count("</think>")
    if o == 0 and c == 0:
        return True
    if o != 1 or c != 1:
        return False
    return text.find("<think>") < text.find("</think>")


def _valid_messages(row: dict) -> bool:
    msgs = row.get("messages")
    if not isinstance(msgs, list) or not msgs:
        return False
    roles = {m.get("role") for m in msgs}
    if not ({"user", "assistant"} <= roles):
        return False
    for m in msgs:
        content = m.get("content")
        if not isinstance(content, str) or not content.strip():
            return False
        if m.get("role") == "assistant" and not _closed_think_ok(content):
            return False
    return True


def _row_hash(row: dict) -> str:
    return hashlib.sha256(json.dumps(row.get("messages"), sort_keys=True, default=str).encode()).hexdigest()


def curate_messages_rows(rows: Sequence[dict]) -> tuple[list[dict], dict[str, int]]:
    """Dedup + validate seed rows. Returns (kept, drop_counts{reason:n})."""
    kept: list[dict] = []
    seen: set[str] = set()
    drops = {"invalid": 0, "duplicate": 0}
    for row in rows:
        if not _valid_messages(row):
            drops["invalid"] += 1
            continue
        h = _row_hash(row)
        if h in seen:
            drops["duplicate"] += 1
            continue
        seen.add(h)
        kept.append(row)
    return kept, drops


def _count_jsonl(path: Path) -> int:
    try:
        return sum(1 for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip())
    except OSError:
        return 0


def _read_jsonl(path: Path) -> list[dict]:
    out = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if ln:
            try:
                out.append(json.loads(ln))
            except json.JSONDecodeError:
                continue
    return out


class DataCurator(Specialist):
    role = "DataCurator"
    accepts = ()              # starts from a fresh goal (+ seed paths)
    produces = "dataset"
    description = "Build/curate the training dataset -> DatasetArtifact (seed-curate or teacher-distill)."

    def __init__(self, *, runner: Optional[ForgeRunner] = None, jobs: Optional[JobManager] = None,
                 host: Optional[str] = None, **kw) -> None:
        super().__init__(**kw)
        self.forge = runner or ForgeRunner()
        self.jobs = jobs or JobManager()
        self.host = host

    def system_prompt(self) -> str:
        return _PROMPT

    def tools(self) -> ToolRegistry:
        return ToolRegistry([ForgeTool(self.forge), LaunchJobTool(self.jobs), TailLogsTool(self.jobs)])

    def run(self, inputs: Sequence[Artifact], goal: str = "", *, mode: str = "curate_seed",
            seed_paths: Optional[Sequence[str]] = None, family: str = "model",
            source: str = "Qwen/Qwen3.5-0.8B", run_name: Optional[str] = None,
            holdout: Optional[str] = None, out_path: Optional[str] = None,
            provider: str = "template") -> Artifact:
        run_name = run_name or f"{family}_curated_v0"
        if mode == "distill":
            seeds = list(seed_paths or [])
            if not seeds:
                raise ValueError("distill mode needs seed_paths (the seed JSONL to expand from)")
            return self._distill(goal, family, source, run_name, seed_path=seeds[0], provider=provider)
        return self._curate_seed(seed_paths or [], family, source, run_name, holdout, out_path)

    def _curate_seed(self, seed_paths, family, source, run_name, holdout, out_path) -> Artifact:
        repo = Path(self.forge.repo)
        rel_out = out_path or f"datasets/finetuning/{run_name}.jsonl"
        dst = repo / rel_out
        dst.parent.mkdir(parents=True, exist_ok=True)
        rows: list[dict] = []
        for sp in seed_paths:
            p = Path(sp) if Path(sp).is_absolute() else repo / sp
            if p.exists():
                rows.extend(_read_jsonl(p))
        kept, drops = curate_messages_rows(rows)
        dst.write_text("\n".join(json.dumps(r) for r in kept) + ("\n" if kept else ""), encoding="utf-8")
        self._emit("tool", tool="curate", ok=bool(kept),
                   output=f"curated {len(kept)} rows (dropped {drops}) -> {rel_out}")
        art = DatasetArtifact(
            uri=rel_out, produced_by=self.role,
            run_id=(self.ledger.run_id if self.ledger else ""),
            gate=Gate(passed=bool(kept), metrics={"rows": len(kept), **drops},
                      verdict="CURATED" if kept else "FAIL: no valid rows"),
            meta={"family": family, "source": source, "run_name": run_name,
                  "format": "messages", "rows": len(kept), **({"holdout": holdout} if holdout else {})},
        )
        self.registry.register(art)
        return art

    def _distill(self, goal, family, source, run_name, *, seed_path: str, provider: str = "template",
                 variant: str = "curated_v0") -> Artifact:
        """Teacher-distillation via the data factory. provider='template' validates the
        pipeline with no LLM; provider='openai_compatible' uses a teacher endpoint (set the
        provider's *_env vars, e.g. OpenRouter/local vLLM) for real <think> CoT."""
        from forgewright.skills.data_factory import write_dataset_config

        write_dataset_config(self.forge.repo, family, seed_path=seed_path, variant=variant, overwrite=True)
        self._emit("assistant", content=f"distill {family}/{variant} via factory (provider={provider})")
        for step in ["generate", "judge", "verify", "filter", "pack"]:
            res = self.forge.run(f"data {step} {family} {variant} --smoke --provider {provider}", timeout=3600)
            self._emit("tool", tool=f"data {step}", ok=res.ok, output=res.output[:160])
            if not res.ok:
                raise RuntimeError(f"data {step} failed: {res.output[:300]}")
        out = f"datasets/generated/{family}_{variant}/dataset.jsonl"
        rows = _count_jsonl(Path(self.forge.repo) / out)
        art = DatasetArtifact(
            uri=out, produced_by=self.role,
            gate=Gate(passed=rows > 0, metrics={"rows": rows},
                      verdict="DISTILLED" if rows > 0 else "FAIL: factory produced no rows"),
            meta={"family": family, "source": source, "run_name": run_name, "format": "messages",
                  "mode": "distill", "provider": provider, "rows": rows},
        )
        self.registry.register(art)
        self._emit("tool", tool="register_artifact", ok=rows > 0, output=f"DatasetArtifact {art.id} ({rows} rows)")
        return art
