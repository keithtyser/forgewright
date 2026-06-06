"""Configuration: brain providers, hardware targets, and runtime governor.

Forgewright's control plane is light and cross-platform. Heavy ML execution
happens on *hardware targets* (the local GPU or SSH boxes). *Brains* are
model-agnostic via LiteLLM, so any of a local vLLM server, a hosted API, or a
(quarantined) subscription OAuth tap can drive the agent.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, Field


def _home() -> Path:
    return Path(os.environ.get("FORGEWRIGHT_HOME", str(Path.home() / ".forgewright")))


# LiteLLM provider-prefix + default endpoint/key-env per brain kind.
_KIND_PREFIX = {
    "vllm": "hosted_vllm",
    "openai": "openai",
    "anthropic": "anthropic",
    "openrouter": "openrouter",
}
_KIND_DEFAULT_BASE = {"vllm": "http://localhost:8000/v1"}
_KIND_DEFAULT_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}

BrainKind = Literal["vllm", "openai", "anthropic", "openrouter", "oauth-claude", "oauth-codex"]


class ProviderConfig(BaseModel):
    """A single brain backend, normalized down to one LiteLLM call."""

    name: str
    kind: BrainKind = "openai"
    model: str
    api_base: Optional[str] = None
    api_key_env: Optional[str] = None
    extra: dict[str, Any] = Field(default_factory=dict)

    def litellm_model(self) -> str:
        prefix = _KIND_PREFIX.get(self.kind)
        # Pass through already-qualified model ids (those containing a "/").
        if prefix and "/" not in self.model:
            return f"{prefix}/{self.model}"
        return self.model

    def resolved_api_base(self) -> Optional[str]:
        return self.api_base or _KIND_DEFAULT_BASE.get(self.kind)

    def resolved_api_key(self) -> Optional[str]:
        env = self.api_key_env or _KIND_DEFAULT_KEY_ENV.get(self.kind)
        return os.environ.get(env) if env else None


class HardwareTarget(BaseModel):
    """Where jobs run: the local machine or a remote box over SSH."""

    name: str
    kind: Literal["local", "ssh"] = "local"
    host: Optional[str] = None  # "user@host" for ssh
    workdir: str = "~/forgewright-runs"
    gpus: Optional[str] = None  # optional manual override; else auto-detected


class Governor(BaseModel):
    """Hard caps enforced from outside the training process (blast-radius limits)."""

    max_steps: int = 80
    max_gpu_hours: float = 24.0
    max_cost_usd: float = 50.0
    max_wall_clock_hours: float = 48.0


class Settings(BaseModel):
    home: Path = Field(default_factory=_home)
    default_provider: Optional[str] = None
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)
    hardware: dict[str, HardwareTarget] = Field(default_factory=dict)
    governor: Governor = Field(default_factory=Governor)

    @property
    def runs_dir(self) -> Path:
        return self.home / "runs"

    @property
    def ledger_dir(self) -> Path:
        return self.home / "ledger"

    @property
    def memory_db(self) -> Path:
        return self.home / "memory.duckdb"

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "Settings":
        """Load from YAML (default ``$FORGEWRIGHT_HOME/providers.yaml``), tolerant of
        partial files; falls back to a local-vLLM default brain + a local target."""
        home = _home()
        cfg_path = path or (home / "providers.yaml")
        raw: dict[str, Any] = {}
        if cfg_path.exists():
            raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

        providers: dict[str, ProviderConfig] = {}
        for name, p in (raw.get("providers") or {}).items():
            providers[name] = ProviderConfig(name=name, **{k: v for k, v in p.items() if k != "name"})
        if not providers:
            providers = {p.name: p for p in _default_providers()}

        hardware: dict[str, HardwareTarget] = {}
        for name, h in (raw.get("hardware") or {}).items():
            hardware[name] = HardwareTarget(name=name, **{k: v for k, v in h.items() if k != "name"})
        hardware.setdefault("local", HardwareTarget(name="local", kind="local"))

        default = raw.get("default_provider") or (
            "local-vllm" if "local-vllm" in providers else next(iter(providers), None)
        )
        return cls(
            home=home,
            default_provider=default,
            providers=providers,
            hardware=hardware,
            governor=Governor(**(raw.get("governor") or {})),
        )

    def provider(self, name: Optional[str] = None) -> ProviderConfig:
        key = name or self.default_provider
        if not key or key not in self.providers:
            raise KeyError(f"Unknown brain provider {key!r}; known: {list(self.providers)}")
        return self.providers[key]


def _default_providers() -> list[ProviderConfig]:
    return [ProviderConfig(name="local-vllm", kind="vllm", model="default")]


def parse_brain_arg(arg: str) -> ProviderConfig:
    """Parse a ``--brain`` shorthand into a ProviderConfig.

    Forms::

        "openai:gpt-5.2"                              kind=openai
        "anthropic:claude-opus-4-8"                   kind=anthropic
        "vllm:qwen3.5-coder-30b@http://host:8000/v1"  kind=vllm with api_base
        "local:vllm/qwen3.5-coder-30b"               tolerated -> kind=vllm
        "bare-model-id"                              passthrough (kind=openai)
    """
    raw = arg.strip()
    if raw.startswith("local:"):
        raw = raw[len("local:") :]
        if raw.startswith("vllm/"):
            raw = "vllm:" + raw[len("vllm/") :]
    if ":" not in raw:
        return ProviderConfig(name=raw, kind="openai", model=raw)

    kind, rest = raw.split(":", 1)
    api_base: Optional[str] = None
    if "@" in rest:
        rest, api_base = rest.rsplit("@", 1)
    known = set(_KIND_PREFIX) | {"oauth-claude", "oauth-codex"}
    if kind not in known:
        # Not a recognized kind -> assume the whole thing was a passthrough model id.
        return ProviderConfig(name=arg, kind="openai", model=arg)
    return ProviderConfig(name=f"{kind}:{rest}", kind=kind, model=rest, api_base=api_base)  # type: ignore[arg-type]


def parse_hardware_arg(arg: str) -> list[HardwareTarget]:
    """Parse ``--hardware``: comma-separated ``local`` or ``ssh://user@host[/workdir]``."""
    targets: list[HardwareTarget] = []
    for tok in (t.strip() for t in arg.split(",") if t.strip()):
        if tok == "local":
            targets.append(HardwareTarget(name="local", kind="local"))
        elif tok.startswith("ssh://"):
            spec = tok[len("ssh://") :]
            workdir = "~/forgewright-runs"
            if "/" in spec:
                host, path = spec.split("/", 1)
                workdir = path if path.startswith("~") else "/" + path
            else:
                host = spec
            targets.append(HardwareTarget(name=host, kind="ssh", host=host, workdir=workdir))
        else:
            targets.append(HardwareTarget(name=tok, kind="local"))
    return targets
