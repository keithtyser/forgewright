"""Forgewright CLI entrypoint.

    forgewright "<goal>" --hardware ssh://user@host,local --brain vllm:model@http://host:8000/v1
    forgewright run "<goal>" ...     # explicit form
    forgewright doctor
    forgewright version

A bare goal (anything that isn't a known subcommand) is routed to `run`, so the
Claude-Code-style `forgewright "<goal>"` works without shadowing the subcommands.
"""
from __future__ import annotations

import sys
import time
import uuid
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from forgewright.brain.provider import Brain
from forgewright.config import Settings, parse_brain_arg, parse_hardware_arg
from forgewright.context.manager import ContextManager
from forgewright.ledger.ledger import Ledger
from forgewright.loop import SYSTEM_PROMPT, Agent
from forgewright.permissions import PermissionPolicy
from forgewright.tools.base import ToolRegistry
from forgewright.tools.discover import DiscoverTool
from forgewright.tools.files import EditFileTool, ReadFileTool, WriteFileTool
from forgewright.agents.run_recipe import RunRecipeTool
from forgewright.registry import Registry
from forgewright.skills.abliterate import ScaffoldAbliterateConfigTool
from forgewright.skills.finetune import ScaffoldFinetuneConfigTool
from forgewright.skills.serving_opt import ServingOptTool
from forgewright.tools.forge import ForgePublishTool, ForgeRunner, ForgeTool, ScaffoldQuantConfigTool
from forgewright.tools.gpu import GPUInspectTool
from forgewright.tools.jobs import (
    JobManager,
    KillJobTool,
    LaunchJobTool,
    ListJobsTool,
    MonitorJobTool,
    TailLogsTool,
)
from forgewright.tools.derive import DerivePlanTool
from forgewright.tools.shell import ShellTool
from forgewright.tools.ssh import SSHTool
from forgewright.tools.sysconfig import EnvConfigTool

cli = typer.Typer(add_completion=False, help="Forgewright — autonomous post-training harness.")
console = Console()


def _resolve_provider(brain: Optional[str], settings: Settings):
    """Pick the brain. Order: explicit --brain > the saved setup-wizard credentials >
    OpenRouter when its key is in the env > the configured default provider. Saved API keys
    are loaded into the env first either way, so an explicit --brain still finds its key."""
    import os as _os

    from forgewright.credentials import apply_credentials

    saved_brain = apply_credentials()  # populate env from ~/.forgewright/credentials.json
    if brain:
        return parse_brain_arg(brain)
    if saved_brain:
        return parse_brain_arg(saved_brain)
    if _os.environ.get("OPENROUTER_API_KEY"):
        return parse_brain_arg("openrouter:deepseek/deepseek-v4-pro")
    return settings.provider()


def _bind_swarm(agent, reporter, permissions) -> None:
    """Give the run_recipe tool this turn's transcript reporter + approval policy, so the
    Director swarm it dispatches streams into the one chat and its approvals surface here."""
    tool = agent.tools.get("run_recipe")
    if tool is not None and hasattr(tool, "bind"):
        tool.bind(reporter=reporter, permissions=permissions)


def build_registry() -> ToolRegistry:
    jm = JobManager()  # one shared job manager across the job tools
    forge = ForgeRunner()  # model-forge ./forge CLI driver (shared by the forge tools)
    return ToolRegistry(
        [
            ShellTool(),
            SSHTool(),
            EnvConfigTool(forge),
            ReadFileTool(),
            WriteFileTool(),
            EditFileTool(),
            GPUInspectTool(),
            DerivePlanTool(forge),
            DiscoverTool(forge),
            LaunchJobTool(jm),
            MonitorJobTool(jm),
            TailLogsTool(jm),
            KillJobTool(jm),
            ListJobsTool(jm),
            ForgeTool(forge),
            ForgePublishTool(forge),
            ScaffoldQuantConfigTool(forge),
            ScaffoldFinetuneConfigTool(forge),
            ScaffoldAbliterateConfigTool(forge),
            ServingOptTool(forge, jm),
            RunRecipeTool(registry=Registry(), jobs=jm, forge=forge),
        ]
    )


@cli.command()
def run(
    goal: str = typer.Argument(..., help="What to do, in plain language."),
    brain: Optional[str] = typer.Option(
        None, "--brain", help="Brain shorthand 'kind:model[@api_base]' or a provider name from config."
    ),
    hardware: Optional[str] = typer.Option(
        None, "--hardware", help="Targets: 'local,ssh://user@host/workdir'."
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Auto-approve destructive actions (unattended)."),
    max_steps: int = typer.Option(0, "--max-steps", help="Max agent steps (0 = run until the goal is met)."),
    config: Optional[Path] = typer.Option(None, "--config", help="providers.yaml path."),
) -> None:
    """Run an autonomous post-training goal."""
    settings = Settings.load(config)
    provider = _resolve_provider(brain, settings)
    targets = parse_hardware_arg(hardware) if hardware else [settings.hardware["local"]]

    def ask(tool, args) -> bool:  # console approver for destructive ops
        return typer.confirm(f"Allow {tool.name} ({tool.risk})? args={args}")

    policy = PermissionPolicy(ask_fn=None if yes else ask, auto_approve=yes)
    run_id = time.strftime("run-%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
    ledger = Ledger(run_id, settings.ledger_dir)

    hw_desc = ", ".join(
        f"{t.name} ({'ssh ' + (t.host or '') if t.kind == 'ssh' else 'local'})" for t in targets if t
    )
    augmented = f"Available hardware targets: {hw_desc}.\n\nGoal: {goal}"

    agent = Agent(
        brain=Brain(provider),
        tools=build_registry(),
        permissions=policy,
        ledger=ledger,
        context=ContextManager(SYSTEM_PROMPT),
        max_steps=max_steps,
    )
    _bind_swarm(agent, None, policy)   # headless: swarm uses the run's approval policy
    console.print(f"[bold cyan]Forgewright[/] {run_id} · brain={provider.litellm_model()} · hw={hw_desc}")
    console.print(f"[dim]goal:[/] {goal}\n")
    result = agent.run(augmented)
    console.rule("result")
    console.print(result.final or "(no final message)")
    console.print(f"[dim]done={result.done} · steps={result.steps} · ledger={ledger.path}[/]")


def _format_args(args: dict) -> str:
    s = ", ".join(f"{k}={v!r}" for k, v in args.items())
    return s[:100] + ("…" if len(s) > 100 else "")


def _live_reporter(console: Console):
    """Print the agent's assistant text + tool calls/results live (for interactive mode)."""

    def report(kind: str, data: dict) -> None:
        if kind == "assistant":
            text = (data.get("content") or "").strip()
            if text:
                console.print(text)
            for name in data.get("tool_calls", []):
                console.print(f"[cyan]→ {name}[/]")
        elif kind == "tool":
            mark = "[green]✓[/]" if data.get("ok") else "[red]✗[/]"
            console.print(f"  {mark} [bold]{data.get('tool')}[/] [dim]{_format_args(data.get('args', {}))}[/]")
            out = (data.get("output") or "").strip()
            if out:
                console.print(f"  [dim]{out if len(out) <= 500 else out[:500] + ' …'}[/]")

    return report


@cli.command()
def interactive(
    brain: Optional[str] = typer.Option(None, "--brain", help="Brain shorthand or provider name."),
    hardware: Optional[str] = typer.Option(None, "--hardware", help="Targets: 'local,ssh://user@host/workdir'."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Auto-approve destructive actions."),
    max_steps: int = typer.Option(0, "--max-steps", help="Max agent steps per message (0 = until the goal is met)."),
    config: Optional[Path] = typer.Option(None, "--config", help="providers.yaml path."),
) -> None:
    """Interactive Forgewright session (the default). Chat with the agent across turns."""
    settings = Settings.load(config)
    provider = _resolve_provider(brain, settings)
    targets = parse_hardware_arg(hardware) if hardware else [settings.hardware["local"]]

    def ask(tool, args) -> str:
        choice = typer.prompt(
            f"Allow {tool.name} ({tool.risk})? args={_format_args(args)}  [y=once / a=all / yolo / n=no]",
            default="y",
        ).strip().lower()
        return {"y": "yes", "a": "all", "yolo": "yolo", "n": "no"}.get(choice, "no")

    policy = PermissionPolicy(ask_fn=None if yes else ask, auto_approve=yes)
    run_id = time.strftime("sess-%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
    ledger = Ledger(run_id, settings.ledger_dir)
    agent = Agent(
        brain=Brain(provider),
        tools=build_registry(),
        permissions=policy,
        ledger=ledger,
        context=ContextManager(SYSTEM_PROMPT),
        max_steps=max_steps,
        reporter=_live_reporter(console),
    )
    _bind_swarm(agent, agent.reporter, policy)   # let chat dispatch the swarm into this transcript

    hw_desc = ", ".join(t.name for t in targets if t) or "local"
    console.print(f"[bold cyan]Forgewright[/] interactive · brain={provider.litellm_model()} · hw={hw_desc}")
    console.print("[dim]Type a goal. /exit to quit · Ctrl-C aborts the current turn.[/]\n")

    seeded = False
    while True:
        try:
            msg = console.input("[bold green]you›[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]bye[/]")
            break
        if not msg:
            continue
        if msg in ("/exit", "/quit", "exit", "quit"):
            console.print("[dim]bye[/]")
            break
        if not seeded:
            msg = f"Available hardware targets: {hw_desc}.\n\n{msg}"
            seeded = True
        try:
            result = agent.run(msg)
        except KeyboardInterrupt:
            console.print("\n[yellow]· turn aborted ·[/]")
            continue
        except Exception as e:  # noqa: BLE001
            console.print(f"[red]error:[/] {e}")
            continue
        if not result.done:
            console.print(f"[yellow]· stopped: {result.final} (steps={result.steps}) ·[/]")
        console.print()
    console.print(f"[dim]ledger: {ledger.path}[/]")


@cli.command()
def serve(
    brain: Optional[str] = typer.Option(None, "--brain", help="Brain shorthand or provider name."),
    config: Optional[Path] = typer.Option(None, "--config", help="providers.yaml path."),
    max_steps: int = typer.Option(0, "--max-steps", help="Max agent steps per turn (0 = until the goal is met)."),
) -> None:
    """Backend serve loop over stdin/stdout newline-JSON (what the terminal-kit TUI spawns).

    Reads user_msg / approval_response lines; streams assistant / tool / approval_request /
    done events back. One persistent agent across turns; approvals surface to the frontend.
    """
    import threading

    from forgewright.frontend.server import serve_stdio

    settings = Settings.load(config)
    provider = _resolve_provider(brain, settings)
    run_id = time.strftime("serve-%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
    ledger = Ledger(run_id, settings.ledger_dir)
    interrupt_ev = threading.Event()   # set by an `interrupt` message; checked between agent steps
    agent = Agent(
        brain=Brain(provider),
        tools=build_registry(),
        permissions=PermissionPolicy(),
        ledger=ledger,
        context=ContextManager(SYSTEM_PROMPT),
        max_steps=max_steps,
        interrupt=interrupt_ev.is_set,
    )

    def handle_turn(text: str, reporter, permissions) -> None:
        # reuse the one agent (persistent context); bind this turn's reporter + approver
        interrupt_ev.clear()   # a fresh turn starts uninterrupted
        agent.reporter = reporter
        agent.permissions = permissions
        _bind_swarm(agent, reporter, permissions)   # swarm streams into this turn's transcript
        agent.run(text)

    def handle_command(name: str, args: dict, reporter, emit) -> None:
        if name == "graph":
            emit(_graph_event(Registry(), limit=int(args.get("limit", 40))))
        elif name == "models":
            emit(_models_event(provider))
        else:
            reporter("assistant", {"role": "agent", "content": f"unknown command: {name}"})

    # record the full bidirectional session transcript (review + future training data)
    transcript = settings.home / "transcripts" / f"{run_id}.jsonl"
    session_meta = {
        "run_id": run_id, "brain": provider.litellm_model(), "kind": provider.kind,
        "started_at": time.time(), "ledger": str(ledger.path),
    }
    serve_stdio(
        handle_turn, instream=sys.stdin, outstream=sys.stdout, handle_command=handle_command,
        record_path=transcript, session_meta=session_meta, interrupt_event=interrupt_ev,
    )


def _graph_event(registry, limit: int = 40) -> dict:
    """Build a `graph` event: the recent provenance DAG (nodes + edges) from the registry."""
    arts = registry.all()[-limit:]
    nodes = []
    for a in arts:
        score = None
        if a.gate is not None:
            for k in ("score", "accuracy", "pass_rate", "reward"):
                v = a.gate.metrics.get(k)
                if isinstance(v, (int, float)):
                    score = float(v)
                    break
        nodes.append({
            "id": a.id, "kind": a.kind, "produced_by": a.produced_by,
            "parents": list(a.parents),
            "passed": (a.gate.passed if a.gate is not None else None),
            "score": score,
        })
    return {"type": "graph", "nodes": nodes}


def _models_event(provider) -> dict:
    """Build a `models` event. For Codex, probe what the live token can reach; otherwise note
    the current brain (discovery is Codex-only)."""
    if provider.kind == "oauth-codex":
        from forgewright.brain.codex_oauth import CodexClient

        try:
            models = CodexClient(model=provider.model).list_models()
            return {"type": "models", "available": models, "current": provider.model, "source": "probe"}
        except Exception as e:  # noqa: BLE001 - fall back to the curated list on the frontend
            return {"type": "models", "available": [], "current": provider.model,
                    "source": "error", "note": str(e)}
    return {"type": "models", "available": [provider.model], "current": provider.model,
            "source": "brain", "note": f"model discovery is Codex-only; current brain is {provider.kind}"}


@cli.command()
def doctor() -> None:
    """Check environment, config, and GPUs."""
    settings = Settings.load()
    console.print("[bold]Forgewright doctor[/]")
    console.print(f"home: {settings.home}")
    console.print(f"providers: {list(settings.providers)} (default={settings.default_provider})")
    console.print(f"hardware: {list(settings.hardware)}")
    console.print("[bold]GPUs:[/]")
    console.print(GPUInspectTool().run().output)


@cli.command()
def version() -> None:
    """Print the Forgewright version."""
    from forgewright import __version__

    console.print(__version__)


_KNOWN = {"run", "doctor", "version", "interactive", "serve"}


def app() -> None:
    """Console-script entry: no args -> interactive; a bare goal -> headless `run`."""
    argv = sys.argv[1:]
    if not argv:
        sys.argv.insert(1, "interactive")
    elif argv[0] not in _KNOWN and not argv[0].startswith("-"):
        sys.argv.insert(1, "run")
    cli()


if __name__ == "__main__":
    app()
