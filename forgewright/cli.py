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
from forgewright.tools.files import EditFileTool, ReadFileTool, WriteFileTool
from forgewright.tools.forge import ForgePublishTool, ForgeRunner, ForgeTool
from forgewright.tools.gpu import GPUInspectTool
from forgewright.tools.jobs import (
    JobManager,
    KillJobTool,
    LaunchJobTool,
    ListJobsTool,
    MonitorJobTool,
    TailLogsTool,
)
from forgewright.tools.shell import ShellTool
from forgewright.tools.ssh import SSHTool

cli = typer.Typer(add_completion=False, help="Forgewright — autonomous post-training harness.")
console = Console()


def build_registry() -> ToolRegistry:
    jm = JobManager()  # one shared job manager across the job tools
    forge = ForgeRunner()  # model-forge ./forge CLI driver (shared by the forge tools)
    return ToolRegistry(
        [
            ShellTool(),
            SSHTool(),
            ReadFileTool(),
            WriteFileTool(),
            EditFileTool(),
            GPUInspectTool(),
            LaunchJobTool(jm),
            MonitorJobTool(jm),
            TailLogsTool(jm),
            KillJobTool(jm),
            ListJobsTool(jm),
            ForgeTool(forge),
            ForgePublishTool(forge),
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
    max_steps: int = typer.Option(80, "--max-steps", help="Max agent steps."),
    config: Optional[Path] = typer.Option(None, "--config", help="providers.yaml path."),
) -> None:
    """Run an autonomous post-training goal."""
    settings = Settings.load(config)
    provider = parse_brain_arg(brain) if brain else settings.provider()
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
    console.print(f"[bold cyan]Forgewright[/] {run_id} · brain={provider.litellm_model()} · hw={hw_desc}")
    console.print(f"[dim]goal:[/] {goal}\n")
    result = agent.run(augmented)
    console.rule("result")
    console.print(result.final or "(no final message)")
    console.print(f"[dim]done={result.done} · steps={result.steps} · ledger={ledger.path}[/]")


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


_KNOWN = {"run", "doctor", "version"}


def app() -> None:
    """Console-script entry: route a bare goal to `run`, else dispatch normally."""
    argv = sys.argv[1:]
    if argv and argv[0] not in _KNOWN and not argv[0].startswith("-"):
        sys.argv.insert(1, "run")
    cli()


if __name__ == "__main__":
    app()
