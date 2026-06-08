"""Enable `python -m forgewright ...` (what the Node TUI spawns for the backend)."""
from forgewright.cli import app

if __name__ == "__main__":
    app()
