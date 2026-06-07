"""The specialist swarm: one focused harness per post-training stage, plus the Director.

Each specialist is the shared `Agent` runtime (loop/brain/tools/ledger) wearing a focused
system prompt + a small tool subset, declaring the artifact `kind` it accepts and the one
it produces. The Director orchestrates them; the user only ever sees one chat.
"""
from forgewright.agents.base import Specialist, label_reporter

__all__ = ["Specialist", "label_reporter"]
