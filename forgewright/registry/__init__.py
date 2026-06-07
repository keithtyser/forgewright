"""The artifact registry — the swarm's blackboard + provenance graph.

Producers `register()` artifacts; consumers `get()` / `latest()` them; `lineage()` walks
the parent edges. Append-only JSONL (one artifact per line) so it is durable, inspectable,
and crash-safe — the same pattern as the run ledger.
"""
from forgewright.registry.registry import Registry

__all__ = ["Registry"]
