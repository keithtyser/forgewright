"""Validate and fingerprint training and evaluation inputs before a job starts."""
from __future__ import annotations

import hashlib
import json
import unicodedata
from pathlib import Path


def read_rows(path: Path) -> list[dict]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as exc:
            raise ValueError(f"Invalid JSON at {path}:{number}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Expected an object at {path}:{number}")
        rows.append(row)
    if not rows:
        raise ValueError(f"Dataset is empty: {path}")
    return rows


def prompt_keys(row: dict) -> set[str]:
    texts = [row.get("prompt")]
    texts += [m.get("content") for m in row.get("messages", [])
              if isinstance(m, dict) and m.get("role") == "user"]
    return {" ".join(unicodedata.normalize("NFKC", t).casefold().split())
            for t in texts if isinstance(t, str) and t.strip()}


def data_path(repo: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else Path(repo) / path


def verify_split(repo: Path, training: str, holdout: str | None) -> dict:
    if not holdout:
        raise ValueError("A separate validation dataset is required before training")
    train_path, eval_path = data_path(repo, training), data_path(repo, holdout)
    train, evaluation = read_rows(train_path), read_rows(eval_path)
    held = set().union(*(prompt_keys(row) for row in evaluation))
    if not held:
        raise ValueError("Validation dataset has no usable prompts")
    if any(prompt_keys(row) & held for row in train):
        raise ValueError("Training data overlaps the validation dataset")
    return {"training_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
            "holdout_sha256": hashlib.sha256(eval_path.read_bytes()).hexdigest(),
            "rows": len(train), "holdout_rows": len(evaluation)}
