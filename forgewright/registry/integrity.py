"""Stream file hashes for local artifacts; never load model weights into memory."""
from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path


class IntegrityError(ValueError):
    pass


def _signature(info: os.stat_result) -> tuple:
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _plain(path: Path) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return
    # lstat also detects junctions on Python 3.11, before Path.is_junction.
    if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & getattr(
        stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0
    ):
        raise IntegrityError(f"Linked artifact paths are unsupported: {path}")


def local_path(uri: str, repo: Path | None = None) -> Path:
    if not uri or "://" in uri:
        raise IntegrityError("A local artifact path is required")
    path = Path(uri).expanduser()
    if not path.is_absolute():
        path = (repo or Path.cwd()) / path
    # Reject links before resolving; resolving first would hide them.
    for component in (path, *path.parents):
        _plain(component)
    if not path.exists():
        raise IntegrityError(f"Local artifact does not exist: {path}")
    return path.resolve()


def _inventory(root: Path) -> list[Path]:
    _plain(root)
    if root.is_file():
        return [root]
    if not root.is_dir():
        raise IntegrityError(f"Artifact is not a regular file or directory: {root}")
    files = []
    def unreadable(error: OSError) -> None:
        raise error

    for folder, dirs, names in os.walk(root, followlinks=False, onerror=unreadable):
        for name in dirs:
            _plain(Path(folder) / name)
        for name in names:
            path = Path(folder) / name
            _plain(path)
            files.append(path)
            if len(files) > 10000:
                raise IntegrityError("Artifact contains more than 10000 files")
    return sorted(files)


def _hash_file(path: Path) -> dict:
    _plain(path)
    before = path.stat()
    if not stat.S_ISREG(before.st_mode):
        raise IntegrityError(f"Artifact contains a non-regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        # Windows stat and fstat can report different ctime values. Compare
        # ctime only within the same API; both still check identity and mtime.
        if _signature(before)[:4] != _signature(opened)[:4]:
            raise IntegrityError(f"Artifact changed while opening: {path}")
        while block := stream.read(1024 * 1024):
            digest.update(block)
        after = os.fstat(stream.fileno())
    _plain(path)
    final = path.stat()
    if (_signature(opened) != _signature(after) or _signature(before) != _signature(final)
            or _signature(after)[:4] != _signature(final)[:4]):
        raise IntegrityError(f"Artifact changed while hashing: {path}")
    return {"size": after.st_size, "sha256": digest.hexdigest()}


def capture_manifest(uri: str, *, repo: Path | None = None, checkpoint: bool = False) -> dict:
    """Hash every file, including configuration/tokenizer files, in a stable tree."""
    try:
        root = local_path(uri, repo)
        paths = _inventory(root)
        if not paths:
            raise IntegrityError("Artifact contains no files")
        if checkpoint and (not root.is_dir() or not any(
            p.suffix.lower() in {".safetensors", ".bin", ".gguf"} for p in paths
        )):
            raise IntegrityError("Checkpoint directory contains no recognized weight files")
        signatures = {p: _signature(p.stat()) for p in paths}
        records = [{"path": p.relative_to(root).as_posix() if root.is_dir() else p.name,
                    **_hash_file(p)} for p in paths]
        if paths != _inventory(root):
            raise IntegrityError("Artifact file list changed while hashing")
        if any(_signature(p.stat()) != signatures[p] for p in paths):
            raise IntegrityError("Artifact files changed while hashing")
        encoded = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
        return {"schema": "forgewright.files.v1", "root": str(root), "files": records,
                "sha256": hashlib.sha256(encoded).hexdigest(), "checkpoint": checkpoint}
    except OSError as error:
        raise IntegrityError(f"Cannot read artifact: {error}") from error


def verify_manifest(uri: str, expected: object, *, repo: Path | None = None,
                    checkpoint: bool = False) -> dict:
    if not isinstance(expected, dict) or expected.get("schema") != "forgewright.files.v1":
        raise IntegrityError("A current file manifest is required; evaluate the checkpoint again")
    current = capture_manifest(uri, repo=repo, checkpoint=checkpoint)
    if current != expected:
        raise IntegrityError("Artifact files or path changed since the recorded verification")
    return current
