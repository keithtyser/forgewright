"""Pass explicit local paths and container settings to generated commands."""
from __future__ import annotations

import os
import shlex
from typing import TypedDict

from forgewright.tools.jobs import shell_path


class ContainerOptions(TypedDict):
    repo: str
    models_dir: str
    hf_home: str
    image: str
    gpus: str


def container_options(repo) -> ContainerOptions:
    return {
        "repo": shell_path(str(repo.resolve())),
        "models_dir": shell_path(os.environ.get("MODEL_FORGE_MODELS_DIR", "~/models")),
        "hf_home": shell_path(os.environ.get("MODEL_FORGE_HF_HOME", "~/.forgewright/hf_home")),
        "image": shlex.quote(os.environ.get("MODEL_FORGE_POSTTRAIN_IMAGE", "model-forge-posttrain-tf5:latest")),
        "gpus": shlex.quote(os.environ.get("MODEL_FORGE_GPUS", "all")),
    }
