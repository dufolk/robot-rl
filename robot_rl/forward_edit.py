"""Helpers for hand-patched standing-gate checkpoints.

Training keeps the original gate weights. Do not soften or freeze them here.
"""
from __future__ import annotations

import json
import zipfile

from robot_rl.checkpoint import checkpoint_path


def read_forward_edit(load_path) -> dict | None:
    path = checkpoint_path(load_path)
    with zipfile.ZipFile(path) as archive:
        if "forward_edit.json" not in archive.namelist():
            return None
        return json.loads(archive.read("forward_edit.json").decode("utf-8"))


def has_forward_edit(load_path) -> bool:
    return read_forward_edit(load_path) is not None
