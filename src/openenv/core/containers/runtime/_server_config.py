# [MODULE DOC]
# HELPER module: parses environment server configuration (the "app" field of
# openenv.yaml and the CMD of a Dockerfile) for runtime providers that need to
# know how to launch the server.
# SPDX-License-Identifier: BSD-3-Clause

"""Shared helpers for reading environment server configuration."""

from __future__ import annotations

import json
import re

import yaml


def parse_openenv_app_field(yaml_content: str) -> str | None:
    # [FUNCTION DOC]
    # Parses YAML text and returns the stripped string value of the top-level "app"
    # key, or None if parsing fails, the document is not a mapping, or the value is
    # missing, not a string, or empty.
    """Extract the top-level ``app`` value from raw ``openenv.yaml`` content."""
    try:
        data = yaml.safe_load(yaml_content) or {}
    except Exception:
        return None

    if not isinstance(data, dict):
        return None

    value = data.get("app")
    if isinstance(value, str):
        value = value.strip()
        return value if value else None
    return None


def parse_dockerfile_cmd(dockerfile_content: str) -> str | None:
    # [FUNCTION DOC]
    # Returns the command of the last line beginning with "CMD" (case-insensitive,
    # ignoring "#" comment lines). Exec-form JSON arrays of strings are joined with
    # spaces; invalid JSON falls back to the raw text. Returns None if no CMD.
    # KNOWN ISSUE: backslash line continuations are not joined, so a multi-line CMD
    # yields only its first line's text.
    """Extract the server command from the last Dockerfile ``CMD``."""
    last_cmd: str | None = None
    for line in dockerfile_content.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        match = re.match(r"CMD\s+(.+)", stripped, flags=re.IGNORECASE)
        if match:
            last_cmd = match.group(1).strip()

    if last_cmd is None:
        return None

    if last_cmd.startswith("["):
        try:
            parts = json.loads(last_cmd)
            if isinstance(parts, list) and all(isinstance(p, str) for p in parts):
                return " ".join(parts)
        except (json.JSONDecodeError, TypeError):
            # Invalid exec-form JSON falls back to the raw CMD below.
            pass

    return last_cmd if last_cmd else None
