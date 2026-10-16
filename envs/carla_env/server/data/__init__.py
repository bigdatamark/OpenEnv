# [PACKAGE DOC]
# Packaged JSON data for CARLA scenarios and loaders for it.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data loading utilities for CARLA scenarios.

Adapted from SinatrasC/carla-env:
https://github.com/SinatrasC/carla-env
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

__all__ = ["load_json", "load_trolley_micro_benchmarks"]

_DATA_DIR = Path(__file__).parent


def load_json(name: str) -> Dict[str, Any]:
    # [FUNCTION DOC]
    # Reads and parses the JSON file with the given name from this package directory.
    # Raises: FileNotFoundError / json decode errors.
    path = _DATA_DIR / name
    with open(path, "r") as f:
        return json.load(f)


def load_trolley_micro_benchmarks() -> Dict[str, Any]:
    # [FUNCTION DOC]
    # Returns the parsed trolley_micro_benchmarks.json mapping (benchmark id -> definition).
    return load_json("trolley_micro_benchmarks.json")
