# [PACKAGE DOC]
# Package entry for the maze environment. Re-exports the client (MazeEnv) and the
# action/observation models so users can import them from the package root. No side effects
# beyond importing the client and models submodules.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Maze Env Environment."""

from .client import MazeEnv
from .models import MazeAction, MazeObservation

__all__ = [
    "MazeAction",
    "MazeObservation",
    "MazeEnv",
]
