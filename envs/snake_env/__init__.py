# [PACKAGE DOC]
# Snake environment package: re-exports the client (SnakeEnv) and the action/observation
# models (SnakeAction, SnakeObservation) as the public API. No side effects beyond importing.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Snake Environment - A multi-agent snake game environment based on marlenv."""

from .client import SnakeEnv
from .models import SnakeAction, SnakeObservation

__all__ = ["SnakeAction", "SnakeObservation", "SnakeEnv"]
