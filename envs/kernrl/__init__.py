# [PACKAGE DOC]
# kernrl package: a GPU kernel optimization environment where agents submit
# CUDA/Triton kernel code for a reference PyTorch problem and get compile,
# correctness and speedup feedback. Re-exports the client (kernrl_env) and the
# action/observation/state models.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""kernrl - GPU kernel optimization environment for training LLMs to write fast CUDA/Triton kernels."""

from .client import kernrl_env
from .models import KernelAction, KernelObservation, KernelState

__all__ = ["KernelAction", "KernelObservation", "KernelState", "kernrl_env"]
