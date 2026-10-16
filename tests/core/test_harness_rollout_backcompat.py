# [MODULE DOC]
# Back-compatibility tests for the openenv.core.harness package split: the rollout submodule's
# public names, their re-export from the package root, a private helper re-export, importability
# of collect modules, and distinctness of the two adapter ABCs.

# SPDX-License-Identifier: BSD-3-Clause

"""Back-compat guarantees for the openenv.core.harness package split."""

from __future__ import annotations

import importlib

import openenv.core.harness as harness_pkg
from openenv.core.harness import rollout

ROLLOUT_PUBLIC_NAMES = [
    "CLIHarnessAdapter",
    "HarnessAdapter",
    "HarnessRolloutResult",
    "HarnessRunLimits",
    "LoopOwningSession",
    "MCPHarnessAdapter",
    "Message",
    "ModelStep",
    "ModelStepResult",
    "RESERVED_TOOL_NAMES",
    "ResourceSession",
    "ResourceSessionFactory",
    "RolloutEvent",
    "SessionMCPBridge",
    "StepEnvSessionAdapter",
    "ToolResult",
    "ToolTraceEntry",
    "TraceEntry",
    "TrainableSession",
    "VerifyResult",
    "build_harness_rollout_func",
]


def test_rollout_all_matches_expected_names():
    # [FUNCTION DOC]
    # rollout.__all__ equals the expected list of public names.
    assert sorted(rollout.__all__) == sorted(ROLLOUT_PUBLIC_NAMES)


def test_rollout_names_reexported_from_package_root():
    # [FUNCTION DOC]
    # Each rollout public name is in the package __all__ and is the identical object.
    for name in ROLLOUT_PUBLIC_NAMES:
        assert name in harness_pkg.__all__
        assert getattr(harness_pkg, name) is getattr(rollout, name)


def test_private_resolve_env_reward_reexported():
    # [FUNCTION DOC]
    # The private _resolve_env_reward helper is importable from the package root.
    # tests/scripts/test_browsergym_harness_eval_examples.py imports this
    # private helper from the package root; keep it importable.
    assert harness_pkg._resolve_env_reward is rollout._resolve_env_reward


def test_collect_modules_import():
    # [FUNCTION DOC]
    # The harness collect module and the CLI collect command import successfully.
    importlib.import_module("openenv.core.harness.collect")
    importlib.import_module("openenv.cli.commands.collect")


def test_adapter_abcs_are_distinct():
    # [FUNCTION DOC]
    # AgenticHarnessAdapter and HarnessAdapter are different classes and both exported.
    assert harness_pkg.AgenticHarnessAdapter is not harness_pkg.HarnessAdapter
    assert "AgenticHarnessAdapter" in harness_pkg.__all__
    assert "HarnessAdapter" in harness_pkg.__all__
