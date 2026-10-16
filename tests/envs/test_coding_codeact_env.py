# [MODULE DOC]
# Tests for PythonCodeActEnv import allow-listing (DEFAULT_SAFE_IMPORTS plus
# additional_imports, preserved across reset) and that an explicit transform
# is kept. Module constants are code snippets importing json and base64.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from coding_env.models import CodeAction
from coding_env.server.python_codeact_env import PythonCodeActEnv
from coding_env.server.python_executor import DEFAULT_SAFE_IMPORTS
from openenv.core.env_server.interfaces import Observation, Transform

# `json` is authorized through DEFAULT_SAFE_IMPORTS, `base64` only when the
# caller asks for it; neither is part of the smolagents base allowlist.
IMPORT_JSON = "import json\nprint(json.dumps({'a': 1}))"
IMPORT_BASE64 = "import base64\nprint(base64.b64encode(b'a').decode())"


class RecordingTransform(Transform):
    # [CLASS DOC]
    # HELPER: transform counting calls and returning the observation unchanged.
    def __init__(self) -> None:
        # [FUNCTION DOC]
        # Sets call count 0.
        self.calls = 0

    def __call__(self, observation: Observation) -> Observation:
        # [FUNCTION DOC]
        # Increments the count and returns the observation.
        self.calls += 1
        return observation


def test_additional_imports_keep_the_default_safe_imports():
    # [FUNCTION DOC]
    # With base64 added, both json and base64 imports run with exit code 0.
    env = PythonCodeActEnv(additional_imports=["base64"])
    env.reset()

    for code in (IMPORT_JSON, IMPORT_BASE64):
        observation = env.step(CodeAction(code=code))
        assert observation.exit_code == 0, observation.stderr


def test_additional_imports_survive_reset():
    # [FUNCTION DOC]
    # The same holds after two resets.
    env = PythonCodeActEnv(additional_imports=["base64"])
    env.reset()
    env.reset()

    for code in (IMPORT_JSON, IMPORT_BASE64):
        observation = env.step(CodeAction(code=code))
        assert observation.exit_code == 0, observation.stderr


def test_default_constructor_authorizes_only_the_safe_imports():
    # [FUNCTION DOC]
    # Default allows only json; base64 import fails with exit code 1.
    env = PythonCodeActEnv()
    env.reset()

    assert DEFAULT_SAFE_IMPORTS == ["json"]
    assert env.step(CodeAction(code=IMPORT_JSON)).exit_code == 0
    assert env.step(CodeAction(code=IMPORT_BASE64)).exit_code == 1


def test_explicit_transform_survives_reset():
    # [FUNCTION DOC]
    # The supplied transform remains set after reset.
    transform = RecordingTransform()
    env = PythonCodeActEnv(transform=transform)
    env.reset()

    assert env.transform is transform
