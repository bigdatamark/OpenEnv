# [MODULE DOC]
# Regression test: each UnityMLAgentsEnvironment instance must pick a distinct Unity
# port (base_port + worker_id). Fakes mlagents_envs modules in sys.modules.
"""Each Unity env instance must talk to its Unity process on its own port."""

import sys
from types import SimpleNamespace

import pytest
from envs.unity_env.server.unity_environment import UnityMLAgentsEnvironment


def test_instances_use_different_ports(monkeypatch, tmp_path):
    # [FUNCTION DOC]
    # Installs fake mlagents_envs modules whose registry entry records the port passed to
    # make() and then raises; loads "PushBlock" in two env instances and asserts the
    # recorded ports differ. Side effects: monkeypatches sys.modules.
    ports = []

    class FakeRegistryEntry:
        # [CLASS DOC]
        # HELPER: Stand-in for RemoteRegistryEntry that records ports instead of launching Unity.
        def __init__(self, **kwargs):
            # [FUNCTION DOC]
            # Ignores all keyword arguments.
            pass

        def make(self, base_port=5005, worker_id=0, **kwargs):
            # [FUNCTION DOC]
            # Appends base_port + worker_id to the enclosing ports list, then raises RuntimeError.
            ports.append(base_port + worker_id)
            raise RuntimeError("stop before starting Unity")

    entry = SimpleNamespace(identifier="PushBlock", expected_reward=0, description="")
    modules = {
        "mlagents_envs": SimpleNamespace(),
        "mlagents_envs.base_env": SimpleNamespace(ActionTuple=None),
        "mlagents_envs.registry": SimpleNamespace(
            default_registry={"PushBlock": entry}
        ),
        "mlagents_envs.registry.remote_registry_entry": SimpleNamespace(
            RemoteRegistryEntry=FakeRegistryEntry
        ),
        "mlagents_envs.side_channel": SimpleNamespace(),
        "mlagents_envs.side_channel.engine_configuration_channel": SimpleNamespace(
            EngineConfigurationChannel=lambda: None
        ),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    for _ in range(2):
        env = UnityMLAgentsEnvironment(cache_dir=str(tmp_path))
        with pytest.raises(RuntimeError, match="stop before starting Unity"):
            env._load_environment("PushBlock")

    assert ports[0] != ports[1]
