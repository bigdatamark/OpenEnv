# [MODULE DOC]
# Tests for observation serialization: serialize_observation copies non-empty
# observation metadata to a top-level "metadata" key (while keeping it nested
# under "observation"), and Reset/Step responses plus GenericEnvClient and
# MCPToolClient result parsing consume that top-level metadata.
# SPDX-License-Identifier: BSD-3-Clause

from openenv.core.env_server.serialization import serialize_observation
from openenv.core.env_server.types import Observation, ResetResponse, StepResponse
from openenv.core.generic_client import GenericEnvClient
from openenv.core.mcp_client import MCPToolClient


class CustomObservation(Observation):
    # [CLASS DOC]
    # Observation subclass with two extra string fields (ally_tree,
    # task_instruction) used to check that subclass fields survive serialization.
    ally_tree: str = ""
    task_instruction: str = ""


def test_serialize_observation_promotes_metadata_to_top_level() -> None:
    # [FUNCTION DOC]
    # Asserts non-empty metadata appears at the top level of the serialized dict.
    obs = Observation(
        done=False,
        reward=0.5,
        metadata={"total_nodes": 5, "task_id": 1},
    )

    result = serialize_observation(obs)

    assert result["metadata"] == {"total_nodes": 5, "task_id": 1}


def test_serialize_observation_keeps_metadata_in_nested_payload() -> None:
    # [FUNCTION DOC]
    # Asserts metadata is also kept inside the nested "observation" payload.
    obs = Observation(metadata={"step": 3})

    result = serialize_observation(obs)

    assert result["observation"]["metadata"] == {"step": 3}


def test_serialize_observation_omits_empty_top_level_metadata() -> None:
    # [FUNCTION DOC]
    # Asserts empty metadata is not promoted to the top level but stays nested
    # as an empty mapping.
    obs = Observation(done=False, reward=0.0, metadata={})

    result = serialize_observation(obs)

    assert "metadata" not in result
    assert result["observation"]["metadata"] == {}


def test_serialize_observation_preserves_subclass_fields() -> None:
    # [FUNCTION DOC]
    # Asserts subclass-specific fields are serialized in the nested payload and
    # metadata is promoted.
    obs = CustomObservation(
        ally_tree="[ref=btn_1 role=button]",
        task_instruction="Book a ticket",
        done=False,
        reward=0.2,
        metadata={"variant": "label_drift"},
    )

    result = serialize_observation(obs)

    assert result["observation"]["ally_tree"] == "[ref=btn_1 role=button]"
    assert result["observation"]["task_instruction"] == "Book a ticket"
    assert result["metadata"] == {"variant": "label_drift"}


def test_reset_response_accepts_top_level_metadata() -> None:
    # [FUNCTION DOC]
    # Asserts ResetResponse built from serialized output exposes the metadata.
    serialized = serialize_observation(Observation(metadata={"reset_key": "val"}))

    reset_response = ResetResponse(**serialized)

    assert reset_response.metadata == {"reset_key": "val"}


def test_step_response_accepts_top_level_metadata() -> None:
    # [FUNCTION DOC]
    # Asserts StepResponse built from serialized output exposes the metadata.
    serialized = serialize_observation(Observation(metadata={"step_key": "val"}))

    step_response = StepResponse(**serialized)

    assert step_response.metadata == {"step_key": "val"}


def test_generic_client_receives_metadata() -> None:
    # [FUNCTION DOC]
    # Builds a GenericEnvClient without calling its constructor and asserts its
    # result parser maps reward, done and top-level metadata into the StepResult.
    payload = serialize_observation(
        Observation(
            done=False,
            reward=0.42,
            metadata={"total_nodes": 6, "completed": ["origin", "dest"]},
        )
    )

    client = GenericEnvClient.__new__(GenericEnvClient)
    step_result = client._parse_result(payload)

    assert step_result.reward == 0.42
    assert step_result.done is False
    assert step_result.metadata == {
        "total_nodes": 6,
        "completed": ["origin", "dest"],
    }


def test_generic_client_handles_missing_metadata() -> None:
    # [FUNCTION DOC]
    # Asserts the generic client sets StepResult.metadata to None when the
    # payload has no metadata.
    payload = {"observation": {"text": "hello"}, "reward": 0.0, "done": False}

    client = GenericEnvClient.__new__(GenericEnvClient)
    step_result = client._parse_result(payload)

    assert step_result.metadata is None


def test_mcp_client_prefers_top_level_metadata() -> None:
    # [FUNCTION DOC]
    # Asserts MCPToolClient result parsing uses top-level metadata over nested
    # observation metadata, for both the StepResult and the observation.
    payload = {
        "observation": {"tools": [], "metadata": {"nested": "old"}},
        "reward": 1.0,
        "done": False,
        "metadata": {"top_level": "new"},
    }

    client = MCPToolClient.__new__(MCPToolClient)
    step_result = client._parse_result(payload)

    assert step_result.metadata == {"top_level": "new"}
    assert step_result.observation.metadata == {"top_level": "new"}
