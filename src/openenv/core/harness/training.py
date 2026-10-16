# [MODULE DOC]
# Strict pydantic models for validated token capture handed to trainers: TrainingTurn
# (one engine call with prompt/completion ids, logprobs and a loss mask over
# prompt+completion) and TrainingTrace (list of uniquely identified turns), with
# conversion from/to trace-entry dicts.
# SPDX-License-Identifier: BSD-3-Clause

"""Validated token capture shared by environment producers and trainers."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TrainingTurn(BaseModel):
    # [CLASS DOC]
    # One engine call; strict, extra fields forbidden. Fields: node_id (non-empty),
    # prompt_token_ids, completion_token_ids, per_token_logps, loss_mask, request, response,
    # metadata. Validated after construction with validate_training_turn.
    """One engine call. The required loss mask covers prompt plus completion."""

    model_config = ConfigDict(strict=True, extra="forbid")

    node_id: str = Field(min_length=1)
    prompt_token_ids: list[int]
    completion_token_ids: list[int]
    per_token_logps: list[float]
    loss_mask: list[int]
    request: dict[str, Any] = Field(default_factory=dict)
    response: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_capture(self) -> TrainingTurn:
        # [FUNCTION DOC]
        # Runs validate_training_turn on the turn's ids, logprobs and mask; returns self.
        # Raises: ValueError (surfaced as a pydantic ValidationError) for malformed turns.
        from .capture.validate import validate_training_turn

        validate_training_turn(
            self.prompt_token_ids,
            self.completion_token_ids,
            self.per_token_logps,
            self.loss_mask,
        )
        return self


class TrainingTrace(BaseModel):
    # [CLASS DOC]
    # Strict model with schema_version 1 and a list of TrainingTurn; node ids must be unique.
    """Selected agent calls, each with an authoritative mask and unique identity.

    Zero-masked turns remain available for context and usage accounting. Task rewards
    come from the session's `verify()` method, independently of these masks.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    schema_version: Literal[1] = 1
    turns: list[TrainingTurn]

    @model_validator(mode="after")
    def unique_calls(self) -> TrainingTrace:
        # [FUNCTION DOC]
        # Returns self; raises ValueError if any node_id repeats.
        ids = [turn.node_id for turn in self.turns]
        if len(set(ids)) != len(ids):
            raise ValueError("training trace contains duplicate node_ids")
        return self

    @classmethod
    def from_entries(cls, entries: list[dict[str, Any]]) -> TrainingTrace:
        # [FUNCTION DOC]
        # Builds a trace from trace-entry dicts: node_id from entry["metadata"]["node_id"], token
        # fields copied as-is, request/response/metadata defaulting to {}.
        # Raises: KeyError for missing required keys; validation errors for malformed turns.
        """Adapt producer-selected entries without inferring tokens or masks."""
        return cls(
            turns=[
                TrainingTurn(
                    node_id=entry["metadata"]["node_id"],
                    prompt_token_ids=entry["prompt_token_ids"],
                    completion_token_ids=entry["completion_token_ids"],
                    per_token_logps=entry["per_token_logps"],
                    loss_mask=entry["loss_mask"],
                    request=entry.get("request", {}),
                    response=entry.get("response", {}),
                    metadata=entry.get("metadata", {}),
                )
                for entry in entries
            ]
        )

    def to_trace_entries(self) -> list[dict[str, Any]]:
        # [FUNCTION DOC]
        # Returns trace-entry dicts: each turn dumped without node_id, with node_id merged into
        # metadata.
        """Return diagnostic records for reward functions and trace viewers."""
        return [
            {
                **turn.model_dump(exclude={"node_id"}),
                "metadata": {**turn.metadata, "node_id": turn.node_id},
            }
            for turn in self.turns
        ]
