# [MODULE DOC]
# Client-side result types shared by environment clients: the generic
# StepResult dataclass and the ObsT/StateT type variables.
# Type definitions for EnvTorch
from dataclasses import dataclass
from typing import Any, Dict, Generic, Optional, TypeVar

# Generic type for observations
ObsT = TypeVar("ObsT")
StateT = TypeVar("StateT")


@dataclass
class StepResult(Generic[ObsT]):
    # [CLASS DOC]
    # Dataclass holding one step's result: observation (generic type), optional
    # float reward, done flag (default False) and optional metadata dict.
    """
    Represents the result of one environment step.

    Attributes:
        observation:
            The environment's observation after the action.
        reward (`float`, *optional*):
            Scalar reward for this step.
        done (`bool`, *optional*, defaults to `False`):
            Whether the episode is finished.
        metadata (`dict`, *optional*):
            Additional metadata returned alongside the observation.
    """

    observation: ObsT
    reward: Optional[float] = None
    done: bool = False
    metadata: Optional[Dict[str, Any]] = None
