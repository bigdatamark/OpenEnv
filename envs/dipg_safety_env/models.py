# [MODULE DOC]
# Pydantic models for the DIPG safety environment.
# envs/dipg_safety_env/models.py

from typing import Any, Dict

from openenv.core.env_server import Action, Observation, State
from pydantic import Field


class DIPGAction(Action):
    # [CLASS DOC]
    # Action: the model's full text response (expected to contain analysis/proof/final channels).
    """The action taken by the agent, which is its generated response."""

    llm_response: str


class DIPGObservation(Observation):
    # [CLASS DOC]
    # Observation: context passage and question (both empty on the terminal step); inherits reward, done.
    """The observation given to the agent: a context and a question."""

    context: str = ""
    question: str = ""


class DIPGState(State):
    # [CLASS DOC]
    # State: current context, question, and expected answer dict (e.g. "final" and "proof").
    """The internal state of the environment for tracking the current challenge."""

    current_context: str = ""
    current_question: str = ""
    # This will hold the ground-truth 'analysis' and 'final' answer
    # for scoring purposes.
    expected_answer: Dict[str, Any] = Field(default_factory=dict)
