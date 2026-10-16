# [MODULE DOC]
# Core server-side interfaces: the chat Message typed dict, the ModelTokenizer
# and TaskProvider structural protocols, the abstract observation Transform,
# and the abstract Environment base class (Gym-style reset/step/state with async
# variants, metadata, optional transform and rubric hooks, optional web UI
# hooks, and cleanup). Type variables bind actions, observations and state to
# the pydantic base types.
# SPDX-License-Identifier: BSD-3-Clause

import inspect
from abc import ABC, abstractmethod
from typing import (
    Any,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    Tuple,
    TYPE_CHECKING,
    TypeVar,
)

from typing_extensions import TypedDict

from .types import Action, EnvironmentMetadata, Observation, State

if TYPE_CHECKING:
    from openenv.core.rubrics import Rubric

ActT = TypeVar("ActT", bound=Action)
ObsT = TypeVar("ObsT", bound=Observation)
StateT = TypeVar("StateT", bound=State)


class Message(TypedDict):
    # [CLASS DOC]
    # Typed dict for a chat message with string role and content (HF chat template
    # format).
    """A message in a conversation.

    Compatible with Huggingface chat template format.
    """

    role: str
    content: str


class ModelTokenizer(Protocol):
    # [CLASS DOC]
    # Structural protocol for HF-style tokenizers used by chat environments.
    """Protocol for tokenizers that support chat templates.

    This protocol defines the interface that tokenizers must implement
    to work with chat-based environments. It's compatible with
    Huggingface transformers tokenizers.
    """

    def apply_chat_template(
        self,
        conversation: list[Message],
        tokenize: bool = True,
        return_tensors: str | None = None,
        **kwargs: Any,
    ) -> Any:
        # [FUNCTION DOC]
        # Protocol method: format (and optionally tokenize) a conversation.
        """Apply a chat template to format and optionally tokenize a conversation.

        Args:
            conversation (`list[Message]`):
                List of message dictionaries with 'role' and 'content'.
            tokenize (`bool`, *optional*, defaults to `True`):
                Whether to tokenize the output.
            return_tensors (`str`, *optional*):
                Format for returned tensors ('pt' for PyTorch).
            **kwargs:
                Additional arguments.

        Returns:
            Formatted and optionally tokenized conversation.
        """
        ...

    def decode(
        self, token_ids: Any, skip_special_tokens: bool = False, **kwargs: Any
    ) -> str:
        # [FUNCTION DOC]
        # Protocol method: decode token ids to text.
        """Decode token IDs back to text.

        Args:
            token_ids (`Any`):
                Token IDs to decode.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether to skip special tokens in output.
            **kwargs:
                Additional arguments.

        Returns:
            `str`: Decoded text string.
        """
        ...


class TaskProvider(Protocol):
    # [CLASS DOC]
    # Structural protocol for optional task discovery methods an Environment may
    # define (sync or async); exposed by the HTTP server under /{env_name}/ and
    # answered with 501 when absent. Must work on a freshly constructed instance and
    # be side-effect-free.
    """
    Optional task discovery API for dataset-backed environments.

    An environment implements this protocol structurally — declare the methods on
    an [`~openenv.core.env_server.interfaces.Environment`] subclass, without
    inheriting from `TaskProvider`. When the methods are present,
    [`~openenv.core.env_server.http_server.HTTPEnvServer`] exposes them as HTTP
    routes under `/{env_name}/…`; when they are absent, those routes return
    `501 Not Implemented`. Each method may be sync or async.

    Task provider methods are for metadata/discovery only and should be
    side-effect-free. They must be callable on a freshly constructed
    environment instance because HTTP compatibility routes may create a
    short-lived instance solely for task discovery.

    Selecting a task is not part of this protocol — pass the chosen split and
    index to `reset()` instead. See the
    [Task API guide](https://huggingface.co/docs/openenv/guides/task-api).

    Examples:

    ```python
    env.list_splits()          # ["train", "test"]
    env.num_tasks("test")      # 7595
    env.get_task("test", 12)   # {"id": "test-12", "index": 12}
    env.reset(split="test", index=12)
    ```
    """

    def list_splits(self) -> list[Any]:
        # [FUNCTION DOC]
        # Protocol method: return split descriptors (strings, dicts or models).
        """
        Return task split descriptors supported by this environment.

        Returns:
            `list[Any]`: Split descriptors. Plain strings, dicts, and Pydantic
                models are all accepted; the server normalizes each entry to
                `{"name": ..., "type": ...}`.
        """
        ...

    def list_tasks(self, split: str) -> list[Any]:
        # [FUNCTION DOC]
        # Protocol method: return task specs for a split (may be a bounded preview).
        """
        Return all task specs for a split.

        Args:
            split (`str`):
                Task split name.

        Returns:
            `list[Any]`: Task specs for the split. Environments backed by very
                large or streamed splits may return a bounded preview, but
                `num_tasks` should still report the true total.
        """
        ...

    def num_tasks(self, split: str) -> int:
        # [FUNCTION DOC]
        # Protocol method: return the number of tasks in a split.
        """
        Return the number of task specs in a split.

        Args:
            split (`str`):
                Task split name.

        Returns:
            `int`: Number of task specs available in the split.
        """
        ...

    def get_task(self, split: str, index: int) -> Any:
        # [FUNCTION DOC]
        # Protocol method: return one task spec; IndexError when out of range.
        """
        Return one task spec by split and index.

        Args:
            split (`str`):
                Task split name.
            index (`int`):
                Task index within the split.

        Returns:
            `Any`: The task spec at that position.

        Raises:
            `IndexError`: If `index` is out of range for the split. The HTTP
                route converts this to a `400 Bad Request`.
        """
        ...

    def get_task_range(
        self,
        split: str,
        start: Optional[int] = None,
        stop: Optional[int] = None,
    ) -> list[Any]:
        # [FUNCTION DOC]
        # Protocol method: return task specs in the slice [start, stop).
        """
        Return task specs for Python slice-style range bounds.

        Args:
            split (`str`):
                Task split name.
            start (`int`, *optional*):
                Inclusive start index. Defaults to the beginning of the split.
            stop (`int`, *optional*):
                Exclusive stop index. Defaults to the end of the split.

        Returns:
            `list[Any]`: Task specs in `[start, stop)`.
        """
        ...


class Transform(ABC, Generic[ObsT]):
    # [CLASS DOC]
    # Abstract callable that maps an observation to a (possibly modified)
    # observation, e.g. to add rewards.
    """Transform observations to add rewards, metrics, or other modifications.

    Transforms follow the TorchRL pattern where they take an observation
    and return a (potentially modified) observation. This allows for
    flexible reward computation and observation augmentation.
    """

    @abstractmethod
    def __call__(self, observation: ObsT) -> ObsT:
        # [FUNCTION DOC]
        # Abstract: return the transformed observation.
        """Transform an observation.

        Args:
            observation (`ObsT`):
                The input observation.

        Returns:
            `ObsT`: The transformed observation.
        """
        pass


class Environment(ABC, Generic[ActT, ObsT, StateT]):
    # [CLASS DOC]
    # Abstract base for environment servers. Class flags:
    # SUPPORTS_CONCURRENT_SESSIONS (default False; required for more than one
    # server session) and REQUIRES_SINGLE_THREAD_EXECUTOR (default False; the
    # server then runs all sessions on one shared thread). Instances hold an
    # optional transform and rubric. Subclasses implement reset, step and the
    # state property.
    """Base class for all environment servers following Gym/Gymnasium API.

    See [rfcs/004-rubrics.md](https://github.com/huggingface/OpenEnv/blob/main/rfcs/004-rubrics.md) for rubric design details.

    Args:
        transform (`Transform`, *optional*):
            Optional transform to apply to observations.
        rubric (`Rubric`, *optional*):
            Optional rubric for reward computation. When provided, the
            rubric's output can be used to set the observation's reward in step().

    Attributes:
        SUPPORTS_CONCURRENT_SESSIONS (`bool`):
            Whether this environment supports concurrent sessions. When ``True``,
            multiple WebSocket connections can each have their own environment
            instance (up to ``max_concurrent_envs``). When ``False`` (default),
            the environment should only be used with a single session at a time.

            Set this to ``True`` in your subclass if the environment uses proper
            session isolation (unique working dirs, no shared mutable state, and
            external resources that can handle concurrent access).
        rubric (`Rubric`, *optional*):
            Optional rubric for computing rewards. Set in ``__init__`` and use in
            ``step()`` to compute observation rewards. Training infrastructure can
            access it for introspection:

            ```python
            for name, r in env.rubric.named_rubrics():
                print(f"{name}: {r.last_score}")
            ```
    """

    # Class-level flag indicating whether this environment supports concurrent sessions
    SUPPORTS_CONCURRENT_SESSIONS: bool = False

    REQUIRES_SINGLE_THREAD_EXECUTOR: bool = False

    # Optional rubric for reward computation
    rubric: Optional["Rubric"]

    def __init__(
        self,
        transform: Optional[Transform[ObsT]] = None,
        rubric: Optional["Rubric"] = None,
    ):
        # [FUNCTION DOC]
        # Stores the optional transform and rubric.
        self.transform = transform
        self.rubric = rubric

    @abstractmethod
    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> ObsT:
        # [FUNCTION DOC]
        # Abstract: start a new episode (optional seed, episode id, extra kwargs) and
        # return the initial observation.
        """Reset the environment and return initial observation."""
        pass

    async def reset_async(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> ObsT:
        # [FUNCTION DOC]
        # Default async reset: calls the synchronous reset with the same arguments (on
        # the calling thread).
        """Async version of reset. Default implementation calls sync reset.

        Override to provide true async implementation.
        """
        return self.reset(seed=seed, episode_id=episode_id, **kwargs)

    @abstractmethod
    def step(
        self,
        action: ActT,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> ObsT:
        # [FUNCTION DOC]
        # Abstract: apply an action (optional timeout and kwargs) and return the
        # observation.
        """Take a step in the environment."""
        pass

    async def step_async(
        self,
        action: ActT,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> ObsT:
        # [FUNCTION DOC]
        # Default async step: calls the synchronous step with the same arguments.
        """Async version of step. Default implementation calls sync step.

        Override to provide true async implementation.
        """
        return self.step(action, timeout_s=timeout_s, **kwargs)

    @property
    @abstractmethod
    def state(self) -> StateT:
        # [FUNCTION DOC]
        # Abstract property returning the current State.
        """Get the current environment state."""
        pass

    def get_metadata(self) -> EnvironmentMetadata:
        # [FUNCTION DOC]
        # Returns default EnvironmentMetadata: name = class name, description
        # "<class name> environment", version "1.0.0".
        """
        Get metadata about this environment.

        Override this method to provide custom metadata for the environment.
        Default implementation returns basic metadata derived from class name.

        Returns:
            [`EnvironmentMetadata`] with environment information.
        """
        return EnvironmentMetadata(
            name=self.__class__.__name__,
            description=f"{self.__class__.__name__} environment",
            version="1.0.0",
        )

    def _apply_transform(self, observation: ObsT) -> ObsT:
        # [FUNCTION DOC]
        # HELPER: Returns the transform applied to observation, or observation
        # unchanged if no transform is set.
        """Apply transform if one is provided."""
        if self.transform is not None:
            return self.transform(observation)
        return observation

    def _apply_rubric(self, action: ActT, observation: ObsT) -> float:
        # [FUNCTION DOC]
        # HELPER: Returns rubric(action, observation) if a rubric is set, else 0.0.
        # KNOWN ISSUE: if the rubric is async, the coroutine is returned instead of a
        # float (use _apply_rubric_async).
        """Apply rubric if one is provided.

        Args:
            action (`ActT`):
                The action taken by the agent.
            observation (`ObsT`):
                The resulting observation.

        Returns:
            `float`: Reward value from the rubric, or 0.0 if no rubric is set.

        Call this in `step()` to compute and assign the reward:

        ```python
        def step(self, action: MyAction, ...) -> MyObservation:
            # ... execute action and create observation ...
            observation.reward = self._apply_rubric(action, observation)
            return observation
        ```
        """
        if self.rubric is not None:
            return self.rubric(action, observation)
        return 0.0

    async def _apply_rubric_async(self, action: ActT, observation: ObsT) -> float:
        # [FUNCTION DOC]
        # HELPER: Calls the rubric if set, awaiting the result when it is a coroutine;
        # returns 0.0 without a rubric.
        """Apply rubric asynchronously if one is provided.

        Args:
            action (`ActT`):
                The action taken by the agent.
            observation (`ObsT`):
                The resulting observation.

        Returns:
            `float`: Reward value from the rubric, or 0.0 if no rubric is set.

        Call this in `step_async()` to compute and assign the reward:

        ```python
        async def step_async(self, action: MyAction, ...) -> MyObservation:
            # ... execute action and create observation ...
            observation.reward = await self._apply_rubric_async(action, observation)
            return observation
        ```
        """
        if self.rubric is not None:
            result = self.rubric(action, observation)
            # If rubric returns a coroutine, await it
            if inspect.iscoroutine(result):
                return await result
            return result
        return 0.0

    def _reset_rubric(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Calls rubric.reset() if a rubric is set.
        """Reset the rubric state if one is provided.

        Call this in `reset()` to clear any trajectory state in the rubric:

        ```python
        def reset(self, ...) -> MyObservation:
            self._reset_rubric()
            # ... create initial observation ...
            return observation
        ```
        """
        if self.rubric is not None:
            self.rubric.reset()

    async def _reset_rubric_async(self) -> None:
        # [FUNCTION DOC]
        # HELPER: If a rubric is set, calls its reset_async (awaiting a coroutine
        # result) when available, else its reset.
        """Reset the rubric state asynchronously if one is provided.

        Call this in `reset_async()` to clear any trajectory state in the rubric:

        ```python
        async def reset_async(self, ...) -> MyObservation:
            await self._reset_rubric_async()
            # ... create initial observation ...
            return observation
        ```
        """
        if self.rubric is not None:
            # Check if rubric has async reset method
            if hasattr(self.rubric, "reset_async"):
                result = self.rubric.reset_async()
                if inspect.iscoroutine(result):
                    await result
            else:
                self.rubric.reset()

    def render_web(self, observation: Dict[str, Any]) -> Optional[str]:
        # [FUNCTION DOC]
        # Web UI hook: returns HTML drawing for a serialized observation (plus reward
        # and done). Default returns None (no drawing).
        """
        HTML that draws an observation in the web interface at `/web`.

        Override it to show the environment's state (a board, a page, a plot).
        The default draws nothing and the interface lists the observation's fields.

        Args:
            observation (`dict`):
                The serialized observation, as returned by `reset()` or `step()`,
                plus top-level `reward` and `done`.
        """
        return None

    def web_actions(
        self, observation: Dict[str, Any]
    ) -> List[Tuple[str, Dict[str, Any]]]:
        # [FUNCTION DOC]
        # Web UI hook: returns a list of (label, action dict) one-click actions for the
        # observation. Default returns an empty list.
        """
        One-click actions for the web interface at `/web`, as `(label, action)` pairs.

        Override it when the valid actions are known from the observation (the legal
        moves of a game). Each action is the dict `step()` receives. The default
        offers none and the interface shows the action's form only.

        Args:
            observation (`dict`):
                The serialized observation, as returned by `reset()` or `step()`,
                plus top-level `reward` and `done`.
        """
        return []

    def close(self) -> None:
        # [FUNCTION DOC]
        # Cleanup hook; default does nothing.
        """Clean up resources used by the environment.

        Override this method to implement custom cleanup logic.
        Called when the environment is being destroyed or reset.
        """
        pass
