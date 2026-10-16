# [MODULE DOC]
# Defines Rubric, the abstract reward-computation base class modelled on PyTorch's
# nn.Module: subclasses implement forward(action, observation); calling the rubric runs
# pre-hooks, forward (sync or async), stores last_score and runs post-hooks. Child rubrics
# assigned as attributes are auto-registered for traversal and path lookup.
# SPDX-License-Identifier: BSD-3-Clause

"""Base Rubric class for reward computation.

Rubrics compute rewards from actions and observations. The API is modeled
after PyTorch's nn.Module: users implement forward(), and the framework
handles child registration and hooks.

See RFC 004 for full design: rfcs/004-rubrics.md
"""

import inspect
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple


class Rubric(ABC):
    # [CLASS DOC]
    # Abstract rubric. Instance state: registered child rubrics (name -> Rubric), forward
    # hooks, forward pre-hooks, and last_score (None until first evaluation).
    """Abstract base class for reward computation.

    A Rubric computes a reward signal from an action and observation.
    Subclasses implement forward() to define the reward logic.

    Examples:

        ```python
        class MyRubric(Rubric):
            def forward(self, action, observation) -> float:
                return 1.0 if action.valid else 0.0

        rubric = MyRubric()
        reward = rubric(action, observation)
        ```

    Child rubrics are auto-registered when assigned as attributes,
    enabling hierarchical composition and introspection.
    """

    _rubric_children: Dict[str, "Rubric"]
    _forward_hooks: List[Callable]
    _forward_pre_hooks: List[Callable]
    last_score: Optional[float]

    def __init__(self):
        # [FUNCTION DOC]
        # Initializes empty children and hook lists and last_score None (bypassing __setattr__).
        # Use object.__setattr__ to avoid triggering __setattr__ during init
        object.__setattr__(self, "_rubric_children", {})
        object.__setattr__(self, "_forward_hooks", [])
        object.__setattr__(self, "_forward_pre_hooks", [])
        object.__setattr__(self, "last_score", None)

    def __setattr__(self, name: str, value: Any) -> None:
        # [FUNCTION DOC]
        # Sets an attribute; if the value is a Rubric it is also registered as a child under that
        # name.
        # KNOWN ISSUE: reassigning a child attribute to a non-Rubric value does not unregister it.
        # Auto-register child rubrics when assigned as attributes
        if isinstance(value, Rubric):
            self._rubric_children[name] = value
        object.__setattr__(self, name, value)

    def __call__(self, action: Any, observation: Any):
        # [FUNCTION DOC]
        # Evaluates the rubric. If forward is a coroutine function, returns a coroutine that runs
        # pre-hooks (awaiting async ones), awaits forward, stores last_score and runs post-hooks.
        # Otherwise runs pre-hooks, forward, stores last_score, runs post-hooks and returns the
        # result.
        # Returns: the reward (float) or an awaitable of it.
        """Evaluate the rubric with hooks.

        Args:
            action: The action taken by the agent.
            observation: The resulting observation.

        Returns:
            `float`: Reward value (typically 0.0 to 1.0).
        """
        # Check if forward method is async BEFORE calling it
        if inspect.iscoroutinefunction(self.forward):
            # Async path - pre-hooks will be called in _call_async
            result = self.forward(action, observation)
            return self._call_async(action, observation, result)
        else:
            # Sync path - call pre-hooks BEFORE forward()
            for hook in self._forward_pre_hooks:
                hook(self, action, observation)
            result = self.forward(action, observation)
            return self._call_sync(action, observation, result)

    def _call_sync(self, action: Any, observation: Any, result: float) -> float:
        # [FUNCTION DOC]
        # HELPER: Sync completion path; delegates to _finish_forward.
        """Synchronous call path."""
        return self._finish_forward(action, observation, result)

    def _run_forward_pre_hooks(self, action: Any, observation: Any) -> None:
        # [FUNCTION DOC]
        # HELPER: Runs pre-forward hooks synchronously with (rubric, action, observation).
        """Run pre-forward hooks synchronously."""
        for hook in self._forward_pre_hooks:
            hook(self, action, observation)

    async def _run_forward_pre_hooks_async(self, action: Any, observation: Any) -> None:
        # [FUNCTION DOC]
        # HELPER: Runs pre-forward hooks, awaiting coroutine-function hooks.
        """Run pre-forward hooks from an async call path."""
        for hook in self._forward_pre_hooks:
            if inspect.iscoroutinefunction(hook):
                await hook(self, action, observation)
            else:
                hook(self, action, observation)

    def _finish_forward(self, action: Any, observation: Any, result: float) -> float:
        # [FUNCTION DOC]
        # HELPER: Stores result as last_score, runs post-forward hooks with (rubric, action,
        # observation, result), returns result.
        """Store the result and run post-forward hooks synchronously."""
        self.last_score = result

        # Post-forward hooks
        for hook in self._forward_hooks:
            hook(self, action, observation, result)

        return result

    async def _finish_forward_async(
        self, action: Any, observation: Any, result: float
    ) -> float:
        # [FUNCTION DOC]
        # HELPER: Async variant of _finish_forward that awaits coroutine-function hooks.
        """Store the result and run post-forward hooks from an async call path."""
        self.last_score = result

        # Post-forward hooks
        for hook in self._forward_hooks:
            if inspect.iscoroutinefunction(hook):
                await hook(self, action, observation, result)
            else:
                hook(self, action, observation, result)

        return result

    async def _call_async(self, action: Any, observation: Any, result_coro) -> float:
        # [FUNCTION DOC]
        # HELPER: Async call path: runs pre-hooks, awaits the forward coroutine, then finishes.
        """Asynchronous call path."""
        # Pre-forward hooks
        await self._run_forward_pre_hooks_async(action, observation)

        # Await the forward result
        result = await result_coro
        return await self._finish_forward_async(action, observation, result)

    @abstractmethod
    def forward(self, action: Any, observation: Any) -> float:
        # [FUNCTION DOC]
        # Abstract. Must compute and return the reward for (action, observation); may be async.
        """Compute the reward. Implement this in subclasses.

        Args:
            action: The action taken by the agent.
            observation: The resulting observation.

        Returns:
            `float`: Reward value (typically 0.0 to 1.0).
        """
        raise NotImplementedError

    def register_forward_hook(
        self, hook: Callable[["Rubric", Any, Any, float], None]
    ) -> None:
        # [FUNCTION DOC]
        # Appends a post-forward hook.
        """Register a hook called after forward().

        Args:
            hook (`Callable`):
                Callable with signature (rubric, action, observation, result).
        """
        self._forward_hooks.append(hook)

    def register_forward_pre_hook(
        self, hook: Callable[["Rubric", Any, Any], None]
    ) -> None:
        # [FUNCTION DOC]
        # Appends a pre-forward hook.
        """Register a hook called before forward().

        Args:
            hook (`Callable`):
                Callable with signature (rubric, action, observation).
        """
        self._forward_pre_hooks.append(hook)

    def children(self) -> Iterator["Rubric"]:
        # [FUNCTION DOC]
        # Yields immediate child rubrics.
        """Iterate over immediate child rubrics."""
        yield from self._rubric_children.values()

    def named_children(self) -> Iterator[Tuple[str, "Rubric"]]:
        # [FUNCTION DOC]
        # Yields (name, rubric) for immediate children.
        """Iterate over immediate child rubrics with names."""
        yield from self._rubric_children.items()

    def rubrics(self) -> Iterator["Rubric"]:
        # [FUNCTION DOC]
        # Yields all descendant rubrics depth-first (pre-order).
        """Iterate over all descendant rubrics (depth-first)."""
        for child in self._rubric_children.values():
            yield child
            yield from child.rubrics()

    def named_rubrics(self, prefix: str = "") -> Iterator[Tuple[str, "Rubric"]]:
        # [FUNCTION DOC]
        # Yields (dot-separated name, rubric) for all descendants depth-first, prefixed by prefix.
        """Iterate over all descendant rubrics with dot-separated names."""
        for name, child in self._rubric_children.items():
            full_name = f"{prefix}.{name}" if prefix else name
            yield full_name, child
            yield from child.named_rubrics(full_name)

    def get_rubric(self, path: str) -> "Rubric":
        # [FUNCTION DOC]
        # Returns the descendant rubric at a dot-separated path of child names.
        # Raises: KeyError if any segment is missing.
        """Access a nested rubric by dot-separated path.

        Args:
            path (`str`):
                Dot-separated path (e.g., "code.syntax").

        Returns:
            `Rubric`: The rubric at the specified path.

        Raises:
            KeyError: If the path does not exist.
        """
        parts = path.split(".")
        current = self
        for part in parts:
            if part not in current._rubric_children:
                raise KeyError(f"Rubric path not found: {path}")
            current = current._rubric_children[part]
        return current

    def reset(self) -> None:
        # [FUNCTION DOC]
        # No-op; subclasses may reset internal state.
        """Reset any internal state. Override in subclasses if needed."""
        pass

    def state_dict(self) -> Dict[str, Any]:
        # [FUNCTION DOC]
        # Returns {}; subclasses may serialize configuration.
        """Serialize rubric configuration for checkpointing."""
        return {}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        # [FUNCTION DOC]
        # No-op; subclasses may restore configuration.
        """Load rubric configuration from checkpoint."""
        pass
