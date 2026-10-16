# [MODULE DOC]
# Container rubrics for composing rewards (like nn.Sequential / ModuleList / ModuleDict):
# Sequential (fail-fast on zero), Gate (threshold), WeightedSum (weighted average with
# parallel async children), RubricList and RubricDict (structural containers without
# aggregation). The aggregating containers override __call__ to transparently support
# sync and async children, returning a coroutine when any child is async or when called
# from inside a running event loop (Sequential).
# SPDX-License-Identifier: BSD-3-Clause

"""Container rubrics for composing reward computations.

These containers provide common aggregation patterns for rubrics,
similar to how PyTorch provides nn.Sequential alongside nn.Module.

See RFC 004 for full design: rfcs/004-rubrics.md
"""

import asyncio
import inspect
from typing import Any, Dict, Iterator, List, Mapping, Tuple, Union

from openenv.core.rubrics.base import Rubric


def _in_async_context() -> bool:
    # [FUNCTION DOC]
    # HELPER: True if an asyncio event loop is running in the current thread.
    """Check if we're currently in an async context."""
    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False


class Sequential(Rubric):
    # [CLASS DOC]
    # Runs child rubrics in order; returns 0.0 as soon as any child scores 0, otherwise the
    # last child's score (1.0 when empty). Children are registered as rubric_<i>.
    """Run rubrics in order, fail-fast on zero.

    Runs child rubrics in order. If any returns 0, stops immediately
    and returns 0. This implements hierarchical gating patterns where
    syntax checks run before execution checks.

    Examples:

        ```python
        rubric = Sequential(
            Gate(Compiles()),
            Gate(PassesTests(), threshold=0.5),
            WeightedSum([PassesTests(), StyleRubric()], weights=[0.7, 0.3])
        )
        ```
    """

    def __init__(self, *rubrics: Rubric):
        # [FUNCTION DOC]
        # Registers each rubric as child "rubric_<i>" and keeps them in order.
        """Initialize with rubrics to run in sequence.

        Args:
            *rubrics (`Rubric`):
                Rubrics to run in order. Stops and returns 0 if any child returns 0.
        """
        super().__init__()
        for i, rubric in enumerate(rubrics):
            setattr(self, f"rubric_{i}", rubric)
        self._rubric_list = list(rubrics)

    def forward(self, action: Any, observation: Any) -> float:
        # [FUNCTION DOC]
        # Sync-only evaluation: calls children in order, returns 0.0 on the first zero, else the
        # last score (1.0 if empty). Not used by __call__.
        """Run rubrics in order, return 0 if any returns 0. Sync version."""
        result = 1.0
        for rubric in self._rubric_list:
            score = rubric(action, observation)
            if score == 0.0:
                return 0.0
            result = score
        return result

    def __call__(self, action: Any, observation: Any):
        # [FUNCTION DOC]
        # Evaluates children, choosing sync or async. Empty: 1.0 (as a coroutine when inside a
        # running loop). Calls the first child; if it returns a coroutine, continues on the async
        # path. Otherwise iterates synchronously, short-circuiting on 0.0 and switching to the
        # async path if a later child returns a coroutine. If all children were sync and a loop is
        # running, returns a coroutine wrapping the result; otherwise runs pre-hooks, stores
        # last_score, runs post-hooks and returns the score.
        # KNOWN ISSUE: pre-forward hooks run after the children have already been evaluated, not
        # before, unlike Rubric.__call__.
        """Override to choose sync or async path based on children."""
        # Empty case - check if in async context
        if not self._rubric_list:
            if _in_async_context():
                return self._empty_async(action, observation)
            else:
                self._run_forward_pre_hooks(action, observation)
                return self._finish_forward(action, observation, 1.0)

        # Call first rubric to see if it's async
        first_result = self._rubric_list[0](action, observation)
        if inspect.iscoroutine(first_result):
            # At least one child is async, use async path
            return self._call_async_detected(action, observation, first_result)
        else:
            # Continue with sync path
            if first_result == 0.0:
                self._run_forward_pre_hooks(action, observation)
                return self._finish_forward(action, observation, 0.0)

            final_result = first_result
            for i, rubric in enumerate(self._rubric_list[1:], start=1):
                score = rubric(action, observation)
                if inspect.iscoroutine(score):
                    # Found async mid-way, switch to async
                    # We already called rubric at index i, so pass the coroutine and remaining rubrics
                    return self._call_async_mid(
                        action,
                        observation,
                        score,
                        self._rubric_list[i + 1 :],
                    )
                if score == 0.0:
                    self._run_forward_pre_hooks(action, observation)
                    return self._finish_forward(action, observation, 0.0)
                final_result = score

            # All sync - check if in async context
            if _in_async_context():
                return self._wrap_sync_result(action, observation, final_result)
            else:
                self._run_forward_pre_hooks(action, observation)
                return self._finish_forward(action, observation, final_result)

    async def _empty_async(self, action, observation):
        # [FUNCTION DOC]
        # HELPER: Async result for an empty Sequential: runs hooks and returns 1.0.
        """Async path for empty sequential."""
        await self._run_forward_pre_hooks_async(action, observation)
        return await self._finish_forward_async(action, observation, 1.0)

    async def _wrap_sync_result(self, action, observation, result):
        # [FUNCTION DOC]
        # HELPER: Async wrapper for a fully synchronous result (runs hooks, returns result).
        """Wrap sync result for async context."""
        await self._run_forward_pre_hooks_async(action, observation)
        return await self._finish_forward_async(action, observation, result)

    async def _call_async_detected(self, action, observation, first_coro):
        # [FUNCTION DOC]
        # HELPER: Async path when the first child is async: awaits it, then evaluates the remaining
        # children (awaiting coroutines), short-circuiting on 0.0.
        """Async path when first child is async."""
        await self._run_forward_pre_hooks_async(action, observation)

        result = await first_coro
        if result == 0.0:
            return await self._finish_forward_async(action, observation, result)

        for rubric in self._rubric_list[1:]:
            score = rubric(action, observation)
            if inspect.iscoroutine(score):
                score = await score
            if score == 0.0:
                return await self._finish_forward_async(action, observation, 0.0)
            result = score

        return await self._finish_forward_async(action, observation, result)

    async def _call_async_mid(self, action, observation, first_async_coro, remaining):
        # [FUNCTION DOC]
        # HELPER: Async path when an async child is found mid-sequence: awaits that coroutine, then
        # evaluates the remaining children, short-circuiting on 0.0.
        """Async path when async detected mid-execution."""
        await self._run_forward_pre_hooks_async(action, observation)

        # Await the first async rubric (already called)
        result = await first_async_coro
        if result == 0.0:
            return await self._finish_forward_async(action, observation, 0.0)

        # Continue with remaining rubrics
        for rubric in remaining:
            score = rubric(action, observation)
            if inspect.iscoroutine(score):
                score = await score
            if score == 0.0:
                return await self._finish_forward_async(action, observation, 0.0)
            result = score

        return await self._finish_forward_async(action, observation, result)

    def __len__(self) -> int:
        # [FUNCTION DOC]
        # Returns the number of child rubrics.
        return len(self._rubric_list)

    def __getitem__(self, index: int) -> Rubric:
        # [FUNCTION DOC]
        # Returns the child rubric at index.
        return self._rubric_list[index]


class Gate(Rubric):
    # [CLASS DOC]
    # Returns the child's score if it is >= threshold (default 1.0), else 0.0.
    """Threshold wrapper - returns 0 if child score is below threshold.

    Useful for hard constraints like "must pass 50% of tests".

    Examples:

        ```python
        rubric = Gate(PassesTests(), threshold=0.5)
        # Returns PassesTests() score if >= 0.5, else 0.0
        ```
    """

    def __init__(self, rubric: Rubric, threshold: float = 1.0):
        # [FUNCTION DOC]
        # Registers the gated child as "rubric" and stores the threshold.
        """Initialize with a rubric and threshold.

        Args:
            rubric (`Rubric`):
                The rubric to gate.
            threshold (`float`, *optional*, defaults to `1.0`):
                Minimum score required. If child returns less than this, Gate returns 0.
        """
        super().__init__()
        self.rubric = rubric
        self.threshold = threshold

    def forward(self, action: Any, observation: Any) -> float:
        # [FUNCTION DOC]
        # Sync-only evaluation: child score if >= threshold else 0.0. Not used by __call__.
        """Return child score if >= threshold, else 0. Sync version."""
        score = self.rubric(action, observation)
        if score < self.threshold:
            return 0.0
        return score

    def __call__(self, action: Any, observation: Any):
        # [FUNCTION DOC]
        # Calls the child; if it returns a coroutine, returns a coroutine applying the threshold;
        # otherwise runs pre-hooks, applies the threshold, finishes and returns the result.
        # KNOWN ISSUE: pre-forward hooks run after the child has been evaluated.
        """Override to handle async child."""
        # Call child
        score = self.rubric(action, observation)

        if inspect.iscoroutine(score):
            # Child is async
            return self._call_async(action, observation, score)
        else:
            # Child is sync
            self._run_forward_pre_hooks(action, observation)
            result = 0.0 if score < self.threshold else score
            return self._finish_forward(action, observation, result)

    async def _call_async(self, action, observation, score_coro):
        # [FUNCTION DOC]
        # HELPER: Async path: runs pre-hooks, awaits the child score, applies the threshold and
        # finishes.
        """Async path."""
        await self._run_forward_pre_hooks_async(action, observation)

        score = await score_coro
        result = 0.0 if score < self.threshold else score
        return await self._finish_forward_async(action, observation, result)


class WeightedSum(Rubric):
    # [CLASS DOC]
    # Weighted sum of child scores with weights that must sum to 1.0.
    """Weighted combination of child rubrics.

    Standard aggregation pattern for multi-criteria evaluation.

    Examples:

        ```python
        rubric = WeightedSum(
            [PassesTests(), StyleRubric()],
            weights=[0.7, 0.3]
        )
        ```
    """

    def __init__(self, rubrics: List[Rubric], weights: List[float]):
        # [FUNCTION DOC]
        # Validates equal lengths and weights summing to 1.0 (tolerance 1e-6), registers children
        # as rubric_<i>, stores rubrics and weights.
        # Raises: ValueError on length mismatch or bad weight sum.
        """Initialize with rubrics and weights.

        Args:
            rubrics (`list[``Rubric``]`):
                List of rubrics to combine.
            weights (`list[float]`):
                Weight for each rubric. Must sum to 1.0.

        Raises:
            ValueError: If lengths don't match or weights don't sum to 1.0.
        """
        super().__init__()
        if len(rubrics) != len(weights):
            raise ValueError(
                f"Number of rubrics ({len(rubrics)}) must match "
                f"number of weights ({len(weights)})"
            )
        if abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {sum(weights)}")

        for i, rubric in enumerate(rubrics):
            setattr(self, f"rubric_{i}", rubric)
        self._rubric_list = list(rubrics)
        self._weights = list(weights)

    def forward(self, action: Any, observation: Any) -> float:
        # [FUNCTION DOC]
        # Sync-only evaluation: returns sum of score * weight. Not used by __call__.
        """Return weighted sum of child scores. Sync version."""
        total = 0.0
        for rubric, weight in zip(self._rubric_list, self._weights):
            score = rubric(action, observation)
            total += score * weight
        return total

    def __call__(self, action: Any, observation: Any):
        # [FUNCTION DOC]
        # Calls all children; if any returned a coroutine, returns a coroutine that gathers them;
        # otherwise runs pre-hooks, computes the weighted sum, finishes and returns it.
        # KNOWN ISSUE: pre-forward hooks run after the children have been evaluated.
        """Override to handle async children with parallel execution."""
        # Call all rubrics
        results = [rubric(action, observation) for rubric in self._rubric_list]

        # Check if any are async
        has_async = any(inspect.iscoroutine(r) for r in results)

        if has_async:
            # Use async path
            return self._call_async(action, observation, results)
        else:
            # Sync path
            self._run_forward_pre_hooks(action, observation)
            total = 0.0
            for score, weight in zip(results, self._weights):
                total += score * weight
            return self._finish_forward(action, observation, total)

    async def _call_async(self, action, observation, results):
        # [FUNCTION DOC]
        # HELPER: Async path: runs pre-hooks, awaits all coroutine results concurrently with
        # asyncio.gather, combines with sync results, computes the weighted sum and finishes.
        """Async path with parallel execution."""
        await self._run_forward_pre_hooks_async(action, observation)

        # Separate sync and async results
        async_tasks = []
        async_indices = []
        scores = [None] * len(results)

        for i, result in enumerate(results):
            if inspect.iscoroutine(result):
                async_tasks.append(result)
                async_indices.append(i)
            else:
                scores[i] = result

        # Await all async tasks in parallel
        if async_tasks:
            async_scores = await asyncio.gather(*async_tasks)
            for i, score in zip(async_indices, async_scores):
                scores[i] = score

        # Compute weighted sum
        total = 0.0
        for score, weight in zip(scores, self._weights):
            total += score * weight

        return await self._finish_forward_async(action, observation, total)

    @property
    def weights(self) -> List[float]:
        # [FUNCTION DOC]
        # Returns a copy of the weights.
        """Get the weights (read-only copy)."""
        return list(self._weights)


class RubricList(Rubric):
    # [CLASS DOC]
    # List container of child rubrics (registered as rubric_<i>); no aggregation.
    """Container for dynamic lists of rubrics.

    Analogous to nn.ModuleList. Does not define aggregation - use within
    a parent rubric that implements custom logic.

    Examples:

        ```python
        class MultiGameRubric(Rubric):
            def __init__(self, games: List[str]):
                super().__init__()
                self.games = RubricList([GameRubric(g) for g in games])

            def forward(self, action, obs) -> float:
                return self.games[obs.game_index](action, obs)
        ```
    """

    def __init__(self, rubrics: List[Rubric] = None):
        # [FUNCTION DOC]
        # Initializes an empty list and appends any given rubrics.
        """Initialize with optional list of rubrics.

        Args:
            rubrics (`list[``Rubric``]`, *optional*):
                List of rubrics to start with.
        """
        super().__init__()
        self._rubrics: List[Rubric] = []
        if rubrics is not None:
            for i, rubric in enumerate(rubrics):
                self.append(rubric)

    def forward(self, action: Any, observation: Any) -> float:
        # [FUNCTION DOC]
        # Always raises NotImplementedError; aggregation belongs in a parent rubric.
        """RubricList does not define aggregation - override in parent."""
        raise NotImplementedError(
            "RubricList.forward() is not implemented. "
            "Use RubricList within a parent rubric that defines aggregation."
        )

    def append(self, rubric: Rubric) -> None:
        # [FUNCTION DOC]
        # Registers the rubric as child "rubric_<index>" and appends it.
        """Add a rubric to the list."""
        index = len(self._rubrics)
        setattr(self, f"rubric_{index}", rubric)
        self._rubrics.append(rubric)

    def extend(self, rubrics: List[Rubric]) -> None:
        # [FUNCTION DOC]
        # Appends each rubric in order.
        """Add multiple rubrics to the list."""
        for rubric in rubrics:
            self.append(rubric)

    def __len__(self) -> int:
        # [FUNCTION DOC]
        # Returns the number of rubrics.
        return len(self._rubrics)

    def __getitem__(self, index: int) -> Rubric:
        # [FUNCTION DOC]
        # Returns the rubric at index.
        return self._rubrics[index]

    def __iter__(self) -> Iterator[Rubric]:
        # [FUNCTION DOC]
        # Returns an iterator over the rubrics.
        return iter(self._rubrics)


class RubricDict(Rubric):
    # [CLASS DOC]
    # Dict container of named child rubrics; no aggregation. Keys become attribute names.
    """Container for named rubrics with keyed access.

    Analogous to nn.ModuleDict. Enables keyed access for multi-task
    environments where different tasks require different rubrics.

    Examples:

        ```python
        class AtariRubric(Rubric):
            def __init__(self):
                super().__init__()
                self.games = RubricDict({
                    "pong": PongRubric(),
                    "breakout": BreakoutRubric(),
                    "space_invaders": SpaceInvadersRubric(),
                })

            def forward(self, action, obs) -> float:
                return self.games[obs.game_id](action, obs)

        # Access: env.rubric.games["pong"]
        ```
    """

    def __init__(self, rubrics: Dict[str, Rubric] = None):
        # [FUNCTION DOC]
        # Initializes an empty mapping and adds any given rubrics.
        """Initialize with optional dictionary of rubrics.

        Args:
            rubrics (`dict[str, ``Rubric``]`, *optional*):
                Dictionary mapping names to rubrics.
        """
        super().__init__()
        self._rubric_dict: Dict[str, Rubric] = {}
        if rubrics is not None:
            for name, rubric in rubrics.items():
                self[name] = rubric

    def forward(self, action: Any, observation: Any) -> float:
        # [FUNCTION DOC]
        # Always raises NotImplementedError; aggregation belongs in a parent rubric.
        """RubricDict does not define aggregation - override in parent."""
        raise NotImplementedError(
            "RubricDict.forward() is not implemented. "
            "Use RubricDict within a parent rubric that defines aggregation."
        )

    def __setitem__(self, key: str, rubric: Rubric) -> None:
        # [FUNCTION DOC]
        # Sets the rubric as attribute key (registering it as a child) and stores it in the mapping.
        # KNOWN ISSUE: because keys are set as attributes, a key that matches an existing attribute
        # or method name (e.g. "forward") overwrites it on the instance.
        """Add a rubric with the given key."""
        setattr(self, key, rubric)
        self._rubric_dict[key] = rubric

    def __getitem__(self, key: str) -> Rubric:
        # [FUNCTION DOC]
        # Returns the rubric for key. Raises KeyError if absent.
        """Get rubric by key."""
        return self._rubric_dict[key]

    def __contains__(self, key: str) -> bool:
        # [FUNCTION DOC]
        # True if key is present.
        """Check if key exists."""
        return key in self._rubric_dict

    def __len__(self) -> int:
        # [FUNCTION DOC]
        # Returns the number of entries.
        return len(self._rubric_dict)

    def __iter__(self) -> Iterator[str]:
        # [FUNCTION DOC]
        # Returns an iterator over keys.
        return iter(self._rubric_dict)

    def keys(self) -> Iterator[str]:
        # [FUNCTION DOC]
        # Returns an iterator over keys.
        """Iterate over keys."""
        return iter(self._rubric_dict.keys())

    def values(self) -> Iterator[Rubric]:
        # [FUNCTION DOC]
        # Returns an iterator over rubrics.
        """Iterate over rubrics."""
        return iter(self._rubric_dict.values())

    def items(self) -> Iterator[Tuple[str, Rubric]]:
        # [FUNCTION DOC]
        # Returns an iterator over (key, rubric) pairs.
        """Iterate over (key, rubric) pairs."""
        return iter(self._rubric_dict.items())

    def update(self, rubrics: Union[Dict[str, Rubric], Mapping[str, Rubric]]) -> None:
        # [FUNCTION DOC]
        # Adds every (name, rubric) from the mapping via __setitem__.
        """Update with rubrics from a dictionary."""
        for name, rubric in rubrics.items():
            self[name] = rubric
