# [MODULE DOC]
# Server-side GPU kernel optimisation environment. Loads reference PyTorch problems from level
# directories, presents one per episode, and evaluates submitted Triton/CUDA code with a local GPU
# evaluator (compilation, correctness against the reference, benchmark speedup, optional profiling).
# Rewards favour correct, faster kernels; episodes end when solved (correct with speedup above 1.05)
# or after max_turns.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
GPU Kernel Optimization Environment Implementation.

Server-side environment for evaluating CUDA/Triton kernels against
PyTorch reference implementations. Provides compilation checking,
correctness verification, and performance benchmarking.
"""

import os
import random
from pathlib import Path
from typing import Optional
from uuid import uuid4

# Support both in-repo and standalone imports
try:
    # In-repo imports (when running from OpenEnv repository)
    from openenv.core.env_server.interfaces import Environment
    from openenv.core.env_server.types import State

    from ..models import KernelAction, KernelObservation, KernelState
    from .evaluator import LocalGPUEvaluator
except ImportError:
    from models import KernelAction, KernelObservation, KernelState

    # Standalone imports (when environment is standalone with openenv from pip)
    from openenv.core.env_server.interfaces import Environment
    from openenv.core.env_server.types import State
    from server.evaluator import LocalGPUEvaluator


class Problem:
    # [CLASS DOC]
    # Plain record of a problem: id ("L<level>_<file stem>"), level, name, agent-facing description and reference code.
    """A kernel optimization problem."""

    def __init__(
        self, id: str, level: int, name: str, description: str, reference_code: str
    ):
        # [FUNCTION DOC]
        # Stores the five fields.
        self.id = id
        self.level = level
        self.name = name
        self.description = description
        self.reference_code = reference_code


class KernelOptEnvironment(Environment):
    # [CLASS DOC]
    # Kernel optimisation environment holding the problems directory, turn limit, GPU device, levels,
    # evaluator, loaded problems, KernelState, the current problem and per-episode feedback history.
    """
    GPU Kernel Optimization Environment.

    A reinforcement learning environment that trains LLMs to write optimized
    CUDA/Triton kernels. Agents submit kernel code and receive feedback including:
    - Compilation status and errors
    - Correctness against reference implementation
    - Speedup compared to PyTorch baseline
    - Profiling data from NSight Systems/Compute

    Requires local GPU with CUDA toolkit for full evaluation support.

    Example:
        >>> env = KernelOptEnvironment()
        >>> obs = env.reset(problem_id="L1_23_Softmax")
        >>> print(obs.problem_description)
        >>>
        >>> obs = env.step(KernelAction(code=triton_kernel))
        >>> print(f"Speedup: {obs.speedup}x")
    """

    def __init__(
        self,
        problems_dir: Optional[str] = None,
        max_turns: int = 10,
        gpu: str = "cuda:0",
        levels: Optional[list[int]] = None,
        atol: float = 0.05,
        rtol: float = 0.02,
        warmup_runs: int = 10,
        benchmark_runs: int = 100,
        enable_nsys: bool = True,
        enable_ncu: bool = False,
        timeout: int = 60,
    ):
        # [FUNCTION DOC]
        # Resolves the problems directory (argument or default lookup, which may raise FileNotFoundError),
        # creates the LocalGPUEvaluator with tolerance, benchmark, profiling and timeout settings, loads
        # problems for the given levels (default 1 to 10) and initialises empty episode state.
        """
        Initialize the kernel optimization environment.

        Args:
            problems_dir: Path to problems directory, or None for default
            max_turns: Maximum turns per episode
            gpu: CUDA device to use (e.g., "cuda:0")
            levels: List of problem levels to include
            atol: Absolute tolerance for correctness checking
            rtol: Relative tolerance for correctness checking
            warmup_runs: Number of warmup runs before benchmarking
            benchmark_runs: Number of benchmark runs for timing
            enable_nsys: Enable NSight Systems profiling
            enable_ncu: Enable NSight Compute profiling
            timeout: Timeout in seconds for kernel execution
        """
        self.problems_dir = (
            Path(problems_dir) if problems_dir else self._default_problems_dir()
        )
        self.max_turns = max_turns
        self.gpu = gpu
        self.levels = levels or [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        # Create evaluator
        self.evaluator = LocalGPUEvaluator(
            device=gpu,
            atol=atol,
            rtol=rtol,
            warmup_runs=warmup_runs,
            benchmark_runs=benchmark_runs,
            enable_nsys=enable_nsys,
            enable_ncu=enable_ncu,
            timeout=timeout,
        )

        # Load problems
        self.problems = self._load_problems()

        # Episode state
        self._state = KernelState()
        self._current_problem: Optional[Problem] = None
        self._feedbacks: list[str] = []
        self._reset_count = 0

    def _default_problems_dir(self) -> Path:
        # [FUNCTION DOC]
        # HELPER: Returns KERNRL_PROBLEMS_DIR if it names an existing path, else the package's "problems"
        # directory if present. Raises FileNotFoundError otherwise.
        """Default to problems directory relative to package."""
        env_dir = os.environ.get("KERNRL_PROBLEMS_DIR")
        if env_dir:
            p = Path(env_dir)
            if p.exists():
                return p

        # Check relative to this file
        pkg_problems = Path(__file__).parent.parent / "problems"
        if pkg_problems.exists():
            return pkg_problems

        raise FileNotFoundError(
            "No problems directory found. Set KERNRL_PROBLEMS_DIR or "
            "ensure 'problems/' exists in the package directory."
        )

    def _load_problems(self) -> list[Problem]:
        # [FUNCTION DOC]
        # HELPER: Reads every non-underscore .py file in each existing level<N> directory (sorted) and returns
        # Problem objects with generated descriptions. Missing level directories are skipped. Reads files.
        """Load all problems from the problems directory."""
        problems = []

        for level in self.levels:
            level_dir = self.problems_dir / f"level{level}"
            if not level_dir.exists():
                continue

            for problem_file in sorted(level_dir.glob("*.py")):
                if problem_file.name.startswith("_"):
                    continue

                code = problem_file.read_text()
                name = problem_file.stem

                problems.append(
                    Problem(
                        id=f"L{level}_{name}",
                        level=level,
                        name=name,
                        description=self._make_description(code, level),
                        reference_code=code,
                    )
                )

        return problems

    def _make_description(self, code: str, level: int) -> str:
        # [FUNCTION DOC]
        # HELPER: Returns the Markdown task description embedding the reference code, tolerances, output
        # format requirements and target device.
        """Create the problem description shown to the agent."""
        return f"""# GPU Kernel Optimization Task

## Objective
Write an optimized GPU kernel (using Triton or CUDA) that computes the same result
as the reference PyTorch implementation below, but faster.

## Reference Implementation
```python
{code}
```

## Requirements
1. Your kernel must produce the same output as the reference (atol={self.evaluator.atol}, rtol={self.evaluator.rtol})
2. Your kernel should be faster than the PyTorch baseline
3. You may use Triton (preferred) or raw CUDA

## Output Format
Provide a complete Python file with:
- A `Model` class with the same interface as the reference
- The `Model.forward()` method should use your optimized kernel
- Include any necessary imports (torch, triton, etc.)

## GPU Target
Device: {self.gpu}
"""

    def _get_gpu_info(self) -> str:
        # [FUNCTION DOC]
        # HELPER: Returns "<GPU name> (<memory> GB)" for the configured CUDA device via torch when available,
        # else "GPU: <device>". All errors are swallowed.
        """Get GPU info string."""
        try:
            import torch

            if torch.cuda.is_available():
                idx = int(self.gpu.split(":")[-1]) if ":" in self.gpu else 0
                name = torch.cuda.get_device_name(idx)
                mem = torch.cuda.get_device_properties(idx).total_memory / 1e9
                return f"{name} ({mem:.1f} GB)"
        except Exception:
            pass
        return f"GPU: {self.gpu}"

    def reset(self, problem_id: Optional[str] = None) -> KernelObservation:
        # [FUNCTION DOC]
        # Starts an episode on problem_id (exact id match first, then the first id containing it as a
        # substring) or a random problem from the global random module. Creates a fresh KernelState, clears
        # feedback and returns the initial observation with description, reference code, GPU info, turn 0 and
        # reward 0.0. Raises ValueError for an unknown id; IndexError if no problems are loaded and none is given.
        # KNOWN ISSUE: the substring fallback can pick an unintended problem (for example "L1_1" also matches "L1_10_...").
        """
        Reset environment and start a new episode.

        Args:
            problem_id: Specific problem to use, or None for random selection

        Returns:
            Initial KernelObservation with problem description
        """
        if problem_id:
            self._current_problem = next(
                (p for p in self.problems if p.id == problem_id), None
            )
            if not self._current_problem:
                # Try partial match
                self._current_problem = next(
                    (p for p in self.problems if problem_id in p.id), None
                )
            if not self._current_problem:
                raise ValueError(f"Problem {problem_id} not found")
        else:
            self._current_problem = random.choice(self.problems)

        self._state = KernelState(
            episode_id=str(uuid4()),
            step_count=0,
            problem_id=self._current_problem.id,
            turn=0,
            max_turns=self.max_turns,
            best_speedup=0.0,
            solved=False,
        )
        self._feedbacks = []
        self._reset_count += 1

        return KernelObservation(
            problem_id=self._current_problem.id,
            problem_description=self._current_problem.description,
            reference_code=self._current_problem.reference_code,
            gpu_info=self._get_gpu_info(),
            turn=0,
            max_turns=self.max_turns,
            feedback="",
            compilation_success=True,
            done=False,
            reward=0.0,
        )

    def step(self, action: KernelAction) -> KernelObservation:  # type: ignore[override]
        # [FUNCTION DOC]
        # Evaluates action.code against the current problem: increments turn and step_count, records feedback,
        # updates best speedup, marks solved when correct and speedup exceeds 1.05, computes the reward and
        # returns an observation with compilation, correctness, max difference, speedup, feedback, done and
        # metadata (step, solved). Raises ValueError for a non-KernelAction and RuntimeError before reset.
        # Side effects: compiles and runs code on the GPU via the evaluator. Steps after done are still evaluated.
        """
        Execute kernel code and return evaluation results.

        Args:
            action: KernelAction containing the kernel code

        Returns:
            KernelObservation with evaluation results
        """
        if not isinstance(action, KernelAction):
            raise ValueError(f"Expected KernelAction, got {type(action)}")

        if self._current_problem is None:
            raise RuntimeError("Must call reset() before step()")

        self._state.turn += 1
        self._state.step_count += 1

        # Evaluate the kernel
        eval_result = self.evaluator.evaluate(
            solution_code=action.code,
            reference_code=self._current_problem.reference_code,
            problem_id=self._current_problem.id,
            step=self._state.turn,
        )

        # Generate feedback
        feedback = eval_result.to_agent_feedback()
        self._feedbacks.append(feedback)

        # Update state
        if (
            eval_result.benchmark
            and eval_result.benchmark.speedup > self._state.best_speedup
        ):
            self._state.best_speedup = eval_result.benchmark.speedup

        if (
            eval_result.correctness
            and eval_result.correctness.correct
            and eval_result.benchmark
            and eval_result.benchmark.speedup > 1.05
        ):
            self._state.solved = True

        # Calculate reward
        reward = self._calculate_reward(eval_result)

        # Check if done
        done = self._state.turn >= self.max_turns or self._state.solved

        return KernelObservation(
            problem_id=self._current_problem.id,
            problem_description=self._current_problem.description,
            reference_code=self._current_problem.reference_code,
            gpu_info=self._get_gpu_info(),
            turn=self._state.turn,
            max_turns=self.max_turns,
            feedback=feedback,
            compilation_success=eval_result.compilation.success,
            compilation_error=eval_result.compilation.error,
            correctness_pass=eval_result.correctness.correct
            if eval_result.correctness
            else None,
            max_diff=eval_result.correctness.max_diff
            if eval_result.correctness
            else None,
            speedup=eval_result.benchmark.speedup if eval_result.benchmark else None,
            done=done,
            reward=reward,
            metadata={"step": self._state.turn, "solved": self._state.solved},
        )

    def _calculate_reward(self, eval_result) -> float:
        # [FUNCTION DOC]
        # HELPER: Reward: -0.5 if compilation failed; -0.25 if correctness was checked and failed; with a
        # non-zero speedup, speedup - 1 capped at 2.0 when faster, else half of (speedup - 1); otherwise 0.0.
        # KNOWN ISSUE: if no correctness result exists but a benchmark does, speedup is rewarded without
        # verifying correctness.
        """Calculate reward based on evaluation results."""
        if not eval_result.compilation.success:
            return -0.5  # Penalty for compilation failure

        if eval_result.correctness and not eval_result.correctness.correct:
            return -0.25  # Penalty for incorrect output

        if eval_result.benchmark and eval_result.benchmark.speedup:
            # Reward proportional to speedup
            speedup = eval_result.benchmark.speedup
            if speedup > 1.0:
                return min(speedup - 1.0, 2.0)  # Cap reward at 2.0
            else:
                return (speedup - 1.0) * 0.5  # Smaller penalty for being slower

        return 0.0

    @property
    def state(self) -> KernelState:
        # [FUNCTION DOC]
        # Property: current KernelState.
        """Get current environment state."""
        return self._state

    @property
    def done(self) -> bool:
        # [FUNCTION DOC]
        # Property: True when the turn limit is reached or the problem is solved.
        """Check if episode is done."""
        return self._state.turn >= self.max_turns or self._state.solved

    @property
    def reward(self) -> float:
        # [FUNCTION DOC]
        # Property: always 0.0 (the real reward is returned by step).
        """Get reward for current state."""
        return 0.0  # Actual reward is returned in step()

    def list_problems(self) -> list[str]:
        # [FUNCTION DOC]
        # Returns the list of loaded problem ids.
        """List all available problem IDs."""
        return [p.id for p in self.problems]

    @property
    def num_problems(self) -> int:
        # [FUNCTION DOC]
        # Property: number of loaded problems.
        """Get number of available problems."""
        return len(self.problems)
