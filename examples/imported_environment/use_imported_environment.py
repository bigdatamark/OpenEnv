#!/usr/bin/env python3
# [MODULE DOC]
# Example that runs the OpenEnv importer CLI on the toy ORS environment, then loads the generated server-side
# environment class directly, resets, lists tools, answers the question, and prints a summary.
# Side effects: subprocess call, filesystem writes (temp or given directory), sys.path and sys.modules mutation.
"""Import and use a tiny ORS/OpenReward-style environment."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction


ENV_NAME = "imported_ors_demo"
EXAMPLE_DIR = Path(__file__).resolve().parent


def import_environment(source_dir: Path, output_dir: Path) -> Path:
    # [FUNCTION DOC]
    # Creates the output directory, deletes any previous generated package, runs "python -m openenv.cli import" for the
    # source directory with the demo name, and returns the generated package directory. Raises CalledProcessError on failure.
    """Run the OpenEnv importer and return the generated package directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_dir = output_dir / ENV_NAME
    if generated_dir.exists():
        shutil.rmtree(generated_dir)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "openenv.cli",
            "import",
            str(source_dir),
            "--name",
            ENV_NAME,
            "--output-dir",
            str(output_dir),
        ],
        check=True,
    )
    return generated_dir


def _clear_demo_modules() -> None:
    # [FUNCTION DOC]
    # HELPER: Removes the demo package and any "ors" modules (and submodules) from sys.modules so a fresh import happens.
    for module_name in list(sys.modules):
        if module_name == ENV_NAME or module_name.startswith(f"{ENV_NAME}."):
            sys.modules.pop(module_name, None)
        if module_name == "ors" or module_name.startswith("ors."):
            sys.modules.pop(module_name, None)


def run_imported_environment(output_dir: Path) -> dict[str, Any]:
    # [FUNCTION DOC]
    # Clears cached modules, temporarily prepends the output dir to sys.path, imports the generated environment class,
    # resets to split "train" index 0, lists tools, calls "answer" with "4", and returns a dict with prompt text, tool names,
    # result text, reward and done. Always restores sys.path and clears modules.
    """Use the generated OpenEnv wrapper directly."""
    _clear_demo_modules()
    sys.path.insert(0, str(output_dir))
    try:
        from imported_ors_demo.server.imported_ors_demo_environment import (  # type: ignore
            ImportedOrsDemoEnvironment,
        )

        env = ImportedOrsDemoEnvironment()
        reset_observation = env.reset(split="train", index=0)
        tools_observation = env.step(ListToolsAction())
        answer_observation = env.step(
            CallToolAction(tool_name="answer", arguments={"value": "4"})
        )

        return {
            "prompt": reset_observation.metadata["prompt"][0]["text"],
            "tools": [tool.name for tool in tools_observation.tools],
            "result": answer_observation.result["blocks"][0]["text"],
            "reward": answer_observation.reward,
            "done": answer_observation.done,
        }
    finally:
        try:
            sys.path.remove(str(output_dir))
        except ValueError:
            pass
        _clear_demo_modules()


def _print_summary(generated_dir: Path, summary: dict[str, Any]) -> None:
    # [FUNCTION DOC]
    # HELPER: Prints the generated directory and the summary fields.
    print(f"Generated environment: {generated_dir}")
    print(f"Prompt: {summary['prompt']}")
    print(f"Tools: {', '.join(summary['tools'])}")
    print(f"Tool result: {summary['result']}")
    print(f"Reward: {summary['reward']}")
    print(f"Done: {summary['done']}")


def main(argv: list[str] | None = None) -> None:
    # [FUNCTION DOC]
    # Parses an optional output directory; imports and runs the environment there, or in a temporary directory that is removed afterwards; prints the summary.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for the generated OpenEnv package. Defaults to a temp dir.",
    )
    args = parser.parse_args(argv)

    source_dir = EXAMPLE_DIR / "source"
    if args.output_dir is not None:
        output_dir = args.output_dir.resolve()
        generated_dir = import_environment(source_dir, output_dir)
        summary = run_imported_environment(output_dir)
        _print_summary(generated_dir, summary)
        return

    with TemporaryDirectory() as tmp:
        output_dir = Path(tmp)
        generated_dir = import_environment(source_dir, output_dir)
        summary = run_imported_environment(output_dir)
        _print_summary(generated_dir, summary)


if __name__ == "__main__":
    main()
