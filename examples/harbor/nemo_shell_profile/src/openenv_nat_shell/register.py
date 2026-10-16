# [MODULE DOC]
# Registers the sandbox shell as a NeMo Agent Toolkit function (config name "openenv_sandbox_shell") via the public plugin API.
"""Register a sandbox shell through NeMo's public function API."""

from nat.plugin_api import Builder, FunctionBaseConfig, FunctionInfo, register_function
from .shell import execute


class ShellConfig(FunctionBaseConfig, name="openenv_sandbox_shell"):
    # [CLASS DOC]
    # Function config for the shell tool with a per-command timeout in seconds (default 60.0).
    timeout: float = 60.0


@register_function(config_type=ShellConfig)
async def sandbox_shell(config: ShellConfig, builder: Builder):
    # [FUNCTION DOC]
    # Registered async generator function: yields a FunctionInfo built from an inner async shell tool, using the tool's docstring as description.
    async def shell(command: str) -> str:
        # [FUNCTION DOC]
        # Tool body: runs the bash command in the sandbox via execute with the configured timeout and returns its JSON result string.
        """Run a bash command in the task sandbox; returns exit code, stdout and stderr.

        Use this to inspect task files, run Python analysis and write the answer file.
        """
        return await execute(command, timeout=config.timeout)

    yield FunctionInfo.from_fn(shell, description=shell.__doc__)
