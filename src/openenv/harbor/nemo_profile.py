# [MODULE DOC]
# Opt-in Harbor agent profile that runs NeMo's ReAct agent with a single sandbox shell
# tool, for qualifying file-based tasks. Imports Harbor's NemoAgent at import time.
"""Opt-in NeMo ReAct configuration for qualification on file-based tasks."""

import yaml
from harbor.agents.installed.nemo_agent import NemoAgent


class NemoShellProfile(NemoAgent):
    # [CLASS DOC]
    # Harbor NemoAgent subclass that keeps Harbor's install/endpoint setup but rewrites
    # the generated NeMo YAML config to use a ReAct workflow with only a shell tool.
    """Reuse Harbor's installation and endpoint configuration with a native NeMo agent."""

    def _generate_config_yaml(self, model_name: str, api_key: str) -> str:
        # [FUNCTION DOC]
        # Builds the NeMo agent YAML: parses the parent's generated YAML, takes the first
        # configured LLM name, sets its temperature to 0.8, replaces all functions with one
        # "shell" tool (type openenv_sandbox_shell, 60s timeout) and replaces the workflow with
        # a react_agent using native tool calling, max 17 tool calls, history 1000 and fixed
        # extra instructions. Inputs: model_name, api_key (passed to the parent).
        # Returns: the YAML string (key order preserved). Raises: KeyError/StopIteration if
        # the parent config has no "llms" entries.
        config = yaml.safe_load(super()._generate_config_yaml(model_name, api_key))
        llm_name = next(iter(config["llms"]))
        config["llms"][llm_name]["temperature"] = 0.8
        config["functions"] = {
            "shell": {"_type": "openenv_sandbox_shell", "timeout": 60}
        }
        config["workflow"] = {
            "_type": "react_agent",
            "llm_name": llm_name,
            "tool_names": ["shell"],
            "use_native_tool_calling": True,
            "max_tool_calls": 17,
            "max_history": 1000,
            "additional_instructions": "Use the shell tool to inspect task files and complete the requested work. Follow the task's submission protocol.",
        }
        return yaml.safe_dump(config, sort_keys=False)
