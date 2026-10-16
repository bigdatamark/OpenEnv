# [MODULE DOC]
# Proof-of-concept script (uv inline metadata) adding the ECHO env-token cross-entropy to TRL's GRPOTrainer. A tiny
# terminal environment is exposed to TRL as a single "run" tool; TRL marks tool output tokens with tool_mask 0, and the
# subclass adds a weighted CE on those tokens using the policy log-probs from GRPO's own forward. Reports held-out
# env-token CE on unseen commands before and after training. Side effects: sys.path change, model download, training,
# writes to echo-trl-demo output dir, stdout.
# KNOWN ISSUE: a training task and a held-out probe refer to config.ini, but the mini terminal only has config.yaml, so those commands return a "No such file" error.
# /// script
# dependencies = [
#     "trl",
#     "jmespath",
# ]
# ///

"""Proof-of-concept: ECHO on TRL's `GRPOTrainer` over an OpenEnv-style environment.

ECHO (https://huggingface.co/papers/2605.24517) keeps the usual GRPO loss on the
agent's action tokens and adds a small cross-entropy on the environment's response
tokens, so the policy also learns to predict the environment (a world model, almost
for free). TRL already exposes everything needed: `environment_factory` drives the
multi-turn rollout and marks the env tokens with `tool_mask == 0` (excluded from the
GRPO loss). ECHO is then a small `GRPOTrainer` subclass that adds a cross-entropy on
exactly those tokens, reusing the forward GRPO already ran.

This reproduces the paper's world-model result (Figure 3: held-out env-token
cross-entropy drops with ECHO, stays flat with vanilla GRPO), not its performance
results. It runs on CPU with a tiny model.

Run:
    python trl_echo_demo.py --world-model-coeff 0.1
    python trl_echo_demo.py --world-model-coeff 0
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn.functional as F
from datasets import Dataset

from trl import GRPOConfig, GRPOTrainer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mini_terminal_env import run_command  # noqa: E402

TASKS = [
    "Use the run tool to list the files, then tell me how many there are.",
    "Use the run tool to read app.log and report the WARN lines.",
    "Use the run tool to grep ERROR in app.log and summarize what you find.",
    "Use the run tool to count the lines in config.ini.",
]

# commands not used in TASKS, to score generalization rather than memorization
HELDOUT_PROBES = ["head app.log", "wc app.log", "cat config.ini"]


class TerminalToolEnv:
    # [CLASS DOC]
    # Environment class for TRL's environment_factory with a single tool method; tracks a reward that is 1.0 when the last command succeeded.
    """OpenEnv-style env exposed to TRL as a single `run` tool. The tool result is
    the environment's response, which TRL marks with `tool_mask == 0`."""

    def __init__(self):
        # [FUNCTION DOC]
        # Initialises reward to 0.0.
        self.reward = 0.0

    def reset(self, **kwargs) -> None:
        # [FUNCTION DOC]
        # Resets the reward to 0.0 and returns None (no initial observation). Ignores kwargs.
        self.reward = 0.0
        return None

    def run(self, command: str) -> str:
        # [FUNCTION DOC]
        # Runs the command in the mini terminal and returns its output; sets reward to 0.0 if the output starts with an error
        # prefix (sh:, cat:, grep:, wc:, head:), else 1.0.
        """Run a shell command in a small terminal and return its output.

        Args:
            command: The shell command to run, e.g. "grep WARN app.log".

        Returns:
            The terminal output.
        """
        out = run_command(command)
        self.reward = (
            0.0 if out.startswith(("sh:", "cat:", "grep:", "wc:", "head:")) else 1.0
        )
        return out


def reward_func(environments, **kwargs):
    # [FUNCTION DOC]
    # Reward function for TRL: returns each environment's current reward.
    return [env.reward for env in environments]


class EchoGRPOTrainer(GRPOTrainer):
    # [CLASS DOC]
    # GRPOTrainer subclass adding world_model_coeff times the mean negative log-prob over completion tokens whose tool_mask is
    # 0 (environment tokens). Captures the policy per-token log-probs from the gradient-carrying forward pass.
    """GRPO plus the ECHO env-token loss, reusing GRPO's own forward.

    Args:
        world_model_coeff (`float`, *optional*, defaults to `0.1`):
            Weight of the cross-entropy on environment tokens. `0` is vanilla GRPO.
    """

    def __init__(self, *args, world_model_coeff: float = 0.1, **kwargs):
        # [FUNCTION DOC]
        # Initialises the base trainer and stores the coefficient and an empty captured log-prob slot.
        super().__init__(*args, **kwargs)
        self.world_model_coeff = world_model_coeff
        self._policy_logps = None

    def _get_per_token_logps_and_entropies(self, *args, **kwargs):
        # [FUNCTION DOC]
        # Calls the base implementation and, if the returned log-probs require gradients, stores them for the loss. Returns the base result unchanged.
        out = super()._get_per_token_logps_and_entropies(*args, **kwargs)
        if out[0].requires_grad:  # the policy forward we backprop through
            self._policy_logps = out[0]
        return out

    def _compute_loss(self, model, inputs):
        # [FUNCTION DOC]
        # Computes the base GRPO loss; if the coefficient is 0 or inputs lack tool_mask returns it unchanged; otherwise adds the
        # coefficient times the env-token CE computed from the captured log-probs and completion_mask * (1 - tool_mask).
        # KNOWN ISSUE: relies on the captured log-probs from the most recent gradient forward; if none was captured (None), it raises AttributeError.
        loss = super()._compute_loss(model, inputs)
        if self.world_model_coeff == 0 or "tool_mask" not in inputs:
            return loss
        # env tokens = completion tokens GRPO masked out (tool_mask == 0)
        obs_mask = (inputs["completion_mask"] * (1 - inputs["tool_mask"])).to(
            self._policy_logps.dtype
        )
        env_ce = -(self._policy_logps * obs_mask).sum() / obs_mask.sum().clamp(min=1.0)
        return loss + self.world_model_coeff * env_ce


@torch.no_grad()
def heldout_env_ce(model, tokenizer) -> float:
    # [FUNCTION DOC]
    # Without gradients, for each held-out probe command computes the model's mean CE on the command output tokens given a
    # "$ <cmd>" prefix, and returns the average across probes (nats/token).
    """Mean cross-entropy on held-out terminal output (nats/token, lower is better)."""
    device = next(model.parameters()).device
    ces = []
    for cmd in HELDOUT_PROBES:
        out = run_command(cmd)
        prefix = f"$ {cmd}\n"
        prompt_len = tokenizer(prefix, return_tensors="pt").input_ids.shape[1]
        ids = tokenizer(prefix + out, return_tensors="pt").input_ids.to(device)
        logits = model(ids).logits[0]
        logp = F.log_softmax(logits[prompt_len - 1 : ids.shape[1] - 1], dim=-1)
        targets = ids[0, prompt_len:]
        ces.append(float(-logp[torch.arange(targets.shape[0]), targets].mean()))
    return sum(ces) / len(ces)


def main():
    # [FUNCTION DOC]
    # Parses model, coefficient and steps; builds a 256-prompt dataset from the four tasks, a small GRPOConfig (no
    # checkpoints, thinking disabled), and the EchoGRPOTrainer with TerminalToolEnv; measures held-out CE, trains, measures again and prints both.
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--world-model-coeff", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=30)
    args = parser.parse_args()

    dataset = Dataset.from_dict(
        {"prompt": [[{"role": "user", "content": t}] for t in TASKS] * 64}
    )

    # only non-default values: small batch/generations and a higher LR for a short
    # CPU run, no checkpoint writing, and Qwen3's thinking disabled
    config = GRPOConfig(
        output_dir="echo-trl-demo",
        per_device_train_batch_size=2,
        num_generations=2,
        max_steps=args.steps,
        learning_rate=1e-5,
        save_strategy="no",
        chat_template_kwargs={"enable_thinking": False},
    )

    trainer = EchoGRPOTrainer(
        model=args.model,
        reward_funcs=reward_func,
        args=config,
        train_dataset=dataset,
        environment_factory=TerminalToolEnv,
        world_model_coeff=args.world_model_coeff,
    )

    before = heldout_env_ce(trainer.model, trainer.processing_class)
    trainer.train()
    after = heldout_env_ce(trainer.model, trainer.processing_class)
    print(f"\nheld-out env-token CE: {before:.3f} -> {after:.3f} (lower is better)")


if __name__ == "__main__":
    main()
