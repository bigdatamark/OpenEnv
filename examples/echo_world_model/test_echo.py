# [MODULE DOC]
# Pytest unit tests for the ECHO loss and the per-token role masks (torch only, no model downloads): lambda 0 equals
# the GRPO term, verifier-free equals coeff times env CE, uniform logits give CE of log V, differentiability, and mask
# alignment/disjointness including world_loss_target "all".
"""Unit tests for the ECHO loss + the per-token role masks.

Run from this directory with the example venv:  python -m pytest test_echo.py -q
(They use torch; they are intentionally separate from OpenEnv's core test suite,
which does not depend on torch.)
"""

from __future__ import annotations

import torch

from echo_loss import echo_loss
from trajectory import (
    ACTION,
    CONTEXT,
    ENV_OUTPUT,
    WARNING,
    Segment,
    Trajectory,
    tokenize_trajectory,
)


def _fake_logits(B, T, V, seed=0):
    # [FUNCTION DOC]
    # HELPER: Returns seeded random logits of shape B x T x V that require gradients.
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, T, V, generator=g, requires_grad=True)


class _CharTokenizer:
    # [CLASS DOC]
    # HELPER: Character tokenizer mapping each character to its code point modulo 256.
    """Minimal tokenizer: one id per character (no model download needed)."""

    def __call__(self, text, add_special_tokens=False):
        # [FUNCTION DOC]
        # Returns a dict with "input_ids" for the text; add_special_tokens is ignored.
        return {"input_ids": [ord(c) % 256 for c in text]}


def test_lambda_zero_is_vanilla_grpo():
    # [FUNCTION DOC]
    # Checks that with coefficient 0 the loss equals the GRPO term while l_env is still reported positive.
    """world_model_coeff=0 ⇒ the env term drops out entirely (loss == GRPO term)."""
    B, T, V = 2, 6, 50
    logits = _fake_logits(B, T, V)
    ids = torch.randint(0, V, (B, T))
    am = torch.zeros(B, T, dtype=torch.bool)
    am[:, 1:3] = True
    om = torch.zeros(B, T, dtype=torch.bool)
    om[:, 3:5] = True
    adv = torch.tensor([1.0, -1.0])
    loss, s = echo_loss(logits, ids, am, om, adv, world_model_coeff=0.0)
    assert abs(loss.item() - s["l_grpo"]) < 1e-5
    assert s["l_env"] > 0.0  # still reported, just not added


def test_verifier_free_is_env_only():
    # [FUNCTION DOC]
    # Checks that with use_rl false the GRPO term is 0 and the loss equals 0.5 times the env CE.
    """use_rl=False ⇒ GRPO term is exactly zero; loss == coeff * env-CE."""
    B, T, V = 2, 6, 50
    logits = _fake_logits(B, T, V)
    ids = torch.randint(0, V, (B, T))
    am = torch.zeros(B, T, dtype=torch.bool)
    am[:, 1:3] = True
    om = torch.zeros(B, T, dtype=torch.bool)
    om[:, 3:5] = True
    adv = torch.tensor([1.0, -1.0])
    loss, s = echo_loss(logits, ids, am, om, adv, world_model_coeff=0.5, use_rl=False)
    assert s["l_grpo"] == 0.0
    assert abs(loss.item() - 0.5 * s["l_env"]) < 1e-5


def test_env_ce_matches_uniform_baseline():
    # [FUNCTION DOC]
    # Checks that all-zero logits yield an env CE of log(V).
    """With ~uniform logits the env cross-entropy is ≈ log(V)."""
    B, T, V = 1, 5, 100
    logits = torch.zeros(B, T, V)  # uniform → CE = log V
    ids = torch.randint(0, V, (B, T))
    om = torch.zeros(B, T, dtype=torch.bool)
    om[:, 1:] = True
    am = torch.zeros(B, T, dtype=torch.bool)
    _, s = echo_loss(logits, ids, am, om, torch.zeros(B), use_rl=False)
    import math

    assert abs(s["l_env"] - math.log(V)) < 1e-4


def test_loss_is_differentiable():
    # [FUNCTION DOC]
    # Checks backpropagation populates gradients on the logits.
    B, T, V = 2, 6, 30
    logits = _fake_logits(B, T, V)
    ids = torch.randint(0, V, (B, T))
    am = torch.zeros(B, T, dtype=torch.bool)
    am[:, 1:3] = True
    om = torch.zeros(B, T, dtype=torch.bool)
    om[:, 3:5] = True
    loss, _ = echo_loss(logits, ids, am, om, torch.tensor([1.0, 0.5]))
    loss.backward()
    assert logits.grad is not None


def test_role_masks_are_disjoint_and_aligned():
    # [FUNCTION DOC]
    # Checks mask lengths match input ids, action and observation masks do not overlap, and counts for env-only observation, warning and action tokens.
    tok = _CharTokenizer()
    traj = Trajectory(
        segments=[
            Segment(CONTEXT, "T:"),
            Segment(ACTION, "ls"),
            Segment(WARNING, "[stdout]"),
            Segment(ENV_OUTPUT, "a b"),
        ],
        reward=1.0,
    )
    t = tokenize_trajectory(tok, traj, world_loss_target="env_only")
    n = t["input_ids"].numel()
    # masks align to input length and never overlap
    assert t["action_mask"].numel() == n
    assert not (t["action_mask"] & t["obs_mask"]).any()
    # env_only excludes the warning tokens from the obs (world-model) target
    assert t["obs_mask"].sum() == 3  # "a b"
    assert t["warning_mask"].sum() == len("[stdout]")
    # action mask covers exactly the action segment
    assert t["action_mask"].sum() == len("ls")


def test_world_loss_target_all_includes_warnings():
    # [FUNCTION DOC]
    # Checks that "all" adds warning tokens to the observation mask compared with env_only.
    tok = _CharTokenizer()
    traj = Trajectory(
        segments=[Segment(WARNING, "[w]"), Segment(ENV_OUTPUT, "xy")], reward=0.0
    )
    env_only = tokenize_trajectory(tok, traj, world_loss_target="env_only")
    all_ = tokenize_trajectory(tok, traj, world_loss_target="all")
    assert env_only["obs_mask"].sum() == 2  # only "xy"
    assert all_["obs_mask"].sum() == 2 + len("[w]")  # warnings included
