# [MODULE DOC]
# Tests for GitServerClient's local git configuration: it must not write into the user's home
# directory and must pass quoted git arguments correctly. Runs real git commands in tmp dirs.

# SPDX-License-Identifier: BSD-3-Clause

"""Tests for GitServerClient git configuration."""

from pathlib import Path

from openenv.core.tools.git_server_client import GitServerClient


def test_git_config_does_not_touch_home(tmp_path, monkeypatch):
    # [FUNCTION DOC]
    # With HOME redirected to an empty temp dir, constructing the client leaves it empty, and git
    # commands run by the client report the client's identity (openenv@local.env).
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: home)

    client = GitServerClient(
        gitea_url="http://gitea:3000",
        username="openenv",
        password="secret",
        workspace_dir=str(tmp_path / "workspace"),
    )

    assert list(home.iterdir()) == []

    # Git commands run by the client use the client's own identity
    assert client.execute_git_command("init repo")[0] == 0
    code, stdout, _ = client.execute_git_command("config --get user.email", "repo")
    assert code == 0
    assert stdout.strip() == "openenv@local.env"


def test_execute_git_command_keeps_quoted_args(tmp_path, monkeypatch):
    # [FUNCTION DOC]
    # A commit message given in double quotes is passed as a single argument and appears intact in
    # the log.
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: home)

    client = GitServerClient(
        gitea_url="http://gitea:3000",
        username="openenv",
        password="secret",
        workspace_dir=str(tmp_path / "workspace"),
    )
    client.execute_git_command("init repo")

    code, _, stderr = client.execute_git_command(
        'commit --allow-empty -m "two words"', "repo"
    )
    assert code == 0, stderr
    _, stdout, _ = client.execute_git_command("log -1 --format=%s", "repo")
    assert stdout.strip() == "two words"
