# [MODULE DOC]
# Tests for openenv._load_package_version, which resolves the installed distribution version
# (preferring "openenv", falling back to "openenv-core", else "0.0.0"). The metadata.version
# lookup is monkeypatched.

"""Tests for package version resolution."""

from __future__ import annotations

from importlib import metadata

import openenv


def test_load_package_version_prefers_openenv(monkeypatch) -> None:
    # [FUNCTION DOC]
    # When "openenv" is installed its version is returned and no other name is queried.
    calls: list[str] = []

    def fake_version(distribution_name: str) -> str:
        # [FUNCTION DOC]
        # HELPER: Records the queried name; returns "0.3.0" for "openenv", else raises PackageNotFoundError.
        calls.append(distribution_name)
        if distribution_name == "openenv":
            return "0.3.0"
        raise metadata.PackageNotFoundError

    monkeypatch.setattr(openenv.metadata, "version", fake_version)

    assert openenv._load_package_version() == "0.3.0"
    assert calls == ["openenv"]


def test_load_package_version_falls_back_to_openenv_core(monkeypatch) -> None:
    # [FUNCTION DOC]
    # When "openenv" is missing the "openenv-core" version is returned.
    def fake_version(distribution_name: str) -> str:
        # [FUNCTION DOC]
        # HELPER: Raises PackageNotFoundError for "openenv", returns "0.2.0" for "openenv-core", and
        # raises AssertionError for anything else.
        if distribution_name == "openenv":
            raise metadata.PackageNotFoundError
        if distribution_name == "openenv-core":
            return "0.2.0"
        raise AssertionError(f"Unexpected distribution name: {distribution_name}")

    monkeypatch.setattr(openenv.metadata, "version", fake_version)

    assert openenv._load_package_version() == "0.2.0"


def test_load_package_version_returns_zero_when_uninstalled(monkeypatch) -> None:
    # [FUNCTION DOC]
    # When neither is installed "0.0.0" is returned.
    monkeypatch.setattr(
        openenv.metadata,
        "version",
        lambda distribution_name: (_ for _ in ()).throw(metadata.PackageNotFoundError),
    )

    assert openenv._load_package_version() == "0.0.0"
