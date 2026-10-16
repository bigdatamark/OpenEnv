# [MODULE DOC]
# Tests for run_validation (the end-to-end validation runner) and
# source_digest (deterministic package content hashing), including runtime
# grader discovery with a fake provider and collector, static-level verdicts,
# and hardening of the digest against special files and symlink swaps.
import hashlib
import os
import shutil

import pytest
from conftest import FIXTURES
from openenv.validation import runner
from openenv.validation.policy import load_policy
from openenv.validation.report import CheckResult, ValidationReport
from openenv.validation.runner import run_validation, source_digest
from openenv.validation.signature import SignatureError
from openenv.validation.types import CheckStatus, Lane, Level, Verdict
from support.runtime import evidence, FakeRuntimeProvider


@pytest.mark.parametrize(
    "mode", ["run", "skip-build", "source-change", "capability", "applies-to"]
)
def test_registered_runtime_graders_use_metadata_throughout_run(
    tmp_path, monkeypatch, mode
):
    # [FUNCTION DOC]
    # Parametrized over modes. Copies the served_probe fixture, registers two
    # Probe graders for existing policy ids with a dependency chain on
    # runtime.startup, patches the default registry and the evidence collector,
    # and runs validation at the runtime level with FakeRuntimeProvider.
    # Expects: for capability / applies-to modes the probes never run nor appear;
    # in "run" mode both run and pass; with skip-build they are skipped with
    # "unmet dependency" evidence; with source-change (the collector writes a file
    # into the package) they are skipped citing the changed source.
    package = tmp_path / "subject"
    shutil.copytree(FIXTURES / "runtime" / "served_probe", package)
    policy = load_policy("v2")
    registry = runner.default_grader_registry(policy)
    calls = []

    class Probe:
        # [CLASS DOC]
        # Runtime-level grader fake whose required capabilities include task_api only
        # in "capability" mode; records its id into calls when run.
        level = Level.RUNTIME
        requires_provider = frozenset()
        requires_capabilities = (
            frozenset({"task_api"}) if mode == "capability" else frozenset()
        )

        def __init__(self, check_id, dependency):
            # [FUNCTION DOC]
            # Stores the check id and a single-element depends_on tuple.
            self.check_id = check_id
            self.depends_on = (dependency,)

        def applies_to(self, manifest):
            # [FUNCTION DOC]
            # Returns False only in "applies-to" mode.
            return mode != "applies-to"

        def run(self, subject):
            # [FUNCTION DOC]
            # Appends its id to calls and returns a passing CheckResult.
            calls.append(self.check_id)
            return CheckResult(
                check_id=self.check_id, status=CheckStatus.PASS, duration_s=0
            )

    # Register test implementations against existing policy IDs. The runner must
    # discover them without adding their IDs to execution or dependency lists.
    probes = [
        Probe("runtime.network_policy", "runtime.startup"),
        Probe("runtime.host_containment", "runtime.network_policy"),
    ]
    for probe in probes:
        registry.register(probe)
    monkeypatch.setattr(runner, "default_grader_registry", lambda policy: registry)

    def collect(*args, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Fake collector returning empty evidence; in source-change mode it
        # first writes changed.txt into the package.
        if mode == "source-change":
            (package / "changed.txt").write_text("changed during collection")
        return evidence()

    monkeypatch.setattr(runner, "collect_runtime_evidence", collect)
    report = run_validation(
        package,
        max_level=Level.RUNTIME,
        provider=FakeRuntimeProvider(),
        skip_build=mode == "skip-build",
        policy=policy,
    )
    results = {result.check_id: result for result in report.results}
    if mode in {"capability", "applies-to"}:
        assert calls == []
        assert all(probe.check_id not in results for probe in probes)
        return
    assert calls == ([] if mode == "skip-build" else [p.check_id for p in probes])
    for probe in probes:
        result = results[probe.check_id]
        assert result.status is (
            CheckStatus.PASS if mode == "run" else CheckStatus.SKIP
        )
        if mode == "skip-build":
            assert result.evidence == ["unmet dependency: " + probe.depends_on[0]]
        elif mode == "source-change":
            assert result.evidence == [
                "unmet dependency: runtime.startup (package source changed during validation)"
            ]


def test_valid_package_passes_static_level():
    # [FUNCTION DOC]
    # served_min_pass at static level passes in the local lane with
    # static.manifest passing and the manifest embedded.
    report = run_validation(
        FIXTURES / "served_min_pass", max_level=Level.STATIC, skip_build=True
    )
    assert report.verdict is Verdict.PASS
    assert report.lane is Lane.LOCAL
    assert report.levels_run == [Level.STATIC]
    assert report.manifest is not None and report.manifest.name == "served-min-pass"
    by_id = {r.check_id: r for r in report.results}
    assert by_id["static.manifest"].status is CheckStatus.PASS


def test_report_round_trips_through_its_schema():
    # [FUNCTION DOC]
    # The produced report survives a JSON round trip.
    report = run_validation(
        FIXTURES / "served_min_pass", max_level=Level.STATIC, skip_build=True
    )
    assert ValidationReport.model_validate_json(report.model_dump_json()) == report


def test_broken_manifest_fails_static_manifest_with_evidence():
    # [FUNCTION DOC]
    # broken_manifest fails with a single failing static.manifest result with
    # evidence and no manifest.
    report = run_validation(
        FIXTURES / "broken_manifest", max_level=Level.STATIC, skip_build=True
    )
    assert report.verdict is Verdict.FAIL
    assert report.manifest is None
    (result,) = report.results
    assert result.check_id == "static.manifest"
    assert result.status is CheckStatus.FAIL
    assert result.evidence, "schema errors must surface as evidence"


def test_out_of_bounds_declaration_fails_static_manifest(tmp_path):
    # [FUNCTION DOC]
    # A floor_margin below the policy bound fails static.manifest citing
    # floor_margin.
    src = (FIXTURES / "served_min_pass" / "openenv.yaml").read_text()
    (tmp_path / "openenv.yaml").write_text(
        src.replace("floor_margin: 0.5", "floor_margin: 0.01")
    )
    report = run_validation(tmp_path, max_level=Level.STATIC, skip_build=True)
    assert report.verdict is Verdict.FAIL
    (result,) = report.results
    assert result.status is CheckStatus.FAIL
    assert "floor_margin" in "\n".join(result.evidence)


def test_ambiguous_package_raises_signature_error():
    # [FUNCTION DOC]
    # An ambiguous package raises SignatureError.
    with pytest.raises(SignatureError, match="ambiguous"):
        run_validation(FIXTURES / "ambiguous_package", max_level=Level.STATIC)


def test_report_embeds_the_pinned_policy_version():
    # [FUNCTION DOC]
    # The report records the supplied policy's version.
    policy = load_policy("v1")
    report = run_validation(
        FIXTURES / "served_min_pass",
        max_level=Level.STATIC,
        skip_build=True,
        policy=policy,
    )
    assert report.policy_version == policy.policy_version


def test_source_digest_is_deterministic_and_content_sensitive(tmp_path):
    # [FUNCTION DOC]
    # The digest is stable, 64 hex chars, and changes when a file is added.
    copy = tmp_path / "pkg"
    shutil.copytree(FIXTURES / "served_min_pass", copy)
    first = source_digest(copy)
    assert first == source_digest(copy)
    assert len(first) == 64
    (copy / "extra.txt").write_text("changed\n")
    assert source_digest(copy) != first


def test_source_digest_uses_portable_relative_paths(tmp_path):
    # [FUNCTION DOC]
    # For a single nested file the digest equals SHA-256 over the POSIX relative
    # path, a NUL, the contents and a NUL.
    package_root = tmp_path / "pkg"
    nested = package_root / "nested"
    nested.mkdir(parents=True)
    (nested / "file.txt").write_bytes(b"contents")

    expected = hashlib.sha256(b"nested/file.txt\0contents\0").hexdigest()
    assert source_digest(package_root) == expected


@pytest.mark.parametrize("flag", ["O_NONBLOCK", "O_NOFOLLOW"])
def test_source_digest_does_not_require_platform_open_flags(
    tmp_path, monkeypatch, flag
):
    # [FUNCTION DOC]
    # The digest still works when os.O_NONBLOCK or os.O_NOFOLLOW is absent.
    (tmp_path / "file.txt").write_bytes(b"contents")
    monkeypatch.delattr(os, flag, raising=False)

    assert len(source_digest(tmp_path)) == 64


@pytest.mark.parametrize("failure", [ValueError, OSError])
def test_initial_source_digest_failure_is_reported(monkeypatch, failure):
    # [FUNCTION DOC]
    # If source_digest raises ValueError/OSError, the run reports a single ERROR
    # result with generic evidence, empty digest and FAIL verdict, never parses
    # the package and never leaks the exception message.
    def fail_digest(*args):
        # [FUNCTION DOC]
        # HELPER: Raises the parametrized exception with a private path message.
        raise failure("private-source-path")

    def fail_parse(*args):
        # [FUNCTION DOC]
        # HELPER: Raises AssertionError if parsing is attempted.
        raise AssertionError("rejected source must not be parsed")

    monkeypatch.setattr("openenv.validation.runner.source_digest", fail_digest)
    monkeypatch.setattr(
        "openenv.validation.parsers.openenv_yaml.OpenEnvYamlParser.parse", fail_parse
    )
    report = run_validation(FIXTURES / "served_min_pass", max_level=Level.STATIC)

    (result,) = report.results
    assert report.source_digest == ""
    assert result.status is CheckStatus.ERROR
    assert (
        result.evidence[-1] == "package source could not be verified before validation"
    )
    assert report.verdict is Verdict.FAIL
    assert "private-source-path" not in report.model_dump_json()


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="platform has no named pipes")
def test_source_digest_rejects_named_pipes(tmp_path):
    # [FUNCTION DOC]
    # A FIFO in the package makes source_digest raise ValueError about regular
    # files (skipped where mkfifo is unavailable).
    os.mkfifo(tmp_path / "pipe")
    with pytest.raises(ValueError, match="regular files"):
        source_digest(tmp_path)


def test_source_digest_rejects_swapped_file_without_nofollow(tmp_path, monkeypatch):
    # [FUNCTION DOC]
    # Without O_NOFOLLOW, a file replaced by a symlink between stat and open is
    # detected and raises ValueError "changed before".
    package = tmp_path / "package"
    package.mkdir()
    source = package / "source.txt"
    source.write_text("public source")
    private = tmp_path / "private.txt"
    private.write_text("private content")
    original_open = os.open

    def swap_before_open(path, flags):
        # [FUNCTION DOC]
        # HELPER: Replaces the source file with a symlink to a private file, then
        # calls the real os.open.
        source.unlink()
        source.symlink_to(private)
        return original_open(path, flags)

    monkeypatch.delattr(os, "O_NOFOLLOW", raising=False)
    monkeypatch.setattr(os, "open", swap_before_open)
    with pytest.raises(ValueError, match="changed before"):
        source_digest(package)
