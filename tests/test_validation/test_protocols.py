# [MODULE DOC]
# Tests that minimal fakes satisfy the validation Parser, Grader,
# RunningSubject and ValidationProvider runtime-checkable protocols, and covers
# ParserRegistry dispatch, GraderRegistry registration/selection rules and
# entry-point loading of graders.
from pathlib import Path

import pytest
from conftest import load_fixture_manifest
from openenv.validation.graders import (
    ENTRY_POINT_GROUP,
    Grader,
    GraderRegistry,
    Subject,
)
from openenv.validation.manifest import NormalizedManifest
from openenv.validation.parsers import Parser, ParserRegistry
from openenv.validation.providers import ExecResult, RunningSubject, ValidationProvider
from openenv.validation.report import CheckResult
from openenv.validation.signature import UnsupportedPackageError
from openenv.validation.types import (
    CheckStatus,
    Level,
    ProviderCapability,
    SignatureKind,
)


class FakeParser:
    # [CLASS DOC]
    # Parser fake for the OPENENV_SERVED signature.
    signature = SignatureKind.OPENENV_SERVED

    def parse(self, package_root: Path) -> NormalizedManifest:
        # [FUNCTION DOC]
        # Ignores package_root and returns the served_min_pass fixture manifest as a
        # NormalizedManifest.
        return NormalizedManifest.model_validate(
            load_fixture_manifest("served_min_pass")
        )


class FakeGrader:
    # [CLASS DOC]
    # Configurable grader fake with check id, level, required manifest
    # capabilities and a fixed applies_to answer; requires no provider
    # capabilities and has no dependencies.
    def __init__(
        self,
        check_id="static.manifest",
        level=Level.STATIC,
        requires_capabilities=frozenset(),
        applies=True,
    ):
        # [FUNCTION DOC]
        # Stores the given check_id, level, requires_capabilities and applies flag and
        # sets empty requires_provider / depends_on.
        self.check_id = check_id
        self.level = level
        self.requires_capabilities = requires_capabilities
        self.requires_provider = frozenset()
        self.depends_on = ()
        self._applies = applies

    def applies_to(self, manifest: NormalizedManifest) -> bool:
        # [FUNCTION DOC]
        # Returns the configured applies flag, ignoring the manifest.
        return self._applies

    def run(self, subject: Subject) -> CheckResult:
        # [FUNCTION DOC]
        # Returns a passing CheckResult for its check id with zero duration; ignores
        # the subject.
        return CheckResult(
            check_id=self.check_id, status=CheckStatus.PASS, duration_s=0.0
        )


class FakeEntryPoints(list):
    # [CLASS DOC]
    # List subclass emulating the importlib.metadata EntryPoints selectable API.
    def select(self, **params):
        # [FUNCTION DOC]
        # Asserts it is called only with group=ENTRY_POINT_GROUP and returns itself.
        assert params == {"group": ENTRY_POINT_GROUP}
        return self


class FakeRunningSubject:
    # [CLASS DOC]
    # Running-subject fake with a fixed localhost base_url and no-op behavior.
    base_url = "http://localhost:8000"

    def inspect(self):
        # [FUNCTION DOC]
        # Returns an empty dict.
        return {}

    def logs(self, max_bytes=65536):
        # [FUNCTION DOC]
        # Returns an empty string.
        return ""

    def exec(self, argv: list, timeout_s: float) -> ExecResult:
        # [FUNCTION DOC]
        # Returns a successful empty ExecResult; arguments ignored.
        return ExecResult(exit_code=0, stdout="", stderr="", duration_s=0.0)

    def stop(self) -> None:
        # [FUNCTION DOC]
        # Does nothing.
        pass


class FakeProvider:
    # [CLASS DOC]
    # Provider fake named "fake" supporting public networking with
    # NETWORK_POLICY and EXEC capabilities.
    name = "fake"
    supported_network_modes = frozenset({"public"})
    capabilities = frozenset(
        {ProviderCapability.NETWORK_POLICY, ProviderCapability.EXEC}
    )

    def build(self, root, execution):
        # [FUNCTION DOC]
        # Returns a fixed fake sha256 image id; arguments ignored.
        return "sha256:" + "a" * 64

    def start(self, spec):
        # [FUNCTION DOC]
        # Returns a new FakeRunningSubject; spec ignored.
        return FakeRunningSubject()


@pytest.fixture()
def manifest():
    # [FUNCTION DOC]
    # Fixture returning the served_min_pass NormalizedManifest.
    return NormalizedManifest.model_validate(load_fixture_manifest("served_min_pass"))


def test_fakes_conform_to_the_protocols():
    # [FUNCTION DOC]
    # Each fake is an instance of its runtime-checkable protocol.
    assert isinstance(FakeParser(), Parser)
    assert isinstance(FakeGrader(), Grader)
    assert isinstance(FakeRunningSubject(), RunningSubject)
    assert isinstance(FakeProvider(), ValidationProvider)


def test_parser_registry_dispatches_by_signature():
    # [FUNCTION DOC]
    # A registered parser is returned for its signature.
    registry = ParserRegistry()
    parser = FakeParser()
    registry.register(parser)
    assert registry.parser_for(SignatureKind.OPENENV_SERVED) is parser


def test_unregistered_signature_raises_parser_not_implemented():
    # [FUNCTION DOC]
    # Requesting an unregistered signature raises UnsupportedPackageError with
    # category "parser-not-implemented".
    registry = ParserRegistry()
    registry.register(FakeParser())
    with pytest.raises(UnsupportedPackageError) as exc_info:
        registry.parser_for(SignatureKind.POSTTRAIN_TASK)
    assert exc_info.value.category == "parser-not-implemented"


def test_grader_registry_rejects_duplicate_check_ids():
    # [FUNCTION DOC]
    # Registering two graders with the same id raises ValueError.
    registry = GraderRegistry()
    registry.register(FakeGrader())
    with pytest.raises(ValueError, match="already registered"):
        registry.register(FakeGrader())


def test_selection_respects_the_level_ceiling(manifest):
    # [FUNCTION DOC]
    # select() excludes graders above max_level.
    registry = GraderRegistry()
    registry.register(FakeGrader(check_id="static.manifest", level=Level.STATIC))
    registry.register(FakeGrader(check_id="semantic.oracle_max", level=Level.SEMANTIC))
    selected = registry.select(manifest, max_level=Level.STATIC)
    assert [g.check_id for g in selected] == ["static.manifest"]


def test_selection_requires_truthy_capabilities(manifest):
    # [FUNCTION DOC]
    # A grader requiring task_api is not selected when the manifest lacks it.
    registry = GraderRegistry()
    registry.register(
        FakeGrader(
            check_id="runtime.task_declaration_accuracy",
            level=Level.RUNTIME,
            requires_capabilities=frozenset({"task_api"}),
        )
    )
    assert manifest.capabilities.task_api is False
    assert registry.select(manifest, max_level=Level.SEMANTIC) == []


def test_selection_respects_applies_to(manifest):
    # [FUNCTION DOC]
    # A grader whose applies_to is False is not selected.
    registry = GraderRegistry()
    registry.register(FakeGrader(check_id="static.manifest", applies=False))
    assert registry.select(manifest, max_level=Level.SEMANTIC) == []


def test_selection_is_ordered_by_level_then_check_id(manifest):
    # [FUNCTION DOC]
    # Selected graders are sorted by level, then check id.
    registry = GraderRegistry()
    registry.register(FakeGrader(check_id="semantic.oracle_max", level=Level.SEMANTIC))
    registry.register(FakeGrader(check_id="static.manifest", level=Level.STATIC))
    registry.register(FakeGrader(check_id="static.layout", level=Level.STATIC))
    selected = registry.select(manifest, max_level=Level.SEMANTIC)
    assert [g.check_id for g in selected] == [
        "static.layout",
        "static.manifest",
        "semantic.oracle_max",
    ]


def test_selection_never_sees_the_signature(manifest):
    # [FUNCTION DOC]
    # Selection results are identical regardless of the manifest signature.
    registry = GraderRegistry()
    registry.register(FakeGrader(check_id="static.manifest", level=Level.STATIC))
    as_harbor = manifest.model_copy(update={"signature": SignatureKind.HARBOR_TASK})
    served_ids = [
        g.check_id for g in registry.select(manifest, max_level=Level.SEMANTIC)
    ]
    harbor_ids = [
        g.check_id for g in registry.select(as_harbor, max_level=Level.SEMANTIC)
    ]
    assert served_ids == harbor_ids


def test_subject_is_frozen(manifest, tmp_path):
    # [FUNCTION DOC]
    # Assigning to a Subject attribute raises.
    subject = Subject(
        root=tmp_path,
        manifest=manifest,
        image_ref=None,
        running=None,
        outputs_dir=tmp_path / "outputs",
    )
    with pytest.raises(Exception):
        subject.image_ref = "other"  # type: ignore[misc]


def test_provider_start_returns_a_running_subject():
    # [FUNCTION DOC]
    # FakeProvider.start returns a RunningSubject whose exec succeeds.
    # KNOWN ISSUE: passes a plain string where a LaunchSpec is expected; works only
    # because the fake ignores its argument.
    running = FakeProvider().start("img:latest")
    assert isinstance(running, RunningSubject)
    assert running.exec(["true"], timeout_s=1.0).exit_code == 0


def test_register_rejects_unknown_capability_names():
    # [FUNCTION DOC]
    # Registering a grader requiring an unknown capability name raises ValueError
    # naming it.
    registry = GraderRegistry()
    with pytest.raises(ValueError, match="telepathy"):
        registry.register(FakeGrader(requires_capabilities=frozenset({"telepathy"})))


def test_entry_point_grader_classes_are_instantiated(manifest, monkeypatch):
    # [FUNCTION DOC]
    # With entry_points patched to yield a class, load_entry_points registers one
    # instance of it, which is selected and runs.
    class FakeEntryPoint:
        # [CLASS DOC]
        # Entry point fake whose load returns the FakeGrader class.
        name = "fake-class"

        def load(self):
            # [FUNCTION DOC]
            # Returns the FakeGrader class.
            return FakeGrader

    import openenv.validation.graders as graders_module

    monkeypatch.setattr(
        graders_module, "entry_points", lambda: FakeEntryPoints([FakeEntryPoint()])
    )
    registry = GraderRegistry()
    assert registry.load_entry_points() == 1
    (grader,) = registry.select(manifest, max_level=Level.SEMANTIC)
    assert grader.run(None).status is CheckStatus.PASS


def test_entry_point_grader_instances_register_as_is(manifest, monkeypatch):
    # [FUNCTION DOC]
    # An entry point yielding a grader instance registers that exact instance.
    instance = FakeGrader()

    class FakeEntryPoint:
        # [CLASS DOC]
        # Entry point fake whose load returns a prebuilt grader instance.
        name = "fake-instance"

        def load(self):
            # [FUNCTION DOC]
            # Returns the captured grader instance.
            return instance

    import openenv.validation.graders as graders_module

    monkeypatch.setattr(
        graders_module, "entry_points", lambda: FakeEntryPoints([FakeEntryPoint()])
    )
    registry = GraderRegistry()
    assert registry.load_entry_points() == 1
    assert registry.select(manifest, max_level=Level.SEMANTIC) == [instance]


def test_entry_point_loading_supports_the_legacy_mapping_api(manifest, monkeypatch):
    # [FUNCTION DOC]
    # When entry_points returns a dict keyed by group (old API), loading still
    # registers the instance.
    instance = FakeGrader()

    class FakeEntryPoint:
        # [CLASS DOC]
        # Entry point fake (no name) whose load returns a grader instance.
        def load(self):
            # [FUNCTION DOC]
            # Returns the captured grader instance.
            return instance

    import openenv.validation.graders as graders_module

    monkeypatch.setattr(
        graders_module,
        "entry_points",
        lambda: {ENTRY_POINT_GROUP: [FakeEntryPoint()]},
    )
    registry = GraderRegistry()
    assert registry.load_entry_points() == 1
    assert registry.select(manifest, max_level=Level.SEMANTIC) == [instance]
