# [MODULE DOC]
# Core enums for environment validation: levels, lanes, per-check statuses, policy severities,
# overall verdicts, package signature kinds, and provider capabilities. Graders emit statuses;
# only the severity policy maps checks to severities.

"""Core enums and small frozen types for environment validation (RFC 008).

The type split carries the central invariant: graders emit [`~openenv.validation.types.CheckStatus`];
only the severity policy maps a check to [`~openenv.validation.types.Severity`]. A grader has no
opinion about whether its failure blocks validation.
"""

from enum import Enum, IntEnum


class Level(IntEnum):
    # [CLASS DOC]
    # Integer enum of validation levels ordered by cost: STATIC=1, RUNTIME=2, SEMANTIC=3, STATISTICAL=4 (reserved).
    """
    Validation levels, ordered by cost budget.

    STATISTICAL is reserved: its check ids exist in the report schema and severity
    policy, but no local implementation ships in this repo.
    """

    STATIC = 1
    RUNTIME = 2
    SEMANTIC = 3
    STATISTICAL = 4


class Lane(str, Enum):
    # [CLASS DOC]
    # String enum of who runs a check: "local" or "hub".
    """
    Who runs a check.

    LOCAL checks run in the author's `openenv validate`. HUB checks are operator-run
    and are never referenced in local reports — an author is never shown a check they
    cannot red-to-green.
    """

    LOCAL = "local"
    HUB = "hub"


class CheckStatus(str, Enum):
    # [CLASS DOC]
    # String enum of grader results: pass, fail, skip, error.
    """
    What a grader may return for a single check.

    SKIP requires a reason (unmet dependency or missing capability). ERROR means the
    grader crashed or its evidence was malformed; the policy fails ERROR closed.
    """

    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"
    ERROR = "error"


class Severity(str, Enum):
    # [CLASS DOC]
    # String enum of policy severities: fail, warn, advisory.
    """What only the severity policy may assign to a check id."""

    FAIL = "fail"
    WARN = "warn"
    ADVISORY = "advisory"


class Verdict(str, Enum):
    # [CLASS DOC]
    # String enum of overall run outcomes: pass, warn, fail.
    """Overall outcome of a validation run. WARN means only warn/advisory findings."""

    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


class SignatureKind(str, Enum):
    # [CLASS DOC]
    # String enum of package formats whose values are the well-known filenames: openenv.yaml,
    # task.toml, task.md.
    """
    Recognized package formats, named by their well-known file.

    A manifest's signature is provenance for the report only — no grader, core or
    third-party, may branch on it.
    """

    OPENENV_SERVED = "openenv.yaml"
    HARBOR_TASK = "task.toml"
    POSTTRAIN_TASK = "task.md"


class ProviderCapability(str, Enum):
    # [CLASS DOC]
    # String enum of capabilities a validation provider may declare: network_policy, exec,
    # image_build, gpu, remote.
    """Capabilities a validation provider may declare; graders require them by name."""

    NETWORK_POLICY = "network_policy"
    EXEC = "exec"
    IMAGE_BUILD = "image_build"
    GPU = "gpu"
    REMOTE = "remote"
