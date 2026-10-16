# [PACKAGE DOC]
# FinQA server package. Lazily exposes FinQAEnvironment via a module-level attribute hook so importing the
# package does not load pandas and data dependencies.
# envs/finqa_env/server/__init__.py
"""Server-side components for the FinQA environment."""


def __getattr__(name):
    # [FUNCTION DOC]
    # HELPER: Module attribute hook: imports and returns FinQAEnvironment for that name.
    # Raises: AttributeError for any other name.
    if name == "FinQAEnvironment":
        from .finqa_environment import FinQAEnvironment

        return FinQAEnvironment
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["FinQAEnvironment"]
