# [MODULE DOC]
# Standalone subprocess entry point that runs untrusted AWM verifier code. Reads one JSON payload from stdin
# (verifier_code, function_name, initial_db_path, final_db_path, final_answer, mode "sql" or "code"), applies
# POSIX resource limits (CPU 20s, 512 MiB address space, 16 MiB files, 64 open files, no new processes), execs the
# code with a whitelisted builtins set and an import hook restricted to an allow-list of stdlib modules (os is
# replaced by a shim exposing only os.path), calls the named verifier function and writes one JSON result to stdout.
# KNOWN ISSUE: the sandbox is weak - whitelisted builtins such as getattr/setattr/type/vars/globals and allowed
# modules allow introspection-based escapes; it relies mainly on rlimits and the parent's timeout.
# KNOWN ISSUE: "print" is whitelisted and writes to the same stdout used for the JSON result, so a verifier that
# prints corrupts the output the parent parses.
"""
Out-of-process verifier runner.

Reads JSON payload from stdin, executes verifier code under OS-level resource
limits

Invoked by ``verifier.py`` via ``subprocess.run``

The runner imports a small whitelist of modules into the exec namespace
(sqlite3, json, re plus a curated builtins set) and intentionally does NOT
inject ``os`` — verifiers in the AWM dataset only need DB I/O, JSON, and
regex.

Protocol:
    stdin  (one JSON object):
        {
            "verifier_code": str,
            "function_name": str,
            "initial_db_path": str,
            "final_db_path": str,
            "final_answer": str | None,        # only used by code mode
            "mode": "sql" | "code",
        }
    stdout (one JSON object):
        For mode == "sql": the verifier's return value, or
            {"execution_status": "error", "error_message": str}
        For mode == "code":
            {"result": "complete"|"others",
             "execution_status": "success"|"error",
             "raw_result"?: <verifier return>,
             "error_message"?: str}
"""

from __future__ import annotations

import builtins as _builtins
import inspect
import json
import re as _re
import sqlite3
import sys
from typing import Any

try:
    import resource
except ImportError:
    resource = None  # Windows — resource limits not enforced


# ---------------------------------------------------------------------------
# Resource limits applied before exec()
# ---------------------------------------------------------------------------
DEFAULT_CPU_SECONDS = 20
DEFAULT_ADDRESS_SPACE_BYTES = 512 * 1024 * 1024  # 512 MiB
DEFAULT_FILE_SIZE_BYTES = 16 * 1024 * 1024  # 16 MiB
DEFAULT_OPEN_FILES = 64
DEFAULT_NPROC = 0  # disallow fork/exec from inside the verifier where supported


# builtin whitelist
_SAFE_BUILTIN_NAMES = frozenset(
    {
        # constants
        "True",
        "False",
        "None",
        # numeric / sequence / mapping types
        "int",
        "float",
        "complex",
        "bool",
        "str",
        "bytes",
        "bytearray",
        "list",
        "tuple",
        "dict",
        "set",
        "frozenset",
        "range",
        "slice",
        "memoryview",
        # iteration / functional helpers
        "iter",
        "next",
        "len",
        "enumerate",
        "zip",
        "map",
        "filter",
        "reversed",
        "sorted",
        "any",
        "all",
        "sum",
        "min",
        "max",
        # numeric helpers
        "abs",
        "round",
        "divmod",
        "pow",
        # type introspection (read-only style)
        "isinstance",
        "issubclass",
        "type",
        "hasattr",
        "getattr",
        "setattr",
        "delattr",
        "callable",
        "id",
        "vars",
        "dir",
        # globals/locals are needed by some idiomatic verifier patterns
        "globals",
        "locals",
        "__build_class__",
        # string / repr helpers
        "repr",
        "str",
        "ord",
        "chr",
        "format",
        "ascii",
        "hex",
        "oct",
        "bin",
        # io that's safe to keep (writes to runner stdout, captured by parent)
        "print",
        # exceptions verifiers commonly raise/catch
        "Exception",
        "ValueError",
        "TypeError",
        "KeyError",
        "IndexError",
        "AttributeError",
        "RuntimeError",
        "ArithmeticError",
        "ZeroDivisionError",
        "StopIteration",
        "AssertionError",
        "NotImplementedError",
        "LookupError",
        "ArithmeticError",
        "BaseException",
        "ImportError",
        "OverflowError",
        "FloatingPointError",
        "UnicodeError",
        "UnicodeDecodeError",
        "UnicodeEncodeError",
        "BufferError",
        "DeprecationWarning",
        "Warning",
        "UserWarning",
    }
)

# Modules verifier code is allowed to import
_ALLOWED_IMPORTS = frozenset(
    {
        "sqlite3",
        "json",
        "re",
        "math",
        "decimal",
        "fractions",
        "datetime",
        "time",
        "calendar",
        "zoneinfo",
        "collections",
        "collections.abc",
        "itertools",
        "functools",
        "operator",
        "string",
        "statistics",
        "heapq",
        "bisect",
        "uuid",
        "hashlib",
        "base64",
        "binascii",
        "unicodedata",
        "typing",
        "enum",
        "dataclasses",
        "copy",
        "pprint",
        "textwrap",
        "difflib",
        "contextlib",
        "traceback",
        "ipaddress",
        "io",
        "csv",
        "urllib.parse",
        "dateutil",
        "pytz",
        "_strptime",
    }
)

# Sentinel injected into the user's exec namespace
_USER_NS_MARKER = "_AWM_SANDBOX_USER"


import os.path as _stdlib_ospath


class _SafeOsShim:
    # [CLASS DOC]
    # HELPER: Stand-in for the os module given to verifier code; exposes only the real os.path as attribute "path".
    """Drop-in for ``os`` exposing only ``os.path``."""

    path = _stdlib_ospath

    def __repr__(self) -> str:
        # [FUNCTION DOC]
        # Returns a fixed string describing the sandboxed shim.
        return "<sandboxed os shim — only os.path is available>"


_OS_SHIM = _SafeOsShim()


def _safe_import(
    name: str,
    globals: Any = None,
    locals: Any = None,
    fromlist: tuple = (),
    level: int = 0,
) -> Any:
    # [FUNCTION DOC]
    # HELPER: Replacement __import__ installed in the verifier's builtins.
    # Behavior: rejects relative imports (level != 0) with ImportError. If the immediate caller's globals carry the
    # sandbox marker (i.e. the import comes from verifier code), "os"/"os.*" returns the os shim and any module whose
    # root name and full name are both outside the allow-list raises ImportError. Otherwise delegates to the real
    # __import__ with the same arguments.
    # KNOWN ISSUE: only direct imports from user code are checked; allowed modules may import anything transitively,
    # and "from os import path" returns the shim (works), but "import os.path as p" binds the shim, not os.path.
    if level != 0:
        raise ImportError("Relative imports are not permitted in verifier code")

    caller = sys._getframe(1)
    is_user_caller = caller.f_globals.get(_USER_NS_MARKER) is True

    if is_user_caller:
        if name == "os" or name.startswith("os."):
            return _OS_SHIM
        root = name.split(".", 1)[0]
        if root not in _ALLOWED_IMPORTS and name not in _ALLOWED_IMPORTS:
            raise ImportError(f"Import of '{name}' is not permitted in verifier code")

    return _builtins.__import__(name, globals, locals, fromlist, level)


def _build_safe_builtins() -> dict[str, Any]:
    # [FUNCTION DOC]
    # HELPER: Returns a new dict of the whitelisted builtin names that exist in the real builtins, plus "__import__"
    # mapped to the restricted import function.
    real = _builtins.__dict__
    safe = {name: real[name] for name in _SAFE_BUILTIN_NAMES if name in real}
    safe["__import__"] = _safe_import
    return safe


# resource limits
def _apply_resource_limits() -> None:
    # [FUNCTION DOC]
    # HELPER: Sets soft and hard rlimits for CPU, address space, file size, open files and (if available) number of
    # processes using the module constants. No-op when the resource module is unavailable (Windows). Errors from
    # unsupported limits are silently ignored. Side effects: permanently lowers limits of the current process.
    if resource is None:
        return  # Windows — fall back to the parent's subprocess timeout.

    def _set(which: int, soft: int, hard: int | None = None) -> None:
        # [FUNCTION DOC]
        # HELPER: Applies one rlimit (hard defaults to soft); swallows ValueError/OSError/resource.error.
        try:
            resource.setrlimit(which, (soft, hard if hard is not None else soft))
        except (ValueError, OSError, resource.error):
            # Some limits (RLIMIT_NPROC, RLIMIT_AS) aren't supported on every
            # platform / inside every container; that's fine — the parent's
            # subprocess timeout still backstops runaway verifiers.
            pass

    _set(resource.RLIMIT_CPU, DEFAULT_CPU_SECONDS)
    _set(resource.RLIMIT_AS, DEFAULT_ADDRESS_SPACE_BYTES)
    _set(resource.RLIMIT_FSIZE, DEFAULT_FILE_SIZE_BYTES)
    _set(resource.RLIMIT_NOFILE, DEFAULT_OPEN_FILES)
    if hasattr(resource, "RLIMIT_NPROC"):
        _set(resource.RLIMIT_NPROC, DEFAULT_NPROC)


def _sanitize_for_json(obj: Any) -> Any:
    # [FUNCTION DOC]
    # HELPER: Recursively converts a value into something JSON-serializable: dicts are rebuilt (tuple keys become
    # their string form), lists/tuples become lists, bytes are UTF-8 decoded ignoring errors, JSON-serializable
    # scalars pass through, anything else becomes its str().
    if isinstance(obj, dict):
        return {
            str(k) if isinstance(k, tuple) else k: _sanitize_for_json(v)
            for k, v in obj.items()
        }
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(item) for item in obj]
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="ignore")
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return str(obj)


def _call_verifier_func(
    verify_func: Any,
    initial_db_path: str,
    final_db_path: str,
    final_answer: str | None,
) -> Any:
    # [FUNCTION DOC]
    # HELPER: Invokes the verifier with arguments matched to its signature. If it declares any of initial_db_path,
    # final_db_path, final_answer by name they are passed as keywords (final_answer None becomes ""). If none match and
    # it takes at least two parameters, passes the two DB paths positionally, plus final_answer ("" if None) when it
    # takes three or more. Returns whatever the verifier returns; propagates its exceptions.
    # KNOWN ISSUE: a function with zero or one non-matching parameter is called with no arguments.
    sig = inspect.signature(verify_func)
    params = sig.parameters

    kwargs: dict[str, Any] = {}
    if "initial_db_path" in params:
        kwargs["initial_db_path"] = initial_db_path
    if "final_db_path" in params:
        kwargs["final_db_path"] = final_db_path
    if "final_answer" in params:
        kwargs["final_answer"] = final_answer or ""

    if not kwargs and len(params) >= 2:
        args = [initial_db_path, final_db_path]
        if len(params) >= 3:
            args.append(final_answer or "")
        return verify_func(*args)

    return verify_func(**kwargs)


def _run(payload: dict[str, Any]) -> dict[str, Any]:
    # [FUNCTION DOC]
    # HELPER: Executes the verifier described by the payload and returns the result dict.
    # Behavior: builds a namespace pre-populated with sqlite3, json, re, the safe builtins, __name__ "__main__" and the
    # sandbox marker; execs the code; looks up function_name; calls it. Compile/lookup/execution failures return an
    # error dict ("execution_status" "error" + "error_message", plus "result" "others" in code mode).
    # Code mode: the return must be a dict with a "result" key, else an error dict; result values other than
    # "complete"/"others" are coerced to "others"; returns result, "success" status and the sanitized raw result.
    # SQL mode: returns the verifier's value unchanged if JSON-serializable, else sanitized.
    # Raises: KeyError if required payload keys are missing.
    # Side effects: executes arbitrary verifier code (which may read/write the SQLite files).
    mode = payload.get("mode")
    code = payload["verifier_code"]
    function_name = payload["function_name"]
    initial_db_path = payload["initial_db_path"]
    final_db_path = payload["final_db_path"]
    final_answer = payload.get("final_answer")

    namespace: dict[str, Any] = {
        "sqlite3": sqlite3,
        "json": json,
        "re": _re,
        "__builtins__": _build_safe_builtins(),
        "__name__": "__main__",
        _USER_NS_MARKER: True,
    }

    try:
        exec(code, namespace)  # noqa: S102 — sandboxed by rlimits + restricted builtins
    except Exception as e:
        if mode == "code":
            return {
                "result": "others",
                "execution_status": "error",
                "error_message": f"Compile error: {e}",
            }
        return {"execution_status": "error", "error_message": f"Compile error: {e}"}

    verify_func = namespace.get(function_name)
    if not callable(verify_func):
        if mode == "code":
            return {
                "result": "others",
                "execution_status": "error",
                "error_message": f"Function '{function_name}' not found",
            }
        return {
            "execution_status": "error",
            "error_message": f"Function '{function_name}' not found",
        }

    try:
        result = _call_verifier_func(
            verify_func, initial_db_path, final_db_path, final_answer
        )
    except Exception as e:
        if mode == "code":
            return {
                "result": "others",
                "execution_status": "error",
                "error_message": f"Execution error: {e}",
            }
        return {"execution_status": "error", "error_message": f"Execution error: {e}"}

    if mode == "code":
        if not isinstance(result, dict) or "result" not in result:
            return {
                "result": "others",
                "execution_status": "error",
                "error_message": f"Invalid return format: {type(result).__name__}",
            }
        result_value = result.get("result", "others")
        if result_value not in ("complete", "others"):
            result_value = "others"
        return {
            "result": result_value,
            "execution_status": "success",
            "raw_result": _sanitize_for_json(result),
        }

    # SQL mode: return whatever the verifier returned, JSON-safe
    try:
        json.dumps(result)
        return result
    except TypeError:
        return _sanitize_for_json(result)


def main() -> int:
    # [FUNCTION DOC]
    # Entry point: applies resource limits, parses stdin as JSON (bad JSON yields an error dict), runs the verifier,
    # and writes the JSON result to stdout (non-str values serialized via str; serialization failures yield an error
    # dict). Always returns exit code 0.
    _apply_resource_limits()

    try:
        payload = json.loads(sys.stdin.read())
    except json.JSONDecodeError as e:
        sys.stdout.write(
            json.dumps(
                {"execution_status": "error", "error_message": f"Bad payload: {e}"}
            )
        )
        return 0

    result = _run(payload)
    try:
        sys.stdout.write(json.dumps(result, default=str))
    except (TypeError, ValueError) as e:
        sys.stdout.write(
            json.dumps(
                {
                    "execution_status": "error",
                    "error_message": f"Result not JSON-serializable: {e}",
                }
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
