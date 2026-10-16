# [MODULE DOC]
# Standalone script run in a separate process to evaluate observations against JSON Schemas
# (Draft 2020-12) in isolation. Reads a JSON payload from stdin and writes a JSON list of problem
# strings to stdout. Input is capped at 32 MiB.

"""Disposable JSON Schema evaluator; invoked as a script to avoid package imports."""

import json
import sys

from jsonschema import Draft202012Validator
from referencing import Registry

# Quoting the 8 MiB raw trace costs at most 16 MiB. The remainder covers the
# 1 MiB combined schemas after conservative 6x normalization and 2x quoting, plus row
# metadata. Limits apply to bytes, independently of the text stream's encoding.
MAX_INPUT_BYTES = 32 * 1024 * 1024


def main():
    # [FUNCTION DOC]
    # Entry point. Best-effort sets CPU rlimit (3s soft/4s hard) and, on Linux, a 512 MiB address-space
    # limit. Reads stdin bytes; if over the cap, writes a single-problem list and returns.
    # Payload keys: schema_json (step schema string), optional reset_schema_json, observations (rows with
    # index, operation, response_json). Builds validators (checking schemas, using an empty registry to
    # block remote refs; reset falls back to step schema). For each row, merges reward and done into the
    # observation dict and records up to 20 problems as "exchange N: schema mismatch at <schema path>
    # (<validator>)" plus up to 5 missing required property names. Any exception adds a generic
    # "invalid or unevaluable observation schema" problem. Writes the problem list as JSON to stdout.
    # KNOWN ISSUE: invalid JSON on stdin raises before the try block, so the process exits with a
    # traceback instead of printing a problem list.
    """Evaluate bounded inputs, printing only error locations, never subject values."""
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CPU, (3, 4))
        if sys.platform == "linux":
            resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024,) * 2)
    except (ImportError, ValueError, OSError):
        pass  # The parent always enforces the independent wall-clock deadline.
    raw = sys.stdin.buffer.read(MAX_INPUT_BYTES + 1)
    if len(raw) > MAX_INPUT_BYTES:
        sys.stdout.write(json.dumps(["schema worker input exceeds its size budget"]))
        return
    payload = json.loads(raw)
    problems = []
    try:
        schemas = {"step": payload["schema_json"]}
        if payload.get("reset_schema_json") is not None:
            schemas["reset"] = payload["reset_schema_json"]
        validators = {}
        for operation, raw_schema in schemas.items():
            schema = json.loads(raw_schema)
            Draft202012Validator.check_schema(schema)
            # An empty registry prevents filesystem and network retrieval.
            validators[operation] = Draft202012Validator(schema, registry=Registry())
        validators.setdefault("reset", validators["step"])
        for row in payload["observations"]:
            data = json.loads(row["response_json"])["data"]
            observation = dict(data["observation"])
            observation.update(reward=data["reward"], done=data["done"])
            for error in validators[row["operation"]].iter_errors(observation):
                location = "/".join(str(x) for x in error.absolute_schema_path)[:160]
                missing = ""
                if error.validator == "required":
                    names = [
                        name
                        for name in error.validator_value
                        if name not in error.instance
                    ]
                    missing = "; missing properties: " + json.dumps(names[:5])[:160]
                problems.append(
                    f"exchange {row['index']}: schema mismatch at {location or '/'} "
                    f"({error.validator}){missing}"
                )
                if len(problems) >= 20:
                    break
            if len(problems) >= 20:
                break
    except Exception:
        problems.append("invalid or unevaluable observation schema")
    sys.stdout.write(json.dumps(problems))


if __name__ == "__main__":
    main()
