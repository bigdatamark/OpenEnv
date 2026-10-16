# [MODULE DOC]
# Maintenance script: generates (or checks) the JSON Schema files for the discovery
# profile models (environment card, catalog snapshot, declaration) under
# src/openenv/discovery/schemas/<PROFILE_VERSION>/.
# SPDX-License-Identifier: BSD-3-Clause

"""Generate or check the packaged discovery profile schemas."""

import argparse
import json
from pathlib import Path

from openenv.discovery.metadata import DiscoveryDeclaration
from openenv.discovery.models import CatalogSnapshot, EnvironmentCard, PROFILE_VERSION


def main() -> int:
    # [FUNCTION DOC]
    # Entry point: for each model, builds its JSON schema by alias, adds draft 2020-12 $schema
    # and a GitHub raw $id, serializes it (indent 2, sorted keys, trailing newline). With
    # --check, records files that are missing or differ; otherwise writes them (creating the
    # directory). Returns: 1 (after printing the stale files) if any differ in check mode, else 0.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    destination = root / "src/openenv/discovery/schemas" / PROFILE_VERSION
    changed = []
    models = {
        "environment-card": EnvironmentCard,
        "catalog": CatalogSnapshot,
        "declaration": DiscoveryDeclaration,
    }
    for name, model in models.items():
        schema = model.model_json_schema(by_alias=True)
        schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
        schema["$id"] = (
            "https://raw.githubusercontent.com/huggingface/OpenEnv/main/"
            f"src/openenv/discovery/schemas/{PROFILE_VERSION}/{name}.schema.json"
        )
        content = json.dumps(schema, indent=2, sort_keys=True) + "\n"
        output = destination / f"{name}.schema.json"
        if arguments.check:
            if not output.exists() or output.read_text() != content:
                changed.append(str(output.relative_to(root)))
        else:
            destination.mkdir(parents=True, exist_ok=True)
            output.write_text(content)
    if changed:
        print("Discovery schemas need regeneration: " + ", ".join(changed))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
