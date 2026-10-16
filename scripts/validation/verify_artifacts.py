#!/usr/bin/env python3
# [MODULE DOC]
# Verification script for a validation-runtime evidence directory: checks SHA256SUMS
# coverage and checksums, a successful run manifest, that the acceptance inventory matches the
# committed tests/validation_runtime/acceptance.json, and that junit.xml contains passing tests
# covering every required acceptance case. Run as a script with the directory path; prints the
# verified test count.
"""Verify complete evidence checksums and successful test execution."""

import argparse
import hashlib
import json
from pathlib import Path
from xml.etree import ElementTree

ACCEPTANCE = (
    Path(__file__).resolve().parents[2] / "tests/validation_runtime/acceptance.json"
)


def verify(root):
    # [FUNCTION DOC]
    # Verifies the artifact directory root (resolved):
    # - every SHA256SUMS line ("<sha>  <relative path>") names a unique regular file inside root
    #   with a matching SHA-256, and the recorded set equals all files under root except SHA256SUMS;
    # - run-manifest.json has success true;
    # - acceptance.json is byte-identical to the committed inventory and its SHA-256 equals the
    #   manifest's acceptance_inventory_sha256;
    # - the manifest's suite is "fast" or a suite in the inventory;
    # - junit.xml's testsuites have no failures, errors or skips and a nonzero total test count;
    # - every required case "<classname>::<name>" for the selected suite was observed.
    # Returns: total test count. Raises: ValueError on any violation; file, JSON, XML or key
    # errors for missing or malformed inputs.
    root = root.resolve()
    recorded = {}
    for line in (root / "SHA256SUMS").read_text().splitlines():
        checksum, relative = line.split("  ", 1)
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file() or relative in recorded:
            raise ValueError(f"Invalid artifact path: {relative}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
            raise ValueError(f"Artifact checksum mismatch: {relative}")
        recorded[relative] = checksum
    actual = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and path != root / "SHA256SUMS"
    }
    if set(recorded) != actual:
        raise ValueError("Artifact inventory differs from SHA256SUMS")
    manifest = json.loads((root / "run-manifest.json").read_text())
    if not manifest["success"]:
        raise ValueError("Run did not complete successfully")
    inventory_bytes = (root / "acceptance.json").read_bytes()
    if inventory_bytes != ACCEPTANCE.read_bytes():
        raise ValueError("Run does not match the committed acceptance inventory")
    if hashlib.sha256(inventory_bytes).hexdigest() != manifest.get(
        "acceptance_inventory_sha256"
    ):
        raise ValueError("Acceptance inventory digest differs from run manifest")
    inventory = json.loads(inventory_bytes)["suites"]
    selected_suite = manifest.get("suite")
    if selected_suite not in {"fast", *inventory}:
        raise ValueError("Unknown acceptance suite in run manifest")
    document = ElementTree.parse(root / "junit.xml").getroot()
    count = 0
    for suite in document.iter("testsuite"):
        count += int(suite.get("tests", "0"))
        if any(int(suite.get(key, "0")) for key in ("failures", "errors", "skipped")):
            raise ValueError("Required test suite contains failures, errors or skips")
    if not count:
        raise ValueError("Empty test suite cannot establish completion")
    observed = {
        f"{case.get('classname', '')}::{case.get('name', '')}"
        for case in document.iter("testcase")
    }
    missing = set(inventory.get(selected_suite, [])) - observed
    if missing:
        raise ValueError(
            "Missing required acceptance cases: " + ", ".join(sorted(missing))
        )
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    arguments = parser.parse_args()
    print(f"Verified {verify(arguments.path)} tests and all artifact hashes")
