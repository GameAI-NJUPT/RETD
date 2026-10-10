#!/usr/bin/env python3
"""Verify the extracted scientific ASTs and optional original source hashes."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def nodes(path: Path) -> dict[str, ast.AST]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.name: node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }


def digest(node: ast.AST) -> str:
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path,
                        help="Optional original experiments directory; never required for running Q1.")
    args = parser.parse_args()
    with (HERE / "provenance.json").open(encoding="utf-8") as handle:
        record = json.load(handle)
    cache: dict[str, dict[str, ast.AST]] = {}
    for item in record["preserved_ast_nodes"]:
        name = item["release"]
        if name not in cache:
            cache[name] = nodes(HERE / name)
        actual = digest(cache[name][item["name"]])
        if actual != item["sha256"]:
            raise AssertionError(f"AST mismatch: {name}:{item['name']}")
        if args.source_dir is not None:
            original = args.source_dir / Path(item["source"]).name
            if ast.dump(nodes(original)[item["name"]], include_attributes=False) != ast.dump(cache[name][item["name"]], include_attributes=False):
                raise AssertionError(f"Original AST mismatch: {item['name']}")
    tree = ast.parse((HERE / "numerical_core.py").read_text(encoding="utf-8"))
    constants = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in record["constants"]
    }
    for name, expected in record["constants"].items():
        actual = constants[name]
        if isinstance(actual, tuple):
            actual = list(actual)
        if actual != expected:
            raise AssertionError(f"Default constant mismatch: {name}")
    config = record["configuration_copy"]
    if sha256_file(HERE / config["file"]) != config["sha256"]:
        raise AssertionError("Frozen selected configuration hash mismatch")
    for path in HERE.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        parsed = ast.parse(text, filename=str(path))
        compile(parsed, str(path), "exec")
        for node in ast.walk(parsed):
            if isinstance(node, ast.Import):
                if any(alias.name.startswith("matplotlib") for alias in node.names):
                    raise AssertionError(f"Plotting import in {path.name}")
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("matplotlib"):
                raise AssertionError(f"Plotting import in {path.name}")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in {"savefig", "subplots", "fill_between", "imshow"}:
                    raise AssertionError(f"Plotting call in {path.name}")
    if args.source_dir is not None:
        for source in record["sources"]:
            path = args.source_dir.parent / source["path"]
            if sha256_file(path) != source["sha256"]:
                raise AssertionError(f"Original source hash mismatch: {source['path']}")
    print(f"PASS: {len(record['preserved_ast_nodes'])} original AST nodes, default constants, frozen configs, syntax and no plotting imports/calls.")
    print("This is a source-integrity/software check, not a full experimental replication.")


if __name__ == "__main__":
    main()
