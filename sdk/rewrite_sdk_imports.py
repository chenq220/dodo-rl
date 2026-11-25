#!/usr/bin/env python
"""将 SDK 构建副本中的工程内导入限定到 dodorl_sdk.dodorl。"""
from __future__ import annotations

import argparse
import re
from pathlib import Path


RUNTIME_PREFIX = "dodorl_sdk.dodorl"
RUNTIME_ROOTS = (
    "algos",
    "common",
    "config",
    "engine",
    "envs",
    "models",
    "portal",
    "roles",
    "rpc",
    "sample",
    "scheduler",
    "utils",
)
SPECIAL_ROOTS = {}

ROOT_PATTERN = "|".join(map(re.escape, (*RUNTIME_ROOTS, *SPECIAL_ROOTS)))
FROM_PATTERN = re.compile(
    rf"(?m)^(?P<prefix>\s*from\s+)(?P<root>{ROOT_PATTERN})(?P<tail>(?:\.[A-Za-z_]\w*)*\s+import\b)"
)
IMPORT_PATTERN = re.compile(
    rf"(?m)^(?P<prefix>\s*import\s+)(?P<root>{ROOT_PATTERN})(?P<tail>(?:\.[A-Za-z_]\w*)*(?:\s+as\s+[A-Za-z_]\w*)?\s*)$"
)
PLUGIN_PATH_PATTERN = re.compile(
    rf"(?P<quote>['\"])(?P<root>{'|'.join(map(re.escape, RUNTIME_ROOTS))})"
    rf"(?P<tail>(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)(?P=quote)"
)


def qualified_root(root: str) -> str:
    return SPECIAL_ROOTS.get(root, f"{RUNTIME_PREFIX}.{root}")


def rewrite_text(text: str) -> tuple[str, int]:
    replacements = 0

    def replace_import(match: re.Match[str]) -> str:
        nonlocal replacements
        replacements += 1
        return f"{match.group('prefix')}{qualified_root(match.group('root'))}{match.group('tail')}"

    def replace_plugin_path(match: re.Match[str]) -> str:
        nonlocal replacements
        replacements += 1
        quote = match.group("quote")
        return f"{quote}{qualified_root(match.group('root'))}{match.group('tail')}{quote}"

    text = FROM_PATTERN.sub(replace_import, text)
    text = IMPORT_PATTERN.sub(replace_import, text)
    text = PLUGIN_PATH_PATTERN.sub(replace_plugin_path, text)
    return text, replacements


def rewrite_tree(root: Path) -> tuple[int, int]:
    changed_files = 0
    replacements = 0
    for path in sorted(root.rglob("*.py")):
        original = path.read_text(encoding="utf-8")
        rewritten, count = rewrite_text(original)
        if count:
            path.write_text(rewritten, encoding="utf-8")
            changed_files += 1
            replacements += count
    return changed_files, replacements


def find_unqualified_imports(root: Path) -> list[str]:
    violations: list[str] = []
    for path in sorted(root.rglob("*.py")):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if FROM_PATTERN.match(line) or IMPORT_PATTERN.match(line) or PLUGIN_PATH_PATTERN.search(line):
                violations.append(f"{path.relative_to(root)}:{line_number}: {line.strip()}")
    return violations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()

    changed_files, replacements = rewrite_tree(args.root)
    violations = find_unqualified_imports(args.root)
    if violations:
        raise SystemExit("Unqualified SDK runtime imports remain:\n" + "\n".join(violations))
    print(f"Qualified {replacements} imports/plugin paths in {changed_files} files")


if __name__ == "__main__":
    main()
