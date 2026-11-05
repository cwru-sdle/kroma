#!/usr/bin/env python3
"""Bump KROMA's version in every file that declares it, and print the new version.

Usage: scripts/bump_version.py patch|minor|major
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# The version lives in these files, and they must agree.
FILES = {
    ROOT / "pyproject.toml": re.compile(r'^version = "([^"]+)"$', re.M),
    ROOT / "src" / "kroma" / "__init__.py": re.compile(r'^__version__ = "([^"]+)"$', re.M),
}


def current_version() -> str:
    """Read the version.

    Returns:
        str: The version every file declares.

    Raises:
        SystemExit: When the files disagree, or one declares no version.
    """
    found = {path.name: pattern.search(path.read_text(encoding="utf-8")) for path, pattern in FILES.items()}
    versions = {match[1] for match in found.values() if match}
    if len(versions) != 1 or not all(found.values()):
        raise SystemExit(f"Versions disagree: { {name: m and m[1] for name, m in found.items()} }")
    return versions.pop()


def next_version(version: str, part: str) -> str:
    """Work out the next version.

    Args:
        version (str): The current ``major.minor.patch`` version.
        part (str): ``patch``, ``minor``, or ``major``.

    Returns:
        str: The next version.

    Raises:
        SystemExit: When the version or the part is not understood.
    """
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise SystemExit(f"Not a major.minor.patch version: {version}")
    major, minor, patch = map(int, version.split("."))
    if part == "patch":
        return f"{major}.{minor}.{patch + 1}"
    if part == "minor":
        return f"{major}.{minor + 1}.0"
    if part == "major":
        return f"{major + 1}.0.0"
    raise SystemExit(f"Use patch, minor, or major, not {part!r}")


def main(part: str) -> str:
    """Write the next version to every file.

    Args:
        part (str): ``patch``, ``minor``, or ``major``.

    Returns:
        str: The new version.
    """
    version = next_version(current_version(), part)
    for path, pattern in FILES.items():
        text = path.read_text(encoding="utf-8")
        start, end = pattern.search(text).span(1)
        path.write_text(text[:start] + version + text[end:], encoding="utf-8")
    return version


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    print(main(sys.argv[1]))
