"""The release tooling: the version stays in one piece, and bumps go where they should."""

import importlib.util
import re
from pathlib import Path

import pytest

import kroma

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("bump_version", ROOT / "scripts" / "bump_version.py")
bump = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bump)


def test_every_file_declares_the_same_version():
    assert bump.current_version() == kroma.__version__


@pytest.mark.parametrize(
    ("part", "expected"), [("patch", "1.2.4"), ("minor", "1.3.0"), ("major", "2.0.0")]
)
def test_next_version(part, expected):
    assert bump.next_version("1.2.3", part) == expected


@pytest.mark.parametrize(("version", "part"), [("1.2", "patch"), ("1.2.3rc1", "patch"), ("1.2.3", "huge")])
def test_bad_input_is_refused(version, part):
    with pytest.raises(SystemExit):
        bump.next_version(version, part)


def fake_repo(tmp_path, monkeypatch, first="0.3.0", second="0.3.0"):
    pyproject, init = tmp_path / "pyproject.toml", tmp_path / "__init__.py"
    pyproject.write_text(f'[project]\nname = "kroma-om"\nversion = "{first}"\ntarget-version = "x"\n')
    init.write_text(f'"""Doc."""\n\n__version__ = "{second}"\n')
    monkeypatch.setattr(bump, "FILES", {
        pyproject: re.compile(r'^version = "([^"]+)"$', re.M),
        init: re.compile(r'^__version__ = "([^"]+)"$', re.M),
    })
    return pyproject, init


def test_a_bump_rewrites_only_the_version(tmp_path, monkeypatch):
    pyproject, init = fake_repo(tmp_path, monkeypatch)
    assert bump.main("minor") == "0.4.0"
    assert pyproject.read_text() == '[project]\nname = "kroma-om"\nversion = "0.4.0"\ntarget-version = "x"\n'
    assert init.read_text() == '"""Doc."""\n\n__version__ = "0.4.0"\n'


def test_files_that_disagree_stop_the_bump(tmp_path, monkeypatch):
    pyproject, _ = fake_repo(tmp_path, monkeypatch, second="0.2.0")
    with pytest.raises(SystemExit, match="disagree"):
        bump.main("patch")
    assert 'version = "0.3.0"' in pyproject.read_text()
