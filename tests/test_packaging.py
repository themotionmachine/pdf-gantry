"""Tests guarding the declared dependency surface.

The default `gantry enrich` path uses the OpenAlex provider, which imports
`httpx`. A core install (no extras) must therefore ship `httpx` — otherwise the
default command crashes with ModuleNotFoundError on a fresh install.
"""

import tomllib
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def _core_dependencies() -> list[str]:
    data = tomllib.loads(PYPROJECT.read_text())
    return data["project"]["dependencies"]


def _dep_names(deps: list[str]) -> set[str]:
    names = set()
    for spec in deps:
        # Strip version specifiers / extras / markers to get the bare name.
        name = spec.split(";")[0].split("[")[0]
        for sep in (">=", "<=", "==", "~=", "!=", ">", "<", " "):
            name = name.split(sep)[0]
        names.add(name.strip().lower())
    return names


def test_httpx_is_a_core_dependency():
    """httpx backs the default enrich provider, so it must be a core dep."""
    assert "httpx" in _dep_names(_core_dependencies())


def test_openalex_module_imports():
    """The default-path provider module must import on a core install."""
    import pdf_gantry.openalex  # noqa: F401
