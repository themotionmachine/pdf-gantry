"""Configuration management for pdf-gantry."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[no-redef]

_CONFIG_PATH = Path("~/.pdf-gantry/config.toml").expanduser()

_DEFAULTS: dict = {
    "papers_dir": "~/Library/Mobile Documents/com~apple~CloudDocs/papers/",
    "index_dir": "~/.pdf-gantry/",
    "vault_dir": "",
}


@dataclass
class Config:
    papers_dir: Path = field(
        default_factory=lambda: Path(
            "~/Library/Mobile Documents/com~apple~CloudDocs/papers/"
        ).expanduser()
    )
    index_dir: Path = field(
        default_factory=lambda: Path("~/.pdf-gantry/").expanduser()
    )
    vault_dir: Optional[Path] = None


def load_config() -> Config:
    """Load configuration from ~/.pdf-gantry/config.toml, falling back to defaults."""
    data: dict = {}
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH, "rb") as fh:
            data = tomllib.load(fh)

    papers_dir = Path(data.get("papers_dir", _DEFAULTS["papers_dir"])).expanduser()
    index_dir = Path(data.get("index_dir", _DEFAULTS["index_dir"])).expanduser()
    vault_dir_raw = data.get("vault_dir", _DEFAULTS["vault_dir"])
    vault_dir: Optional[Path] = Path(vault_dir_raw).expanduser() if vault_dir_raw else None

    return Config(papers_dir=papers_dir, index_dir=index_dir, vault_dir=vault_dir)


def save_config(cfg: Config) -> None:
    """Persist configuration to ~/.pdf-gantry/config.toml."""
    _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f'papers_dir = "{cfg.papers_dir}"\n',
        f'index_dir = "{cfg.index_dir}"\n',
        f'vault_dir = "{cfg.vault_dir or ""}"\n',
    ]
    _CONFIG_PATH.write_text("".join(lines), encoding="utf-8")
