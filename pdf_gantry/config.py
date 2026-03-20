"""Configuration management for pdf-gantry."""

from pathlib import Path
from typing import Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class GantryConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="GANTRY_",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    papers_dir: Path = Path("~/Library/Mobile Documents/com~apple~CloudDocs/papers/").expanduser()
    index_dir: Path = Path("~/.pdf-gantry/").expanduser()
    vault_dir: Optional[Path] = None

    @field_validator("papers_dir", "index_dir", mode="before")
    @classmethod
    def expand_path(cls, v: object) -> Path:
        return Path(str(v)).expanduser()

    @field_validator("vault_dir", mode="before")
    @classmethod
    def expand_optional_path(cls, v: object) -> Optional[Path]:
        if v is None:
            return None
        return Path(str(v)).expanduser()


def load_config() -> GantryConfig:
    """Load configuration, with values from env vars or defaults."""
    return GantryConfig()
