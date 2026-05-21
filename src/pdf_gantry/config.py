"""Config loading, validation, and persistence."""

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


GANTRY_DIR = Path("~/.gantry").expanduser()
CONFIG_PATH = GANTRY_DIR / "config.yaml"
DEFAULT_PAPERS_DIR = Path("~/Library/Mobile Documents/com~apple~CloudDocs/Papers").expanduser()


@dataclass
class EmbeddingConfig:
    model: str = "nomic-ai/nomic-embed-text-v2-moe"
    dimensions: int = 768
    batch_size: int = 32


@dataclass
class ProcessingConfig:
    workers: int = 4
    scan_threshold: float = 0.8
    default_method: str = "pymupdf4llm"


@dataclass
class OutputConfig:
    format: str = "text"
    default_limit: int = 20


@dataclass
class Config:
    papers_dir: Path = field(default_factory=lambda: DEFAULT_PAPERS_DIR)
    index_dir: Path = field(default_factory=lambda: GANTRY_DIR)
    vault_dir: Path | None = None
    bib_path: Path | None = None
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @property
    def db_path(self) -> Path:
        return self.index_dir / "index.db"


def _expand(p: str | Path) -> Path:
    return Path(os.path.expanduser(str(p)))


def load_config() -> Config:
    """Load config from YAML file, env vars, with defaults as fallback."""
    cfg = Config()

    # Load from file if it exists
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            data = yaml.safe_load(f) or {}

        if "papers_dir" in data:
            cfg.papers_dir = _expand(data["papers_dir"])
        if "index_dir" in data:
            cfg.index_dir = _expand(data["index_dir"])
        if "vault_dir" in data and data["vault_dir"]:
            cfg.vault_dir = _expand(data["vault_dir"])
        if "bib_path" in data and data["bib_path"]:
            cfg.bib_path = _expand(data["bib_path"])

        if "embedding" in data and isinstance(data["embedding"], dict):
            for k, v in data["embedding"].items():
                if hasattr(cfg.embedding, k):
                    setattr(cfg.embedding, k, v)

        if "processing" in data and isinstance(data["processing"], dict):
            for k, v in data["processing"].items():
                if hasattr(cfg.processing, k):
                    setattr(cfg.processing, k, v)

        if "output" in data and isinstance(data["output"], dict):
            for k, v in data["output"].items():
                if hasattr(cfg.output, k):
                    setattr(cfg.output, k, v)

    # Environment variable overrides
    if env_papers := os.environ.get("GANTRY_PAPERS_DIR"):
        cfg.papers_dir = _expand(env_papers)
    if env_index := os.environ.get("GANTRY_INDEX_DIR"):
        cfg.index_dir = _expand(env_index)
    if env_vault := os.environ.get("GANTRY_VAULT_DIR"):
        cfg.vault_dir = _expand(env_vault)
    if env_bib := os.environ.get("GANTRY_BIB_PATH"):
        cfg.bib_path = _expand(env_bib)

    return cfg


def save_config(cfg: Config) -> None:
    """Write config to YAML file."""
    GANTRY_DIR.mkdir(parents=True, exist_ok=True)

    data: dict = {
        "papers_dir": str(cfg.papers_dir),
        "index_dir": str(cfg.index_dir),
    }
    if cfg.vault_dir:
        data["vault_dir"] = str(cfg.vault_dir)
    if cfg.bib_path:
        data["bib_path"] = str(cfg.bib_path)

    data["embedding"] = {
        "model": cfg.embedding.model,
        "dimensions": cfg.embedding.dimensions,
        "batch_size": cfg.embedding.batch_size,
    }
    data["processing"] = {
        "workers": cfg.processing.workers,
        "scan_threshold": cfg.processing.scan_threshold,
        "default_method": cfg.processing.default_method,
    }
    data["output"] = {
        "format": cfg.output.format,
        "default_limit": cfg.output.default_limit,
    }

    with open(CONFIG_PATH, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def set_config_value(key: str, value: str) -> Config:
    """Set a single config value and save."""
    cfg = load_config()
    if key == "papers_dir":
        cfg.papers_dir = _expand(value)
    elif key == "index_dir":
        cfg.index_dir = _expand(value)
    elif key == "vault_dir":
        cfg.vault_dir = _expand(value) if value else None
    elif key == "bib_path":
        cfg.bib_path = _expand(value) if value else None
    else:
        raise ValueError(f"Unknown config key: {key}")
    save_config(cfg)
    return cfg
