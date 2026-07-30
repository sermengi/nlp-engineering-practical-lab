from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictConfigModel(BaseModel):
    """Base model for repository configs."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


def ensure_non_empty(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        msg = "must not be empty"
        raise ValueError(msg)
    return normalized


class ProjectConfig(StrictConfigModel):
    name: str = Field(..., description="Human-readable project name.")
    description: str | None = Field(default=None, description="Optional project description.")
    config_version: str = Field(default="1", description="Version of the config schema.")

    @field_validator("name", "config_version")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        return ensure_non_empty(value)

    @field_validator("description")
    @classmethod
    def validate_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return ensure_non_empty(value)


class LoggingConfig(StrictConfigModel):
    level: str = "INFO"
    save_console_log: bool = True

    @field_validator("level")
    @classmethod
    def validate_level(cls, value: str) -> str:
        normalized = ensure_non_empty(value).upper()
        allowed_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if normalized not in allowed_levels:
            msg = f"logging level must be one of: {', '.join(sorted(allowed_levels))}"
            raise ValueError(msg)
        return normalized


class CachePathsConfig(StrictConfigModel):
    root: Path | None = None
    huggingface: Path | None = None
    datasets: Path | None = None
    models: Path | None = None


class RunNamingConfig(StrictConfigModel):
    strategy: str = "timestamp"
    prefix: str | None = None

    @field_validator("strategy")
    @classmethod
    def validate_strategy(cls, value: str) -> str:
        normalized = ensure_non_empty(value)
        allowed_strategies = {"timestamp", "slug", "manual"}
        if normalized not in allowed_strategies:
            msg = f"run naming strategy must be one of: {', '.join(sorted(allowed_strategies))}"
            raise ValueError(msg)
        return normalized

    @field_validator("prefix")
    @classmethod
    def validate_prefix(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return ensure_non_empty(value)


PathLike = str | Path
RawConfig = dict[str, Any]
