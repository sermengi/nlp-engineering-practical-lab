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


PathLike = str | Path
RawConfig = dict[str, Any]
