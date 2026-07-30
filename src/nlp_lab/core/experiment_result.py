from pathlib import Path
from typing import Any

from pydantic import Field, field_validator

from nlp_lab.core.config.common import StrictConfigModel, ensure_non_empty
from nlp_lab.core.run_artifacts import ErrorRecord, PredictionRecord, RuntimeMeasurements


class ExperimentArtifact(StrictConfigModel):
    name: str
    path: Path
    description: str | None = None

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        return ensure_non_empty(value)

    @field_validator("description")
    @classmethod
    def validate_description(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return ensure_non_empty(value)


class ExperimentResult(StrictConfigModel):
    metrics: dict[str, float] = Field(default_factory=dict)
    runtime: RuntimeMeasurements
    predictions: list[PredictionRecord] = Field(default_factory=list)
    errors: list[ErrorRecord] = Field(default_factory=list)
    artifacts: list[ExperimentArtifact] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @field_validator("metrics")
    @classmethod
    def validate_metric_names(cls, value: dict[str, float]) -> dict[str, float]:
        ambiguous_names = {"f1", "precision", "recall"}
        matched_names = sorted(set(value) & ambiguous_names)
        if matched_names:
            msg = (
                f"metric names must be explicit, found ambiguous names: {', '.join(matched_names)}"
            )
            raise ValueError(msg)
        return value

    @field_validator("notes")
    @classmethod
    def validate_notes(cls, value: list[str]) -> list[str]:
        return [ensure_non_empty(note) for note in value]

    def summary_lines(self) -> list[str]:
        lines: list[str] = []
        for artifact in self.artifacts:
            lines.append(f"- Artifact: {artifact.name} ({artifact.path})")
        for note in self.notes:
            lines.append(f"- {note}")
        return lines

    def to_serializable_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
