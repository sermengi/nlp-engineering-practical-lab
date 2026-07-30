from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator

from nlp_lab.core.config.common import StrictConfigModel, ensure_non_empty

ExecutionEnvironment = Literal["local", "modal", "ci"]
DevicePreference = Literal["auto", "cpu", "cuda", "mps"]
PaddingStrategy = Literal["max_length", "longest", "do_not_pad"]
AverageMethod = Literal["micro", "macro", "weighted", "binary", "samples"]
CacheBehavior = Literal["use", "refresh", "disabled"]


class RuntimeConfig(StrictConfigModel):
    seed: int = Field(default=42, ge=0)
    output_root: Path = Field(default=Path("outputs"))
    deterministic: bool = False
    environment: ExecutionEnvironment = "local"


class ModelConfig(StrictConfigModel):
    model_id: str
    revision: str | None = None
    dtype: str | None = None
    trust_remote_code: bool = False
    cache_behavior: CacheBehavior = "use"

    @field_validator("model_id")
    @classmethod
    def validate_model_id(cls, value: str) -> str:
        return ensure_non_empty(value)

    @field_validator("revision", "dtype")
    @classmethod
    def validate_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return ensure_non_empty(value)


class DatasetConfig(StrictConfigModel):
    dataset_id: str | None = None
    local_path: Path | None = None
    split: str = "train"
    text_column: str = "text"
    label_column: str | None = "label"
    prediction_column: str = "prediction"
    sample_limit: int | None = Field(default=None, gt=0)
    revision: str | None = None

    @model_validator(mode="after")
    def validate_dataset_source(self) -> "DatasetConfig":
        if self.dataset_id is None and self.local_path is None:
            msg = "either dataset_id or local_path must be provided"
            raise ValueError(msg)
        return self

    @field_validator("dataset_id", "split", "text_column", "prediction_column", "revision")
    @classmethod
    def validate_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return ensure_non_empty(value)

    @field_validator("label_column")
    @classmethod
    def validate_optional_column(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return ensure_non_empty(value)


class PreprocessingConfig(StrictConfigModel):
    max_length: int = Field(default=512, gt=0)
    truncation: bool = True
    padding: PaddingStrategy = "max_length"


class InferenceConfig(StrictConfigModel):
    batch_size: int = Field(default=8, gt=0)
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    device: DevicePreference = "auto"


class EvaluationConfig(StrictConfigModel):
    metrics: list[str] = Field(default_factory=lambda: ["accuracy"])
    average: AverageMethod = "macro"
    save_predictions: bool = True
    save_errors: bool = True

    @field_validator("metrics")
    @classmethod
    def validate_metrics(cls, value: list[str]) -> list[str]:
        if not value:
            msg = "at least one metric must be configured"
            raise ValueError(msg)
        return [ensure_non_empty(metric) for metric in value]
