from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, field_validator, model_validator

from nlp_lab.core.config.common import StrictConfigModel, ensure_non_empty

ExecutionEnvironment = Literal["local", "modal", "ci"]
DevicePreference = Literal["auto", "cpu", "cuda", "mps"]
PaddingStrategy = Literal["dynamic", "max_length", "longest", "do_not_pad"]
AverageMethod = Literal["micro", "macro", "weighted", "binary", "samples"]
CacheBehavior = Literal["use", "refresh", "disabled"]
ModelDType = Literal["auto", "float32", "float16", "bfloat16"]


def default_averaging() -> list[AverageMethod]:
    return ["macro"]


class RuntimeConfig(StrictConfigModel):
    seed: int = Field(default=42, ge=0)
    output_root: Path = Field(default=Path("outputs"))
    deterministic: bool = False
    environment: ExecutionEnvironment = "local"

    @field_validator("output_root", mode="before")
    @classmethod
    def validate_output_root(cls, value: object) -> object:
        if isinstance(value, str):
            ensure_non_empty(value)
        return value


class ModelConfig(StrictConfigModel):
    model_id: str
    revision: str | None = None
    dtype: ModelDType | None = None
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
    subset: str | None = None
    split: str = "train"
    text_column: str = "text"
    label_column: str | None = "label"
    prediction_column: str = "prediction"
    max_samples: int | None = Field(
        default=None,
        gt=0,
        validation_alias=AliasChoices("max_samples", "sample_limit"),
    )
    revision: str | None = None

    @model_validator(mode="after")
    def validate_dataset_source(self) -> "DatasetConfig":
        if self.dataset_id is None and self.local_path is None:
            msg = "either dataset_id or local_path must be provided"
            raise ValueError(msg)
        if self.dataset_id is not None and self.local_path is not None:
            msg = "dataset_id and local_path cannot both be provided"
            raise ValueError(msg)
        return self

    @property
    def sample_limit(self) -> int | None:
        return self.max_samples

    @field_validator(
        "dataset_id",
        "subset",
        "split",
        "text_column",
        "prediction_column",
        "revision",
    )
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

    @field_validator("local_path", mode="before")
    @classmethod
    def validate_local_path(cls, value: object) -> object:
        if isinstance(value, str):
            ensure_non_empty(value)
        return value


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
    averaging: list[AverageMethod] = Field(
        default_factory=default_averaging,
        validation_alias=AliasChoices("averaging", "average"),
    )
    save_predictions: bool = True
    save_errors: bool = True

    @property
    def average(self) -> AverageMethod:
        return self.averaging[0]

    @field_validator("metrics")
    @classmethod
    def validate_metrics(cls, value: list[str]) -> list[str]:
        if not value:
            msg = "at least one metric must be configured"
            raise ValueError(msg)
        return [ensure_non_empty(metric) for metric in value]

    @field_validator("averaging", mode="before")
    @classmethod
    def normalize_averaging(cls, value: object) -> object:
        if isinstance(value, str):
            return [value]
        return value


class RemoteConfig(StrictConfigModel):
    provider: Literal["modal"] = "modal"
    gpu: str | None = None
    cpu: int = Field(default=2, gt=0)
    memory_mb: int = Field(default=4096, gt=0)
    timeout_seconds: int = Field(default=900, gt=0)

    @field_validator("gpu")
    @classmethod
    def validate_gpu(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return ensure_non_empty(value)


class RemoteStorageConfig(StrictConfigModel):
    volume_name: str
    remote_root: Path = Path("/artifacts")
    hf_cache: Path = Path("/cache/huggingface")

    @field_validator("volume_name")
    @classmethod
    def validate_volume_name(cls, value: str) -> str:
        return ensure_non_empty(value)


class ModalConfig(StrictConfigModel):
    remote: RemoteConfig = Field(default_factory=RemoteConfig)
    storage: RemoteStorageConfig
