from typing import Literal

from pydantic import Field, field_validator, model_validator

from nlp_lab.core.config.common import (
    CachePathsConfig,
    LoggingConfig,
    ProjectConfig,
    RunNamingConfig,
    StrictConfigModel,
    ensure_non_empty,
)
from nlp_lab.core.config.runtime import (
    DatasetConfig,
    EvaluationConfig,
    InferenceConfig,
    ModelConfig,
    PreprocessingConfig,
    RuntimeConfig,
)

TaskName = Literal["text-classification"]


class ExperimentMetadataConfig(StrictConfigModel):
    name: str = "default"
    task: TaskName = "text-classification"
    description: str | None = None

    @field_validator("name", "task")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        return ensure_non_empty(value)

    @field_validator("description")
    @classmethod
    def validate_description(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return ensure_non_empty(value)


class CommonConfig(StrictConfigModel):
    project: ProjectConfig
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    cache: CachePathsConfig = Field(default_factory=CachePathsConfig)
    run_naming: RunNamingConfig = Field(default_factory=RunNamingConfig)


class ExperimentConfig(StrictConfigModel):
    experiment: ExperimentMetadataConfig = Field(default_factory=ExperimentMetadataConfig)
    project: ProjectConfig | None = None
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    cache: CachePathsConfig = Field(default_factory=CachePathsConfig)
    run_naming: RunNamingConfig = Field(default_factory=RunNamingConfig)
    model: ModelConfig
    dataset: DatasetConfig
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    @property
    def experiment_name(self) -> str:
        return self.experiment.name

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_experiment_name(cls, data: object) -> object:
        if not isinstance(data, dict) or "experiment_name" not in data:
            return data
        migrated = dict(data)
        experiment_name = migrated.pop("experiment_name")
        migrated.setdefault("experiment", {"name": experiment_name})
        return migrated

    @model_validator(mode="after")
    def validate_task_contract(self) -> "ExperimentConfig":
        if self.experiment.task == "text-classification":
            if self.dataset.text_column is None or self.dataset.label_column is None:
                msg = "text-classification requires text_column and label_column"
                raise ValueError(msg)
        return self
