from pydantic import Field, field_validator

from nlp_lab.core.config.common import ProjectConfig, StrictConfigModel, ensure_non_empty
from nlp_lab.core.config.runtime import (
    DatasetConfig,
    EvaluationConfig,
    InferenceConfig,
    ModelConfig,
    PreprocessingConfig,
    RuntimeConfig,
)


class ExperimentConfig(StrictConfigModel):
    experiment_name: str = Field(default="default")
    project: ProjectConfig
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    model: ModelConfig
    dataset: DatasetConfig
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    @field_validator("experiment_name")
    @classmethod
    def validate_experiment_name(cls, value: str) -> str:
        return ensure_non_empty(value)
