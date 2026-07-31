import json
from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import Field, model_validator

from nlp_lab.core.config.common import PathLike, RawConfig, StrictConfigModel
from nlp_lab.core.config.experiment import CommonConfig, ExperimentConfig
from nlp_lab.core.config.runtime import ModalConfig

ConfigModelT = TypeVar("ConfigModelT", bound=StrictConfigModel)


class ConfigOverrides(StrictConfigModel):
    batch_size: int | None = Field(default=None, gt=0)
    max_samples: int | None = Field(default=None, gt=0)
    model_id: str | None = None
    dataset_split: str | None = None
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    output_root: Path | None = None
    seed: int | None = Field(default=None, ge=0)
    cache_huggingface: Path | None = None
    cache_datasets: Path | None = None
    cache_models: Path | None = None

    @model_validator(mode="after")
    def require_at_least_one_override(self) -> "ConfigOverrides":
        if all(value is None for value in self.model_dump().values()):
            msg = "at least one supported override must be provided"
            raise ValueError(msg)
        return self


def load_config_dict(path: PathLike) -> RawConfig:
    config_path = Path(path)
    suffix = config_path.suffix.lower()

    with config_path.open(encoding="utf-8") as file:
        if suffix in {".yaml", ".yml"}:
            data = yaml.safe_load(file)
        elif suffix == ".json":
            data = json.load(file)
        else:
            msg = f"unsupported config file extension: {suffix or '<none>'}"
            raise ValueError(msg)

    if data is None:
        return {}
    if not isinstance(data, dict):
        msg = "config file must contain a mapping at the top level"
        raise ValueError(msg)
    return data


def load_config(path: PathLike, model_type: type[ConfigModelT]) -> ConfigModelT:
    return model_type.model_validate(load_config_dict(path))


def load_experiment_config(path: PathLike) -> ExperimentConfig:
    return load_config(path, ExperimentConfig)


def load_common_config(path: PathLike) -> CommonConfig:
    return load_config(path, CommonConfig)


def load_modal_config(path: PathLike) -> ModalConfig:
    return load_config(path, ModalConfig)


def merge_config_dicts(base: RawConfig, override: RawConfig) -> RawConfig:
    merged = dict(base)
    for key, value in override.items():
        base_value = merged.get(key)
        if isinstance(base_value, dict) and isinstance(value, dict):
            merged[key] = merge_config_dicts(base_value, value)
        else:
            merged[key] = value
    return merged


def overrides_to_config_dict(overrides: ConfigOverrides) -> RawConfig:
    data: RawConfig = {}
    if overrides.batch_size is not None:
        data = merge_config_dicts(data, {"inference": {"batch_size": overrides.batch_size}})
    if overrides.max_samples is not None:
        data = merge_config_dicts(data, {"dataset": {"max_samples": overrides.max_samples}})
    if overrides.model_id is not None:
        data = merge_config_dicts(data, {"model": {"model_id": overrides.model_id}})
    if overrides.dataset_split is not None:
        data = merge_config_dicts(data, {"dataset": {"split": overrides.dataset_split}})
    if overrides.threshold is not None:
        data = merge_config_dicts(data, {"inference": {"threshold": overrides.threshold}})
    if overrides.output_root is not None:
        data = merge_config_dicts(data, {"runtime": {"output_root": overrides.output_root}})
    if overrides.seed is not None:
        data = merge_config_dicts(data, {"runtime": {"seed": overrides.seed}})
    cache_overrides: RawConfig = {}
    if overrides.cache_huggingface is not None:
        cache_overrides["huggingface"] = overrides.cache_huggingface
    if overrides.cache_datasets is not None:
        cache_overrides["datasets"] = overrides.cache_datasets
    if overrides.cache_models is not None:
        cache_overrides["models"] = overrides.cache_models
    if cache_overrides:
        data = merge_config_dicts(data, {"cache": cache_overrides})
    return data


def load_layered_experiment_config(
    common_path: PathLike,
    experiment_path: PathLike,
    overrides: ConfigOverrides | RawConfig | None = None,
) -> ExperimentConfig:
    merged = merge_config_dicts(load_config_dict(common_path), load_config_dict(experiment_path))
    if overrides is not None:
        validated_overrides = (
            overrides
            if isinstance(overrides, ConfigOverrides)
            else ConfigOverrides.model_validate(overrides)
        )
        merged = merge_config_dicts(merged, overrides_to_config_dict(validated_overrides))
    return ExperimentConfig.model_validate(merged)
