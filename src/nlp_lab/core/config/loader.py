import json
from pathlib import Path
from typing import TypeVar

import yaml

from nlp_lab.core.config.common import PathLike, RawConfig, StrictConfigModel
from nlp_lab.core.config.experiment import ExperimentConfig

ConfigModelT = TypeVar("ConfigModelT", bound=StrictConfigModel)


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
