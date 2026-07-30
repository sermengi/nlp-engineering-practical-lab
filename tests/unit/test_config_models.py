from pathlib import Path

import pytest
from pydantic import ValidationError

from nlp_lab.core.config import (
    DatasetConfig,
    ExperimentConfig,
    InferenceConfig,
    ModelConfig,
    ProjectConfig,
    load_experiment_config,
)


@pytest.mark.unit
def test_experiment_config_accepts_nested_dicts() -> None:
    config = ExperimentConfig.model_validate(
        {
            "experiment_name": "sentiment-baseline",
            "project": {"name": "nlp-lab", "config_version": "1"},
            "model": {"model_id": "distilbert-base-uncased"},
            "dataset": {"dataset_id": "imdb", "split": "test", "sample_limit": 100},
        }
    )

    assert config.project == ProjectConfig(name="nlp-lab", config_version="1")
    assert config.runtime.seed == 42
    assert config.dataset.text_column == "text"
    assert config.preprocessing.max_length == 512
    assert config.inference.device == "auto"
    assert config.evaluation.metrics == ["accuracy"]


@pytest.mark.unit
def test_config_models_reject_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        ModelConfig(model_id="bert-base-uncased", unexpected=True)  # type: ignore[call-arg]


@pytest.mark.unit
def test_dataset_config_requires_dataset_id_or_local_path() -> None:
    with pytest.raises(ValidationError, match="either dataset_id or local_path"):
        DatasetConfig()


@pytest.mark.unit
def test_inference_threshold_is_bounded() -> None:
    with pytest.raises(ValidationError):
        InferenceConfig(threshold=1.1)


@pytest.mark.unit
def test_load_experiment_config_from_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        """
experiment_name: local-smoke
project:
  name: nlp-lab
model:
  model_id: distilbert-base-uncased
dataset:
  local_path: data/raw/sample.csv
inference:
  batch_size: 4
evaluation:
  metrics:
    - accuracy
    - f1
""".strip(),
        encoding="utf-8",
    )

    config = load_experiment_config(config_path)

    assert config.experiment_name == "local-smoke"
    assert config.dataset.local_path == Path("data/raw/sample.csv")
    assert config.inference.batch_size == 4
    assert config.evaluation.metrics == ["accuracy", "f1"]
