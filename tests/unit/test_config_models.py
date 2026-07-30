from datetime import datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from nlp_lab.core.config import (
    ConfigOverrides,
    DatasetConfig,
    ExperimentConfig,
    InferenceConfig,
    ModalConfig,
    ModelConfig,
    PreprocessingConfig,
    ProjectConfig,
    RuntimeConfig,
    compute_config_hash,
    generate_run_id,
    load_common_config,
    load_experiment_config,
    load_layered_experiment_config,
    load_modal_config,
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
    assert config.experiment.name == "sentiment-baseline"
    assert config.experiment_name == "sentiment-baseline"
    assert config.runtime.seed == 42
    assert config.dataset.text_column == "text"
    assert config.dataset.max_samples == 100
    assert config.dataset.sample_limit == 100
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
def test_inference_batch_size_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        InferenceConfig(batch_size=0)


@pytest.mark.unit
def test_preprocessing_max_length_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        PreprocessingConfig(max_length=0)


@pytest.mark.unit
def test_dataset_max_samples_must_be_positive_when_set() -> None:
    with pytest.raises(ValidationError):
        DatasetConfig(dataset_id="imdb", max_samples=0)


@pytest.mark.unit
def test_dataset_config_rejects_dataset_id_and_local_path_together() -> None:
    with pytest.raises(ValidationError, match="cannot both be provided"):
        DatasetConfig(dataset_id="imdb", local_path="data/raw/sample.csv")


@pytest.mark.unit
def test_runtime_output_root_must_not_be_empty() -> None:
    with pytest.raises(ValidationError, match="must not be empty"):
        RuntimeConfig(output_root=" ")


@pytest.mark.unit
def test_model_dtype_must_be_supported() -> None:
    with pytest.raises(ValidationError):
        ModelConfig(model_id="bert-base-uncased", dtype="int8")


@pytest.mark.unit
def test_experiment_task_must_be_supported() -> None:
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(
            {
                "experiment": {"name": "ner", "task": "token-classification"},
                "model": {"model_id": "bert-base-uncased"},
                "dataset": {"dataset_id": "conll2003"},
            }
        )


@pytest.mark.unit
def test_text_classification_requires_label_column() -> None:
    with pytest.raises(ValidationError, match="requires text_column and label_column"):
        ExperimentConfig.model_validate(
            {
                "experiment": {"name": "classification", "task": "text-classification"},
                "model": {"model_id": "bert-base-uncased"},
                "dataset": {
                    "dataset_id": "imdb",
                    "text_column": "text",
                    "label_column": None,
                },
            }
        )


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
  average: weighted
""".strip(),
        encoding="utf-8",
    )

    config = load_experiment_config(config_path)

    assert config.experiment_name == "local-smoke"
    assert config.dataset.local_path == Path("data/raw/sample.csv")
    assert config.inference.batch_size == 4
    assert config.evaluation.metrics == ["accuracy", "f1"]
    assert config.evaluation.averaging == ["weighted"]


@pytest.mark.unit
def test_load_common_default_config() -> None:
    config = load_common_config("configs/common/default.yaml")

    assert config.project.name == "nlp-engineering-practical-lab"
    assert config.runtime.output_root == Path("outputs/experiments")
    assert config.logging.level == "INFO"
    assert config.cache.huggingface == Path(".cache/huggingface")
    assert config.run_naming.strategy == "timestamp"


@pytest.mark.unit
def test_load_classification_baseline_experiment_config() -> None:
    config = load_experiment_config("configs/experiments/classification_baseline.yaml")

    assert config.experiment.name == "classification-baseline"
    assert config.experiment.task == "text-classification"
    assert config.model.model_id == "placeholder-model"
    assert config.dataset.dataset_id == "placeholder-dataset"
    assert config.dataset.max_samples is None
    assert config.preprocessing.padding == "dynamic"
    assert config.evaluation.averaging == ["macro", "weighted"]


@pytest.mark.unit
def test_load_modal_default_config() -> None:
    config = load_modal_config("configs/modal/default.yaml")

    assert isinstance(config, ModalConfig)
    assert config.remote.provider == "modal"
    assert config.remote.cpu == 2
    assert config.remote.memory_mb == 4096
    assert config.storage.volume_name == "nlp-lab-artifacts"
    assert config.storage.remote_root == Path("/artifacts")


@pytest.mark.unit
def test_load_layered_experiment_config_merges_common_then_experiment_then_overrides() -> None:
    config = load_layered_experiment_config(
        "configs/common/default.yaml",
        "configs/experiments/classification_baseline.yaml",
        overrides={
            "batch_size": 32,
            "max_samples": 25,
            "model_id": "override-model",
            "dataset_split": "validation",
            "threshold": 0.7,
            "output_root": "outputs/overrides",
            "seed": 7,
        },
    )

    assert config.project == ProjectConfig(name="nlp-engineering-practical-lab")
    assert config.logging.level == "INFO"
    assert config.runtime.seed == 7
    assert config.runtime.output_root == Path("outputs/overrides")
    assert config.model.model_id == "override-model"
    assert config.dataset.split == "validation"
    assert config.dataset.max_samples == 25
    assert config.inference.batch_size == 32
    assert config.inference.threshold == 0.7
    assert config.preprocessing.max_length == 256


@pytest.mark.unit
def test_load_layered_experiment_config_rejects_unsupported_override_fields() -> None:
    with pytest.raises(ValidationError):
        load_layered_experiment_config(
            "configs/common/default.yaml",
            "configs/experiments/classification_baseline.yaml",
            overrides={"preprocessing": {"max_length": 128}},
        )


@pytest.mark.unit
def test_config_overrides_require_supported_fields() -> None:
    with pytest.raises(ValidationError):
        ConfigOverrides()


@pytest.mark.unit
def test_generate_run_id_uses_timestamp_experiment_name_and_config_hash() -> None:
    config = load_layered_experiment_config(
        "configs/common/default.yaml",
        "configs/experiments/classification_baseline.yaml",
    )

    run_id = generate_run_id(config, timestamp=datetime(2026, 7, 30, 12, 55, 30))

    assert run_id == f"20260730-125530_classification-baseline_{compute_config_hash(config)}"


@pytest.mark.unit
def test_config_hash_excludes_timestamp() -> None:
    config = load_layered_experiment_config(
        "configs/common/default.yaml",
        "configs/experiments/classification_baseline.yaml",
    )

    first_run_id = generate_run_id(config, timestamp=datetime(2026, 7, 30, 12, 55, 30))
    second_run_id = generate_run_id(config, timestamp=datetime(2026, 7, 30, 13, 2, 15))

    assert first_run_id.rsplit("_", maxsplit=1)[1] == second_run_id.rsplit("_", maxsplit=1)[1]


@pytest.mark.unit
def test_config_hash_changes_when_hash_payload_changes() -> None:
    baseline = load_layered_experiment_config(
        "configs/common/default.yaml",
        "configs/experiments/classification_baseline.yaml",
    )
    changed_batch_size = load_layered_experiment_config(
        "configs/common/default.yaml",
        "configs/experiments/classification_baseline.yaml",
        overrides={"batch_size": 32},
    )
    changed_output_root = load_layered_experiment_config(
        "configs/common/default.yaml",
        "configs/experiments/classification_baseline.yaml",
        overrides={"output_root": "outputs/other"},
    )

    assert compute_config_hash(baseline) != compute_config_hash(changed_batch_size)
    assert compute_config_hash(baseline) == compute_config_hash(changed_output_root)


@pytest.mark.unit
def test_config_hash_changes_when_seed_changes() -> None:
    baseline = load_layered_experiment_config(
        "configs/common/default.yaml",
        "configs/experiments/classification_baseline.yaml",
    )
    changed_seed = load_layered_experiment_config(
        "configs/common/default.yaml",
        "configs/experiments/classification_baseline.yaml",
        overrides={"seed": 123},
    )

    assert compute_config_hash(baseline) != compute_config_hash(changed_seed)
