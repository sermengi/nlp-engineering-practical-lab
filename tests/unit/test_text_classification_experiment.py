from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from nlp_lab.core.config import load_layered_experiment_config
from nlp_lab.core.run_context import RunContext
from nlp_lab.experiments import text_classification
from nlp_lab.experiments.text_classification import (
    compute_classification_metrics,
    compute_confusion_matrix,
    hf_text_classification_experiment,
    load_text_classification_samples,
    normalize_label,
)
from nlp_lab.models import LoadedSequenceClassifier


@pytest.mark.unit
def test_load_text_classification_samples_reads_local_csv(tmp_path: Path) -> None:
    dataset_path = tmp_path / "samples.csv"
    dataset_path.write_text(
        "sample_id,text,label\n"
        'sample-001,"A useful compact baseline.",POSITIVE\n'
        'sample-002,"A weak and boring result.",NEGATIVE\n',
        encoding="utf-8",
    )

    samples = load_text_classification_samples(
        dataset_path,
        text_column="text",
        label_column="label",
        max_samples=1,
    )

    assert samples == [
        {
            "sample_id": "sample-001",
            "text": "A useful compact baseline.",
            "label": "POSITIVE",
        }
    ]


@pytest.mark.unit
def test_compute_classification_metrics_uses_explicit_metric_names() -> None:
    metrics = compute_classification_metrics(
        ["POSITIVE", "NEGATIVE", "NEGATIVE", "POSITIVE"],
        ["LABEL_1", "LABEL_0", "LABEL_1", "LABEL_1"],
    )

    assert metrics["accuracy"] == 0.75
    assert metrics["macro_precision"] == pytest.approx(0.8333333333)
    assert metrics["macro_recall"] == pytest.approx(0.75)
    assert metrics["macro_f1"] == pytest.approx(0.7333333333)
    assert metrics["weighted_f1"] == pytest.approx(0.7333333333)
    assert metrics["class_support_negative"] == 2.0
    assert metrics["class_support_positive"] == 2.0
    assert "f1" not in metrics


@pytest.mark.unit
def test_compute_confusion_matrix_counts_normalized_labels() -> None:
    matrix = compute_confusion_matrix(
        ["POSITIVE", "NEGATIVE", "NEGATIVE", "POSITIVE"],
        ["LABEL_1", "LABEL_0", "LABEL_1", "LABEL_1"],
    )

    assert matrix == {
        "negative": {"negative": 1, "positive": 1},
        "positive": {"negative": 0, "positive": 2},
    }


@pytest.mark.unit
def test_normalize_label_maps_common_binary_label_aliases() -> None:
    assert normalize_label("LABEL_0") == "negative"
    assert normalize_label("1") == "positive"
    assert normalize_label("neutral") == "neutral"


class FakeTensor:
    def __init__(self, size: int) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def to(self, device: str) -> "FakeTensor":
        return self


class FakeVector:
    def __init__(self, values: Sequence[float | int]) -> None:
        self.values = list(values)

    def __getitem__(self, index: int) -> "FakeScalar":
        return FakeScalar(self.values[index])


class FakeScalar:
    def __init__(self, value: float | int) -> None:
        self.value = value

    def item(self) -> float | int:
        return self.value


class FakeLogits:
    def __init__(self, label_ids: list[int]) -> None:
        self.label_ids = label_ids


class FakeTokenizer:
    def __call__(self, texts: list[str], **kwargs: Any) -> dict[str, FakeTensor]:
        return {"input_ids": FakeTensor(len(texts))}


class FakeModel:
    config = SimpleNamespace(id2label={0: "NEGATIVE", 1: "POSITIVE"})

    def __call__(self, **encoded: FakeTensor) -> SimpleNamespace:
        batch_size = len(encoded["input_ids"])
        return SimpleNamespace(logits=FakeLogits([index % 2 for index in range(batch_size)]))


class FakeInferenceMode:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *args: object) -> None:
        return None


class FakeTorch:
    def inference_mode(self) -> FakeInferenceMode:
        return FakeInferenceMode()

    def softmax(self, logits: FakeLogits, dim: int) -> FakeLogits:
        return logits

    def max(self, probabilities: FakeLogits, dim: int) -> tuple[FakeVector, FakeVector]:
        return (
            FakeVector([0.8 for _ in probabilities.label_ids]),
            FakeVector(probabilities.label_ids),
        )


@pytest.mark.unit
def test_hf_text_classification_experiment_records_runtime_without_real_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_path = tmp_path / "samples.csv"
    dataset_path.write_text(
        "sample_id,text,label\n"
        'sample-001,"A useful compact baseline.",NEGATIVE\n'
        'sample-002,"A weak and boring result.",POSITIVE\n'
        'sample-003,"A great result.",NEGATIVE\n',
        encoding="utf-8",
    )
    experiment_config_path = tmp_path / "experiment.yaml"
    experiment_config_path.write_text(
        f"""
experiment:
  name: fake-classification
  task: text-classification
model:
  model_id: fake-model
dataset:
  local_path: {dataset_path}
  split: test
  text_column: text
  label_column: label
  max_samples: 3
preprocessing:
  max_length: 32
  truncation: true
  padding: dynamic
inference:
  batch_size: 2
  threshold: 0.5
  device: cpu
evaluation:
  metrics:
    - accuracy
  save_predictions: true
  save_errors: true
""",
        encoding="utf-8",
    )
    config = load_layered_experiment_config(
        "configs/common/default.yaml",
        experiment_config_path,
        overrides={"output_root": tmp_path, "batch_size": 2, "max_samples": 3},
    )
    context = RunContext.create(config, run_id="run-001")
    context.output_dir.mkdir(parents=True)

    monkeypatch.setattr(text_classification, "load_torch", lambda: FakeTorch())
    monkeypatch.setattr(
        text_classification,
        "load_sequence_classifier",
        lambda config: LoadedSequenceClassifier(
            tokenizer=FakeTokenizer(),
            model=FakeModel(),
            device="cpu",
            dtype=None,
        ),
    )

    result = hf_text_classification_experiment(context)

    assert result.runtime.sample_count == 3
    assert result.runtime.batch_count == 2
    assert result.runtime.data_loading_seconds is not None
    assert result.runtime.model_load_seconds is not None
    assert result.runtime.preprocessing_seconds is not None
    assert result.runtime.inference_seconds is not None
    assert result.runtime.evaluation_seconds is not None
    assert result.runtime.average_batch_latency_seconds is not None
    assert result.runtime.samples_per_second == pytest.approx(
        result.runtime.sample_count / result.runtime.inference_seconds
    )
    assert len(result.predictions) == 3
