import csv
from collections.abc import Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

from nlp_lab.core.artifacts import write_json
from nlp_lab.core.experiment_result import ExperimentArtifact, ExperimentResult
from nlp_lab.core.run_artifacts import ErrorRecord, PredictionRecord, RuntimeMeasurements
from nlp_lab.core.run_context import RunContext
from nlp_lab.models import LoadedSequenceClassifier, load_sequence_classifier


def hf_text_classification_experiment(context: RunContext) -> ExperimentResult:
    torch = load_torch()

    config = context.config
    samples = load_text_classification_samples(
        config.dataset.local_path,
        text_column=config.dataset.text_column,
        label_column=config.dataset.label_column,
        max_samples=config.dataset.max_samples,
    )
    true_labels = [sample["label"] for sample in samples]

    model_started = perf_counter()
    loaded = load_sequence_classifier(config)
    model_load_seconds = perf_counter() - model_started

    inference_started = perf_counter()
    predictions = run_inference(
        loaded,
        samples,
        batch_size=config.inference.batch_size,
        max_length=config.preprocessing.max_length,
        truncation=config.preprocessing.truncation,
        padding=config.preprocessing.padding,
        torch=torch,
    )
    predicted_labels = [str(prediction.predicted_label) for prediction in predictions]
    inference_seconds = perf_counter() - inference_started

    evaluation_started = perf_counter()
    metrics = compute_classification_metrics(true_labels, predicted_labels)
    confusion_matrix = compute_confusion_matrix(true_labels, predicted_labels)
    errors = [
        ErrorRecord(
            sample_id=prediction.sample_id,
            true_label=prediction.true_label,
            predicted_label=prediction.predicted_label,
            confidence=prediction.confidence,
            error_type="misclassification",
        )
        for prediction in predictions
        if prediction.is_correct is False
    ]
    evaluation_seconds = perf_counter() - evaluation_started
    total_duration_seconds = model_load_seconds + inference_seconds + evaluation_seconds
    artifacts = write_metadata_artifacts(
        context,
        loaded,
        sample_count=len(samples),
        confusion_matrix=confusion_matrix,
    )

    return ExperimentResult(
        metrics=metrics,
        runtime=RuntimeMeasurements(
            total_duration_seconds=total_duration_seconds,
            model_load_seconds=model_load_seconds,
            inference_seconds=inference_seconds,
            evaluation_seconds=evaluation_seconds,
            samples_per_second=len(samples) / inference_seconds if inference_seconds > 0 else None,
            batch_size=config.inference.batch_size,
        ),
        predictions=predictions if config.evaluation.save_predictions else [],
        errors=errors if config.evaluation.save_errors else [],
        artifacts=artifacts,
        notes=[
            f"Loaded {config.model.model_id} on {loaded.device} for local smoke testing.",
            f"Evaluated {len(samples)} local samples from {config.dataset.local_path}.",
        ],
    )


def load_torch() -> Any:
    from nlp_lab.models.loading import import_optional_dependency

    return import_optional_dependency("torch")


def load_text_classification_samples(
    path: Path | None,
    *,
    text_column: str,
    label_column: str | None,
    max_samples: int | None,
) -> list[dict[str, str]]:
    if path is None:
        msg = "local smoke experiment requires dataset.local_path"
        raise ValueError(msg)
    if label_column is None:
        msg = "local smoke experiment requires dataset.label_column"
        raise ValueError(msg)

    samples: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            msg = f"dataset must contain a header row: {path}"
            raise ValueError(msg)
        require_dataset_columns(
            reader.fieldnames,
            required_columns=["sample_id", text_column, label_column],
            path=path,
        )
        for row in reader:
            sample_id = require_column(row, "sample_id", path)
            text = require_column(row, text_column, path)
            label = require_column(row, label_column, path)
            samples.append(
                {
                    "sample_id": sample_id,
                    "text": text,
                    "label": label,
                }
            )
            if max_samples is not None and len(samples) >= max_samples:
                break

    if not samples:
        msg = f"dataset contains no samples: {path}"
        raise ValueError(msg)
    return samples


def require_dataset_columns(
    fieldnames: Sequence[str],
    *,
    required_columns: Sequence[str],
    path: Path,
) -> None:
    missing_columns = sorted(set(required_columns) - set(fieldnames))
    if missing_columns:
        msg = f"dataset {path} is missing required columns: {', '.join(missing_columns)}"
        raise ValueError(msg)


def require_column(row: dict[str, str], column: str, path: Path) -> str:
    value = row.get(column)
    if value is None or not value.strip():
        msg = f"missing non-empty column {column!r} in {path}"
        raise ValueError(msg)
    return value.strip()


def run_inference(
    loaded: LoadedSequenceClassifier,
    samples: Sequence[dict[str, str]],
    *,
    batch_size: int,
    max_length: int,
    truncation: bool,
    padding: str,
    torch: Any,
) -> list[PredictionRecord]:
    predictions: list[PredictionRecord] = []
    for batch_samples in iter_batches(samples, batch_size):
        encoded = tokenize_batch(
            loaded.tokenizer,
            batch_samples,
            max_length=max_length,
            truncation=truncation,
            padding=padding,
        )
        encoded = {key: value.to(loaded.device) for key, value in encoded.items()}
        with torch.inference_mode():
            outputs = loaded.model(**encoded)
            probabilities = torch.softmax(outputs.logits, dim=-1)
        batch_confidences, batch_label_ids = torch.max(probabilities, dim=-1)

        for offset, sample in enumerate(batch_samples):
            label_id = int(batch_label_ids[offset].item())
            predicted_label = label_for_id(loaded.model.config.id2label, label_id)
            true_label = sample["label"]
            predictions.append(
                PredictionRecord(
                    sample_id=sample["sample_id"],
                    text=sample["text"],
                    true_label=true_label,
                    predicted_label=predicted_label,
                    confidence=float(batch_confidences[offset].item()),
                    is_correct=normalize_label(predicted_label) == normalize_label(true_label),
                )
            )
    return predictions


def iter_batches(
    samples: Sequence[dict[str, str]],
    batch_size: int,
) -> list[Sequence[dict[str, str]]]:
    return [samples[index : index + batch_size] for index in range(0, len(samples), batch_size)]


def tokenize_batch(
    tokenizer: Any,
    samples: Sequence[dict[str, str]],
    *,
    max_length: int,
    truncation: bool,
    padding: str,
) -> Any:
    return tokenizer(
        [sample["text"] for sample in samples],
        padding=tokenizer_padding_value(padding),
        truncation=truncation,
        max_length=max_length,
        return_tensors="pt",
    )


def label_for_id(id2label: object, label_id: int) -> str:
    if isinstance(id2label, dict):
        return str(id2label.get(label_id, id2label.get(str(label_id), str(label_id))))
    return str(label_id)


def compute_classification_metrics(
    true_labels: Sequence[str],
    predicted_labels: Sequence[str],
) -> dict[str, float]:
    if len(true_labels) != len(predicted_labels):
        msg = "true and predicted label counts must match"
        raise ValueError(msg)
    if not true_labels:
        msg = "at least one labeled sample is required"
        raise ValueError(msg)

    normalized_true = [normalize_label(label) for label in true_labels]
    normalized_predicted = [normalize_label(label) for label in predicted_labels]
    labels = sorted(set(normalized_true) | set(normalized_predicted))
    accuracy = sum(
        expected == actual
        for expected, actual in zip(normalized_true, normalized_predicted, strict=True)
    ) / len(normalized_true)

    precision_values: list[float] = []
    recall_values: list[float] = []
    f1_values: list[float] = []
    weighted_f1_sum = 0.0
    for label in labels:
        true_positive = sum(
            expected == label and actual == label
            for expected, actual in zip(normalized_true, normalized_predicted, strict=True)
        )
        false_positive = sum(
            expected != label and actual == label
            for expected, actual in zip(normalized_true, normalized_predicted, strict=True)
        )
        false_negative = sum(
            expected == label and actual != label
            for expected, actual in zip(normalized_true, normalized_predicted, strict=True)
        )
        precision = safe_divide(true_positive, true_positive + false_positive)
        recall = safe_divide(true_positive, true_positive + false_negative)
        support = sum(expected == label for expected in normalized_true)
        f1 = safe_divide(2 * precision * recall, precision + recall)
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)
        weighted_f1_sum += f1 * support

    metrics = {
        "accuracy": accuracy,
        "macro_precision": sum(precision_values) / len(precision_values),
        "macro_recall": sum(recall_values) / len(recall_values),
        "macro_f1": sum(f1_values) / len(f1_values),
        "weighted_f1": weighted_f1_sum / len(normalized_true),
    }
    for label in labels:
        metrics[f"class_support_{label}"] = float(
            sum(expected == label for expected in normalized_true)
        )
    return metrics


def compute_confusion_matrix(
    true_labels: Sequence[str],
    predicted_labels: Sequence[str],
) -> dict[str, dict[str, int]]:
    normalized_true = [normalize_label(label) for label in true_labels]
    normalized_predicted = [normalize_label(label) for label in predicted_labels]
    labels = sorted(set(normalized_true) | set(normalized_predicted))
    matrix = {label: {predicted: 0 for predicted in labels} for label in labels}
    for expected, actual in zip(normalized_true, normalized_predicted, strict=True):
        matrix[expected][actual] += 1
    return matrix


def normalize_label(label: str | int) -> str:
    normalized = str(label).strip().lower()
    label_aliases = {
        "0": "negative",
        "label_0": "negative",
        "negative": "negative",
        "neg": "negative",
        "1": "positive",
        "label_1": "positive",
        "positive": "positive",
        "pos": "positive",
    }
    return label_aliases.get(normalized, normalized)


def safe_divide(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def tokenizer_padding_value(padding: str) -> bool | str:
    if padding == "dynamic":
        return True
    return padding


def write_metadata_artifacts(
    context: RunContext,
    loaded: LoadedSequenceClassifier,
    *,
    sample_count: int,
    confusion_matrix: dict[str, dict[str, int]],
) -> list[ExperimentArtifact]:
    config = context.config
    model_metadata_path = context.output_dir / "model_metadata.json"
    dataset_metadata_path = context.output_dir / "dataset_metadata.json"
    confusion_matrix_path = context.output_dir / "confusion_matrix.json"
    write_json(
        model_metadata_path,
        {
            "model_id": config.model.model_id,
            "revision": config.model.revision,
            "trust_remote_code": config.model.trust_remote_code,
            "configured_dtype": config.model.dtype,
            "loaded_dtype": loaded.dtype,
            "device": loaded.device,
            "id2label": {
                str(key): str(value)
                for key, value in getattr(loaded.model.config, "id2label", {}).items()
            },
        },
    )
    write_json(
        dataset_metadata_path,
        {
            "local_path": str(config.dataset.local_path),
            "split": config.dataset.split,
            "text_column": config.dataset.text_column,
            "label_column": config.dataset.label_column,
            "sample_id_column": "sample_id",
            "sample_count": sample_count,
            "max_samples": config.dataset.max_samples,
        },
    )
    write_json(confusion_matrix_path, confusion_matrix)
    return [
        ExperimentArtifact(name="model_metadata", path=model_metadata_path),
        ExperimentArtifact(name="dataset_metadata", path=dataset_metadata_path),
        ExperimentArtifact(name="confusion_matrix", path=confusion_matrix_path),
    ]
