from pathlib import Path

import pytest

from nlp_lab.experiments.text_classification import (
    compute_classification_metrics,
    compute_confusion_matrix,
    load_text_classification_samples,
    normalize_label,
)


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
