from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from nlp_lab.core.experiment_result import ExperimentArtifact, ExperimentResult
from nlp_lab.core.run_artifacts import (
    PredictionRecord,
    RuntimeMeasurements,
    build_run_artifact_paths,
    write_experiment_result,
)


@pytest.mark.unit
def test_experiment_result_holds_outputs_without_filesystem_access() -> None:
    result = ExperimentResult(
        metrics={"accuracy": 0.91, "macro_f1": 0.8},
        runtime=RuntimeMeasurements(total_duration_seconds=42.1, batch_size=16),
        predictions=[
            PredictionRecord(
                sample_id="test-001",
                text="Hydraulic pressure warning detected.",
                true_label=1,
                predicted_label=1,
                confidence=0.94,
                is_correct=True,
            )
        ],
        artifacts=[ExperimentArtifact(name="confusion_matrix", path=Path("plots/confusion.png"))],
        notes=["Model missed part of the minority class."],
    )

    assert result.metrics["macro_f1"] == 0.8
    assert result.runtime.batch_size == 16
    assert result.predictions[0].sample_id == "test-001"
    assert result.artifacts[0].name == "confusion_matrix"
    assert "filesystem" not in ExperimentResult.model_fields


@pytest.mark.unit
def test_experiment_result_rejects_ambiguous_metric_names() -> None:
    with pytest.raises(ValidationError, match="metric names must be explicit"):
        ExperimentResult(metrics={"f1": 0.8}, runtime=RuntimeMeasurements(batch_size=16))


@pytest.mark.unit
def test_write_experiment_result_maps_result_to_artifact_files(tmp_path: Path) -> None:
    paths = build_run_artifact_paths(tmp_path)
    result = ExperimentResult(
        metrics={"accuracy": 0.91, "macro_f1": 0.8},
        runtime=RuntimeMeasurements(total_duration_seconds=42.1, batch_size=16),
        predictions=[
            PredictionRecord(
                sample_id="test-001",
                true_label=1,
                predicted_label=1,
                confidence=0.94,
                is_correct=True,
            )
        ],
        notes=["Short observation."],
    )

    write_experiment_result(paths, result)

    assert yaml.safe_load(paths.metrics.read_text(encoding="utf-8")) == {
        "accuracy": 0.91,
        "macro_f1": 0.8,
    }
    assert yaml.safe_load(paths.runtime.read_text(encoding="utf-8"))["batch_size"] == 16
    assert len(paths.predictions.read_text(encoding="utf-8").splitlines()) == 1
    assert paths.summary.read_text(encoding="utf-8").startswith("# Experiment Summary")
