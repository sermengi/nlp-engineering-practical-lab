from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

from nlp_lab.core.artifacts import (
    ArtifactRunExistsError,
    ArtifactSerializationError,
    LocalFilesystemArtifactWriter,
    write_json,
)
from nlp_lab.core.config import load_layered_experiment_config
from nlp_lab.core.experiment_result import ExperimentResult
from nlp_lab.core.run_artifacts import PredictionRecord, RuntimeMeasurements


@pytest.mark.unit
def test_local_filesystem_writer_initializes_standard_run_directory(tmp_path: Path) -> None:
    config = load_layered_experiment_config(
        "configs/common/default.yaml",
        "configs/experiments/classification_baseline.yaml",
        overrides={"output_root": tmp_path},
    )
    writer = LocalFilesystemArtifactWriter()

    paths, metadata = writer.initialize_run(
        config,
        run_id="run-001",
        started_at=datetime(2026, 7, 30, 12, 55, 30, tzinfo=UTC),
    )

    assert paths.run_dir == tmp_path / "run-001"
    assert paths.resolved_config.exists()
    assert paths.run_metadata.exists()
    assert paths.environment.exists()
    assert metadata.status == "RUNNING"


@pytest.mark.unit
def test_local_filesystem_writer_prevents_overwriting_existing_run(tmp_path: Path) -> None:
    config = load_layered_experiment_config(
        "configs/common/default.yaml",
        "configs/experiments/classification_baseline.yaml",
        overrides={"output_root": tmp_path},
    )
    writer = LocalFilesystemArtifactWriter()

    writer.initialize_run(config, run_id="run-001")

    with pytest.raises(ArtifactRunExistsError, match="already exists"):
        writer.initialize_run(config, run_id="run-001")


@pytest.mark.unit
def test_local_filesystem_writer_writes_experiment_result(tmp_path: Path) -> None:
    config = load_layered_experiment_config(
        "configs/common/default.yaml",
        "configs/experiments/classification_baseline.yaml",
        overrides={"output_root": tmp_path},
    )
    writer = LocalFilesystemArtifactWriter()
    paths, _ = writer.initialize_run(config, run_id="run-001")
    result = ExperimentResult(
        metrics={"accuracy": 0.91, "macro_f1": 0.8},
        runtime=RuntimeMeasurements(batch_size=16),
        predictions=[
            PredictionRecord(
                sample_id="test-001",
                predicted_label=1,
                confidence=0.94,
            )
        ],
        notes=["Short observation."],
    )

    writer.write_result(paths, result)

    assert yaml.safe_load(paths.metrics.read_text(encoding="utf-8"))["macro_f1"] == 0.8
    assert yaml.safe_load(paths.runtime.read_text(encoding="utf-8"))["batch_size"] == 16
    assert len(paths.predictions.read_text(encoding="utf-8").splitlines()) == 1
    assert paths.summary.exists()


@pytest.mark.unit
def test_serialization_errors_report_target_context(tmp_path: Path) -> None:
    with pytest.raises(ArtifactSerializationError, match="failed to serialize JSON artifact"):
        write_json(tmp_path / "bad.json", {"not_serializable": object()})
