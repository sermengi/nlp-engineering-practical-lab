from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

from nlp_lab.core.artifacts import LocalFilesystemArtifactWriter
from nlp_lab.core.config import compute_config_hash, load_layered_experiment_config
from nlp_lab.core.experiment_result import ExperimentResult
from nlp_lab.core.run_artifacts import PredictionRecord, RuntimeMeasurements


@pytest.mark.unit
def test_run_management_success_criteria_are_met(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_secret_value")
    local_root = tmp_path / "outputs" / "experiments"
    modal_root = tmp_path / "artifacts" / "experiments"
    writer = LocalFilesystemArtifactWriter()

    config = load_layered_experiment_config(
        "configs/common/default.yaml",
        "configs/experiments/classification_baseline.yaml",
        overrides={"output_root": local_root, "batch_size": 16},
    )
    same_config = load_layered_experiment_config(
        "configs/common/default.yaml",
        "configs/experiments/classification_baseline.yaml",
        overrides={"output_root": local_root, "batch_size": 16},
    )

    assert compute_config_hash(config) == compute_config_hash(same_config)

    first_paths, first_metadata = writer.initialize_run(
        config,
        run_id="run-001",
        started_at=datetime(2026, 7, 30, 12, 55, 30, tzinfo=UTC),
    )
    second_paths, _ = writer.initialize_run(
        config,
        run_id="run-002",
        started_at=datetime(2026, 7, 30, 12, 56, 30, tzinfo=UTC),
    )

    assert first_paths.run_dir != second_paths.run_dir
    assert first_paths.resolved_config.exists()
    assert first_paths.run_metadata.exists()
    assert first_paths.environment.exists()

    result = ExperimentResult(
        metrics={"accuracy": 0.91, "macro_f1": 0.8},
        runtime=RuntimeMeasurements(total_duration_seconds=1.2, batch_size=16),
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
    )
    writer.write_result(first_paths, result)
    writer.complete_run(
        first_paths,
        first_metadata,
        completed_at=datetime(2026, 7, 30, 12, 56, 12, tzinfo=UTC),
    )

    completed_run = yaml.safe_load(first_paths.run_metadata.read_text(encoding="utf-8"))
    assert completed_run["status"] == "COMPLETED"
    assert yaml.safe_load(first_paths.metrics.read_text(encoding="utf-8"))["macro_f1"] == 0.8
    assert len(first_paths.predictions.read_text(encoding="utf-8").splitlines()) == 1

    failed_paths, failed_metadata = writer.initialize_run(config, run_id="run-failed")
    try:
        raise RuntimeError("download failed with token hf_secret_value")
    except RuntimeError as exc:
        writer.fail_run(
            failed_paths,
            failed_metadata,
            exc,
            failed_at=datetime(2026, 7, 30, 12, 57, 12, tzinfo=UTC),
        )

    failed_run = yaml.safe_load(failed_paths.run_metadata.read_text(encoding="utf-8"))
    assert failed_paths.run_dir.exists()
    assert failed_paths.resolved_config.exists()
    assert failed_paths.environment.exists()
    assert failed_run["status"] == "FAILED"
    assert failed_run["exception_type"] == "RuntimeError"
    assert "hf_secret_value" not in failed_paths.run_metadata.read_text(encoding="utf-8")
    assert "[REDACTED]" in failed_run["error_message"]

    modal_config = load_layered_experiment_config(
        "configs/common/default.yaml",
        "configs/experiments/classification_baseline.yaml",
        overrides={"output_root": modal_root},
    )
    modal_paths, modal_metadata = writer.initialize_run(
        modal_config,
        run_id="modal-run-001",
        execution_mode="modal",
    )

    assert modal_paths.run_dir == modal_root / "modal-run-001"
    assert modal_metadata.execution_mode == "modal"

    artifact_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in [
            first_paths.resolved_config,
            first_paths.run_metadata,
            first_paths.environment,
            first_paths.metrics,
            first_paths.predictions,
            failed_paths.run_metadata,
            failed_paths.environment,
            modal_paths.environment,
        ]
    )
    assert "hf_secret_value" not in artifact_text
