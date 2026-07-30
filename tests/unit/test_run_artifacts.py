from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

from nlp_lab.core.config import ProjectConfig, load_layered_experiment_config
from nlp_lab.core.run_artifacts import (
    ENVIRONMENT_FILENAME,
    ERRORS_FILENAME,
    METRICS_FILENAME,
    PREDICTIONS_FILENAME,
    RESOLVED_CONFIG_FILENAME,
    RUN_METADATA_FILENAME,
    RUNTIME_FILENAME,
    SUMMARY_FILENAME,
    ErrorRecord,
    PredictionRecord,
    RunMetadata,
    RuntimeMeasurements,
    build_run_artifact_paths,
    build_run_metadata,
    collect_environment_info,
    initialize_run_artifacts,
    mark_run_completed,
    mark_run_failed,
    write_errors,
    write_metrics,
    write_predictions,
    write_runtime,
    write_summary,
)


@pytest.mark.unit
def test_build_run_artifact_paths_defines_standard_contract(tmp_path: Path) -> None:
    paths = build_run_artifact_paths(tmp_path / "run-001")

    assert paths.resolved_config.name == RESOLVED_CONFIG_FILENAME
    assert paths.run_metadata.name == RUN_METADATA_FILENAME
    assert paths.environment.name == ENVIRONMENT_FILENAME
    assert paths.metrics.name == METRICS_FILENAME
    assert paths.runtime.name == RUNTIME_FILENAME
    assert paths.predictions.name == PREDICTIONS_FILENAME
    assert paths.errors.name == ERRORS_FILENAME
    assert paths.summary.name == SUMMARY_FILENAME


@pytest.mark.unit
def test_initialize_run_artifacts_writes_resolved_config_run_and_environment(
    tmp_path: Path,
) -> None:
    config = load_layered_experiment_config(
        "configs/common/default.yaml",
        "configs/experiments/classification_baseline.yaml",
        overrides={"output_root": tmp_path, "batch_size": 32},
    )

    paths = initialize_run_artifacts(
        config,
        started_at=datetime(2026, 7, 30, 12, 55, 30, tzinfo=UTC),
    )

    assert paths.run_dir.parent == tmp_path
    assert paths.resolved_config.exists()
    assert paths.run_metadata.exists()
    assert paths.environment.exists()

    resolved_config = yaml.safe_load(paths.resolved_config.read_text(encoding="utf-8"))
    assert resolved_config["project"] == ProjectConfig(
        name="nlp-engineering-practical-lab"
    ).model_dump(mode="json")
    assert resolved_config["inference"]["batch_size"] == 32

    run_metadata = yaml.safe_load(paths.run_metadata.read_text(encoding="utf-8"))
    assert run_metadata["run_id"] == paths.run_dir.name
    assert run_metadata["experiment_name"] == "classification-baseline"
    assert run_metadata["task"] == "text-classification"
    assert run_metadata["status"] == "RUNNING"
    assert run_metadata["started_at"] == "2026-07-30T12:55:30Z"
    assert run_metadata["execution_mode"] == "local"


@pytest.mark.unit
def test_run_metadata_accepts_explicit_lifecycle_statuses() -> None:
    base_metadata = RunMetadata(
        run_id="run-001",
        experiment_name="classification-baseline",
        task="text-classification",
        status="CREATED",
        started_at=datetime(2026, 7, 30, 12, 55, 30, tzinfo=UTC),
        execution_mode="local",
    )

    assert base_metadata.status == "CREATED"


@pytest.mark.unit
def test_mark_run_completed_updates_status_and_completion_time(tmp_path: Path) -> None:
    config = load_layered_experiment_config(
        "configs/common/default.yaml",
        "configs/experiments/classification_baseline.yaml",
        overrides={"output_root": tmp_path},
    )
    paths = initialize_run_artifacts(
        config,
        started_at=datetime(2026, 7, 30, 12, 55, 30, tzinfo=UTC),
    )
    running_metadata = build_run_metadata(
        config,
        paths.run_dir.name,
        "RUNNING",
        datetime(2026, 7, 30, 12, 55, 30, tzinfo=UTC),
    )

    completed = mark_run_completed(
        paths.run_metadata,
        running_metadata,
        completed_at=datetime(2026, 7, 30, 12, 56, 12, tzinfo=UTC),
    )

    run_metadata = yaml.safe_load(paths.run_metadata.read_text(encoding="utf-8"))
    assert completed.status == "COMPLETED"
    assert run_metadata["status"] == "COMPLETED"
    assert run_metadata["completed_at"] == "2026-07-30T12:56:12Z"


@pytest.mark.unit
def test_mark_run_failed_preserves_artifacts_and_records_failure(tmp_path: Path) -> None:
    config = load_layered_experiment_config(
        "configs/common/default.yaml",
        "configs/experiments/classification_baseline.yaml",
        overrides={"output_root": tmp_path},
    )
    paths = initialize_run_artifacts(
        config,
        started_at=datetime(2026, 7, 30, 12, 55, 30, tzinfo=UTC),
    )
    running_metadata = build_run_metadata(
        config,
        paths.run_dir.name,
        "RUNNING",
        datetime(2026, 7, 30, 12, 55, 30, tzinfo=UTC),
    )

    try:
        raise RuntimeError("batch_size failed safely")
    except RuntimeError as exc:
        failed = mark_run_failed(
            paths.run_metadata,
            running_metadata,
            exc,
            failed_at=datetime(2026, 7, 30, 12, 56, 12, tzinfo=UTC),
        )

    run_metadata = yaml.safe_load(paths.run_metadata.read_text(encoding="utf-8"))
    assert paths.run_dir.exists()
    assert paths.resolved_config.exists()
    assert paths.environment.exists()
    assert failed.status == "FAILED"
    assert run_metadata["status"] == "FAILED"
    assert run_metadata["failed_at"] == "2026-07-30T12:56:12Z"
    assert run_metadata["exception_type"] == "RuntimeError"
    assert run_metadata["error_message"] == "batch_size failed safely"
    assert "RuntimeError: batch_size failed safely" in run_metadata["traceback_log"]


@pytest.mark.unit
def test_run_artifact_writers_emit_json_and_jsonl(tmp_path: Path) -> None:
    paths = build_run_artifact_paths(tmp_path)

    write_metrics(paths.metrics, {"accuracy": 0.91, "macro_f1": 0.8, "weighted_f1": 0.9})
    write_runtime(paths.runtime, RuntimeMeasurements(total_duration_seconds=42.1, batch_size=16))
    write_predictions(
        paths.predictions,
        [
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
    write_errors(
        paths.errors,
        [
            ErrorRecord(
                sample_id="test-047",
                true_label=1,
                predicted_label=0,
                confidence=0.88,
                error_type="false_negative",
            )
        ],
    )
    write_summary(paths.summary, "Classification Baseline", ["- Macro F1: 0.80"])

    assert yaml.safe_load(paths.metrics.read_text(encoding="utf-8"))["macro_f1"] == 0.8
    assert yaml.safe_load(paths.runtime.read_text(encoding="utf-8"))["batch_size"] == 16
    assert len(paths.predictions.read_text(encoding="utf-8").splitlines()) == 1
    assert len(paths.errors.read_text(encoding="utf-8").splitlines()) == 1
    assert paths.summary.read_text(encoding="utf-8").startswith("# Classification Baseline")


@pytest.mark.unit
def test_write_metrics_rejects_ambiguous_metric_names(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="metric names must be explicit"):
        write_metrics(tmp_path / "metrics.json", {"f1": 0.8})


@pytest.mark.unit
def test_collect_environment_info_omits_secret_like_fields() -> None:
    environment = collect_environment_info()

    assert "python_version" in environment
    assert "platform" in environment
    assert "git_commit" in environment
    assert "git_dirty" in environment
    assert not any("token" in key.lower() or "secret" in key.lower() for key in environment)
