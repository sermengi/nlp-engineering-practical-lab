from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from nlp_lab.core.config import load_layered_experiment_config
from nlp_lab.core.run_context import GitState, RunContext


@pytest.mark.unit
def test_run_context_create_holds_run_metadata_without_runtime_objects(tmp_path: Path) -> None:
    config = load_layered_experiment_config(
        "configs/common/default.yaml",
        "configs/experiments/classification_baseline.yaml",
        overrides={"output_root": tmp_path},
    )

    context = RunContext.create(
        config,
        started_at=datetime(2026, 7, 30, 12, 55, 30, tzinfo=UTC),
        git=GitState(commit="abc123", dirty=False),
    )

    assert context.run_id.startswith("20260730-125530_classification-baseline_")
    assert context.started_at == datetime(2026, 7, 30, 12, 55, 30, tzinfo=UTC)
    assert context.output_dir == tmp_path / context.run_id
    assert context.config is config
    assert context.execution_mode == "local"
    assert context.git.commit == "abc123"
    assert context.git.dirty is False
    assert context.status == "CREATED"
    assert "model_object" not in RunContext.model_fields
    assert "dataset_object" not in RunContext.model_fields


@pytest.mark.unit
def test_run_context_supports_remote_metadata(tmp_path: Path) -> None:
    config = load_layered_experiment_config(
        "configs/common/default.yaml",
        "configs/experiments/classification_baseline.yaml",
        overrides={"output_root": tmp_path},
    )

    context = RunContext.create(
        config,
        run_id="run-001",
        execution_mode="modal",
        worker_id="worker-safe-id",
        remote_provider="modal",
        git=GitState(commit=None, dirty=True),
    )

    assert context.execution_mode == "modal"
    assert context.worker_id == "worker-safe-id"
    assert context.remote_provider == "modal"


@pytest.mark.unit
def test_run_context_converts_to_run_metadata(tmp_path: Path) -> None:
    config = load_layered_experiment_config(
        "configs/common/default.yaml",
        "configs/experiments/classification_baseline.yaml",
        overrides={"output_root": tmp_path},
    )
    context = RunContext.create(
        config,
        run_id="run-001",
        started_at=datetime(2026, 7, 30, 12, 55, 30, tzinfo=UTC),
        git=GitState(commit="abc123", dirty=False),
    ).with_status("RUNNING")

    metadata = context.to_run_metadata()

    assert metadata.run_id == "run-001"
    assert metadata.experiment_name == "classification-baseline"
    assert metadata.task == "text-classification"
    assert metadata.status == "RUNNING"
    assert metadata.execution_mode == "local"


@pytest.mark.unit
def test_run_context_rejects_unknown_runtime_objects(tmp_path: Path) -> None:
    config = load_layered_experiment_config(
        "configs/common/default.yaml",
        "configs/experiments/classification_baseline.yaml",
        overrides={"output_root": tmp_path},
    )

    with pytest.raises(ValidationError):
        RunContext(
            run_id="run-001",
            started_at=datetime(2026, 7, 30, 12, 55, 30, tzinfo=UTC),
            output_dir=tmp_path / "run-001",
            config=config,
            execution_mode="local",
            git=GitState(commit="abc123", dirty=False),
            model_object=object(),
        )
