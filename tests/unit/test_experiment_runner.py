import json
from pathlib import Path

import pytest

from nlp_lab.core.experiment_result import ExperimentResult
from nlp_lab.core.observability import wrap_stage_error
from nlp_lab.core.run_context import RunContext
from nlp_lab.experiments.dummy import failing_dummy_experiment, successful_dummy_experiment
from nlp_lab.experiments.runner import ExperimentRunFailedError, ExperimentRunner


@pytest.mark.unit
def test_experiment_runner_completes_successful_run(tmp_path: Path) -> None:
    run = ExperimentRunner().run(
        common_config_path="configs/common/default.yaml",
        experiment_config_path="configs/experiments/classification_baseline.yaml",
        overrides={"output_root": tmp_path, "seed": 123, "max_samples": 2, "batch_size": 8},
        experiment_fn=successful_dummy_experiment,
        execution_mode="local",
        run_id="run-001",
    )

    metadata = json.loads(run.paths.run_metadata.read_text(encoding="utf-8"))

    assert run.context.status == "COMPLETED"
    assert run.seed.seed == 123
    assert run.paths.run_dir == tmp_path / "run-001"
    assert metadata["status"] == "COMPLETED"
    assert metadata["execution_mode"] == "local"
    assert run.paths.resolved_config.exists()
    assert run.paths.environment.exists()
    runtime = json.loads(run.paths.runtime.read_text(encoding="utf-8"))
    log_records = [
        json.loads(line) for line in run.paths.console_log.read_text(encoding="utf-8").splitlines()
    ]
    assert json.loads(run.paths.metrics.read_text(encoding="utf-8"))["accuracy"] == 1.0
    assert runtime["batch_size"] == 8
    assert runtime["config_loading_seconds"] is not None
    assert runtime["artifact_writing_seconds"] is not None
    assert runtime["sample_count"] == 2
    assert runtime["batch_count"] == 1
    assert runtime["process_peak_memory_mb"] is not None
    assert {record["stage"] for record in log_records} >= {
        "run_start",
        "experiment",
        "run_complete",
    }
    assert len(run.paths.predictions.read_text(encoding="utf-8").splitlines()) == 2


@pytest.mark.unit
def test_experiment_runner_marks_failed_run_and_preserves_artifacts(tmp_path: Path) -> None:
    with pytest.raises(ExperimentRunFailedError) as raised:
        ExperimentRunner().run(
            common_config_path="configs/common/default.yaml",
            experiment_config_path="configs/experiments/classification_baseline.yaml",
            overrides={"output_root": tmp_path},
            experiment_fn=failing_dummy_experiment,
            execution_mode="local",
            run_id="run-001",
        )

    paths = raised.value.paths
    metadata = json.loads(paths.run_metadata.read_text(encoding="utf-8"))

    assert paths.run_dir.exists()
    assert paths.resolved_config.exists()
    assert paths.environment.exists()
    assert metadata["status"] == "FAILED"
    assert metadata["exception_type"] == "RuntimeError"
    assert "intentional dummy failure" in metadata["error_message"]
    assert "traceback_log" in metadata
    assert metadata["error_stage"] == "experiment"
    assert metadata["error_category"] == "experiment_error"
    assert paths.console_log.exists()


def staged_failure_experiment(context: RunContext) -> ExperimentResult:
    try:
        raise ValueError("bad text column")
    except ValueError as exc:
        raise wrap_stage_error("data_loading", "data_error", exc) from exc


@pytest.mark.unit
def test_experiment_runner_records_stage_and_category_for_staged_errors(tmp_path: Path) -> None:
    with pytest.raises(ExperimentRunFailedError) as raised:
        ExperimentRunner().run(
            common_config_path="configs/common/default.yaml",
            experiment_config_path="configs/experiments/classification_baseline.yaml",
            overrides={"output_root": tmp_path},
            experiment_fn=staged_failure_experiment,
            execution_mode="local",
            run_id="run-001",
        )

    metadata = json.loads(raised.value.paths.run_metadata.read_text(encoding="utf-8"))

    assert metadata["status"] == "FAILED"
    assert metadata["error_stage"] == "data_loading"
    assert metadata["error_category"] == "data_error"


def interrupted_experiment(context: RunContext) -> ExperimentResult:
    raise KeyboardInterrupt()


@pytest.mark.unit
def test_experiment_runner_marks_interrupted_run(tmp_path: Path) -> None:
    with pytest.raises(ExperimentRunFailedError) as raised:
        ExperimentRunner().run(
            common_config_path="configs/common/default.yaml",
            experiment_config_path="configs/experiments/classification_baseline.yaml",
            overrides={"output_root": tmp_path},
            experiment_fn=interrupted_experiment,
            execution_mode="local",
            run_id="run-001",
        )

    metadata = json.loads(raised.value.paths.run_metadata.read_text(encoding="utf-8"))

    assert raised.value.kind == "interrupted"
    assert metadata["status"] == "INTERRUPTED"
    assert metadata["error_category"] == "interrupted"
    assert raised.value.paths.console_log.exists()


def secret_failure_experiment(context: RunContext) -> ExperimentResult:
    raise RuntimeError("failed with Authorization: Bearer hf_abcd1234abcd1234")


@pytest.mark.unit
def test_experiment_runner_redacts_secrets_from_failure_metadata_and_log(tmp_path: Path) -> None:
    with pytest.raises(ExperimentRunFailedError) as raised:
        ExperimentRunner().run(
            common_config_path="configs/common/default.yaml",
            experiment_config_path="configs/experiments/classification_baseline.yaml",
            overrides={"output_root": tmp_path},
            experiment_fn=secret_failure_experiment,
            execution_mode="local",
            run_id="run-001",
        )

    metadata_text = raised.value.paths.run_metadata.read_text(encoding="utf-8")
    console_text = raised.value.paths.console_log.read_text(encoding="utf-8")

    assert "hf_abcd1234abcd1234" not in metadata_text
    assert "hf_abcd1234abcd1234" not in console_text
    assert "[REDACTED]" in metadata_text
    assert "[REDACTED]" in console_text
