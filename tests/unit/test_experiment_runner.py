import json
from pathlib import Path

import pytest

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
    assert json.loads(run.paths.metrics.read_text(encoding="utf-8"))["accuracy"] == 1.0
    assert json.loads(run.paths.runtime.read_text(encoding="utf-8"))["batch_size"] == 8
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
