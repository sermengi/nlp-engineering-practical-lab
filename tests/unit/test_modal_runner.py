from pathlib import Path

import pytest

from nlp_lab.experiments.runner import ExperimentRunFailedError
from nlp_lab.remote import run_modal_experiment


@pytest.mark.unit
def test_run_modal_experiment_uses_common_runner_and_returns_short_summary(tmp_path: Path) -> None:
    summary = run_modal_experiment(
        experiment_config_path="configs/experiments/local_smoke_tiny_sst2.yaml",
        experiment="dummy-success",
        output_root=tmp_path,
        batch_size=2,
    )

    assert summary["status"] == "COMPLETED"
    assert summary["execution_mode"] == "modal"
    assert Path(str(summary["run_dir"])).parent == tmp_path
    assert "accuracy" in summary["metrics"]
    assert "predictions" not in summary
    assert set(summary["artifact_paths"]) >= {
        "run_metadata",
        "environment",
        "metrics",
        "runtime",
        "predictions",
        "errors",
    }


@pytest.mark.unit
def test_run_modal_experiment_preserves_failed_run_artifacts(tmp_path: Path) -> None:
    with pytest.raises(ExperimentRunFailedError) as failure:
        run_modal_experiment(
            experiment_config_path="configs/experiments/local_smoke_tiny_sst2.yaml",
            experiment="dummy-failure",
            output_root=tmp_path,
        )

    paths = failure.value.paths
    assert paths.run_dir.exists()
    assert '"status": "FAILED"' in paths.run_metadata.read_text(encoding="utf-8")
