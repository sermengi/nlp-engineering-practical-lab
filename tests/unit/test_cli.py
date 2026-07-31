import json
from pathlib import Path

import pytest

from nlp_lab.cli import main


@pytest.mark.unit
def test_cli_run_returns_zero_for_successful_dummy_experiment(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        [
            "run",
            "--config",
            "configs/experiments/classification_baseline.yaml",
            "--output-root",
            str(tmp_path),
            "--seed",
            "123",
            "--max-samples",
            "1",
            "--batch-size",
            "8",
        ]
    )

    stdout = capsys.readouterr().out
    run_dir = Path(stdout.strip().removeprefix("Run completed: "))
    metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert run_dir.parent == tmp_path
    assert metadata["status"] == "COMPLETED"


@pytest.mark.unit
def test_cli_run_returns_one_for_experiment_failure(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        [
            "run",
            "--config",
            "configs/experiments/classification_baseline.yaml",
            "--output-root",
            str(tmp_path),
            "--dummy-experiment",
            "failure",
        ]
    )

    stderr = capsys.readouterr().err
    run_dir = Path(stderr.split("Run artifacts: ", maxsplit=1)[1].strip())
    metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))

    assert exit_code == 1
    assert metadata["status"] == "FAILED"


@pytest.mark.unit
def test_cli_run_returns_two_for_config_validation_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        [
            "run",
            "--config",
            "configs/experiments/classification_baseline.yaml",
            "--output-root",
            str(tmp_path),
            "--max-samples",
            "0",
        ]
    )

    stderr = capsys.readouterr().err

    assert exit_code == 2
    assert "Config validation failed" in stderr
    assert not list(tmp_path.iterdir())
