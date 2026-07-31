import json
from pathlib import Path

import pytest

from nlp_lab.cli import main
from tests.unit.parity_helpers import create_run_dir


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


@pytest.mark.unit
def test_cli_compare_runs_writes_report_and_returns_zero_for_parity(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    local_dir = tmp_path / "local-run"
    remote_dir = tmp_path / "remote-run"
    report_path = tmp_path / "parity.json"
    create_run_dir(local_dir)
    create_run_dir(remote_dir, execution_mode="modal")

    exit_code = main(
        [
            "compare-runs",
            "--local-run-dir",
            str(local_dir),
            "--remote-run-dir",
            str(remote_dir),
            "--report",
            str(report_path),
        ]
    )

    stdout = capsys.readouterr().out
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert "Parity status: PASSED" in stdout
    assert report["status"] == "PASSED"


@pytest.mark.unit
def test_cli_compare_runs_returns_one_for_parity_failure(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    local_dir = tmp_path / "local-run"
    remote_dir = tmp_path / "remote-run"
    create_run_dir(local_dir)
    create_run_dir(remote_dir, predicted_label="negative")

    exit_code = main(
        [
            "compare-runs",
            "--local-run-dir",
            str(local_dir),
            "--remote-run-dir",
            str(remote_dir),
        ]
    )

    stdout = capsys.readouterr().out

    assert exit_code == 1
    assert "Parity status: FAILED" in stdout


@pytest.mark.unit
def test_cli_compare_runs_returns_two_when_artifacts_cannot_be_read(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        [
            "compare-runs",
            "--local-run-dir",
            str(tmp_path / "missing-local"),
            "--remote-run-dir",
            str(tmp_path / "missing-remote"),
        ]
    )

    stderr = capsys.readouterr().err

    assert exit_code == 2
    assert "Run parity comparison failed" in stderr


@pytest.mark.unit
def test_cli_acceptance_test_dry_run_writes_report(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report_path = tmp_path / "acceptance.json"
    parity_report_path = tmp_path / "parity.json"

    exit_code = main(
        [
            "acceptance-test",
            "--dry-run",
            "--skip-clean-checks",
            "--skip-remote",
            "--report",
            str(report_path),
            "--parity-report",
            str(parity_report_path),
        ]
    )

    stdout = capsys.readouterr().out
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert "Acceptance test plan:" in stdout
    assert "Acceptance status: SKIPPED" in stdout
    assert report["status"] == "SKIPPED"
