import json
from pathlib import Path

import pytest

from nlp_lab.core.parity import compare_run_parity, write_parity_report
from tests.unit.parity_helpers import create_run_dir, read_jsonl, write_jsonl


@pytest.mark.unit
def test_compare_run_parity_passes_with_tolerated_confidence_and_environment_diff(
    tmp_path: Path,
) -> None:
    local_dir = tmp_path / "local-run"
    remote_dir = tmp_path / "remote-run"
    create_run_dir(local_dir, confidence=0.9000, execution_mode="local")
    create_run_dir(remote_dir, confidence=0.9005, execution_mode="modal")

    report = compare_run_parity(
        local_dir,
        remote_dir,
        metric_tolerance=1e-6,
        confidence_tolerance=1e-3,
    )

    assert report.status == "PASSED"
    checks = {check.name: check for check in report.checks}
    assert checks["resolved_config"].details["local_hash"] == checks["resolved_config"].details[
        "remote_hash"
    ]
    assert checks["prediction_schema"].status == "PASSED"
    assert checks["metrics"].status == "PASSED"
    assert checks["environment"].status == "PASSED"
    assert checks["environment"].details["differences"]["execution_mode"] == {
        "local": "local",
        "remote": "modal",
    }


@pytest.mark.unit
def test_compare_run_parity_fails_on_prediction_and_metric_drift(tmp_path: Path) -> None:
    local_dir = tmp_path / "local-run"
    remote_dir = tmp_path / "remote-run"
    create_run_dir(local_dir, predicted_label="positive", accuracy=1.0)
    create_run_dir(remote_dir, predicted_label="negative", accuracy=0.0)

    report = compare_run_parity(local_dir, remote_dir)
    checks = {check.name: check for check in report.checks}

    assert report.status == "FAILED"
    assert checks["predictions"].details["label_mismatches"] == ["sample-001"]
    assert checks["metrics"].details["value_mismatches"][0]["metric"] == "accuracy"


@pytest.mark.unit
def test_compare_run_parity_reports_config_schema_and_artifact_differences(
    tmp_path: Path,
) -> None:
    local_dir = tmp_path / "local-run"
    remote_dir = tmp_path / "remote-run"
    create_run_dir(local_dir)
    create_run_dir(remote_dir, model_revision="remote-revision", include_errors=False)
    remote_predictions = read_jsonl(remote_dir / "predictions.jsonl")
    remote_predictions[0].pop("is_correct")
    write_jsonl(remote_dir / "predictions.jsonl", remote_predictions)

    report = compare_run_parity(local_dir, remote_dir)
    checks = {check.name: check for check in report.checks}

    assert report.status == "FAILED"
    assert checks["artifact_completeness"].details["remote_missing"] == ["errors"]
    assert checks["resolved_config"].details["mismatched_paths"] == ["model"]
    assert checks["prediction_schema"].details["remote_missing_required_fields"] == ["is_correct"]


@pytest.mark.unit
def test_write_parity_report_writes_json_summary(tmp_path: Path) -> None:
    local_dir = tmp_path / "local-run"
    remote_dir = tmp_path / "remote-run"
    report_path = tmp_path / "reports" / "parity.json"
    create_run_dir(local_dir)
    create_run_dir(remote_dir)

    report = compare_run_parity(local_dir, remote_dir)
    write_parity_report(report_path, report)

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["status"] == "PASSED"
    assert payload["local_run_dir"] == str(local_dir)
