import json
from pathlib import Path

import pytest

from nlp_lab.acceptance import (
    AcceptanceOptions,
    build_acceptance_plan,
    parse_remote_run_id,
    run_acceptance,
)


@pytest.mark.unit
def test_build_acceptance_plan_covers_local_remote_export_prerequisites() -> None:
    steps = build_acceptance_plan(AcceptanceOptions())
    names = [step.name for step in steps]

    assert names[:5] == [
        "dependency_installation",
        "lint",
        "format_check",
        "type_check",
        "unit_tests",
    ]
    assert "local_successful_run" in names
    assert "local_failed_run" in names
    assert "modal_cpu_run_first" in names
    assert "modal_cpu_run_second" in names
    assert "modal_gpu_run" in names
    assert "modal_storage_report" in names


@pytest.mark.unit
def test_acceptance_plan_can_skip_remote_and_gpu_steps() -> None:
    steps = build_acceptance_plan(
        AcceptanceOptions(run_clean_checks=False, run_remote=False, run_gpu=False)
    )
    names = [step.name for step in steps]

    assert names == ["local_successful_run", "local_failed_run"]


@pytest.mark.unit
def test_run_acceptance_dry_run_writes_skipped_report(tmp_path: Path) -> None:
    report_path = tmp_path / "acceptance.json"

    report = run_acceptance(
        AcceptanceOptions(report_path=report_path, dry_run=True, run_clean_checks=False)
    )

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert report.status == "SKIPPED"
    assert payload["status"] == "SKIPPED"
    assert {step["status"] for step in payload["steps"]} == {"SKIPPED"}


@pytest.mark.unit
def test_parse_remote_run_id_reads_json_summary_from_modal_logs() -> None:
    stdout = "\n".join(
        [
            "Modal log line",
            '{"status": "COMPLETED", "run_id": "run-20260731-abc123"}',
        ]
    )

    assert parse_remote_run_id(stdout) == "run-20260731-abc123"
