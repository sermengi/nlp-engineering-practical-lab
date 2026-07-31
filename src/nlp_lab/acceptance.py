from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

from nlp_lab.core.artifacts.serializers import write_json
from nlp_lab.core.parity import compare_run_parity, write_parity_report
from nlp_lab.experiments.runner import (
    EXPERIMENT_FAILURE_EXIT_CODE,
    SUCCESS_EXIT_CODE,
)

AcceptanceStatus = Literal["PASSED", "FAILED", "SKIPPED"]


@dataclass(frozen=True)
class AcceptanceOptions:
    config: Path = Path("configs/experiments/local_smoke_tiny_sst2.yaml")
    gpu_config: Path = Path("configs/experiments/modal_smoke_tiny_sst2_gpu.yaml")
    local_output_root: Path = Path("outputs/experiments/acceptance/local")
    failed_output_root: Path = Path("outputs/experiments/acceptance/failed")
    remote_download_root: Path = Path("outputs/modal-downloads")
    report_path: Path = Path("outputs/reports/acceptance.json")
    parity_report_path: Path = Path("outputs/reports/acceptance-parity.json")
    run_clean_checks: bool = True
    run_remote: bool = True
    run_gpu: bool = True
    dry_run: bool = False


@dataclass(frozen=True)
class CommandStep:
    name: str
    description: str
    command: list[str]
    expected_exit_code: int = SUCCESS_EXIT_CODE


@dataclass(frozen=True)
class AcceptanceStepResult:
    name: str
    status: AcceptanceStatus
    command: list[str] = field(default_factory=list)
    return_code: int | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AcceptanceReport:
    status: AcceptanceStatus
    steps: list[AcceptanceStepResult]
    local_run_dir: Path | None = None
    local_failed_run_dir: Path | None = None
    remote_cpu_run_dirs: list[Path] = field(default_factory=list)
    remote_gpu_run_dir: Path | None = None
    parity_report_path: Path | None = None

    @property
    def passed(self) -> bool:
        return self.status == "PASSED"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "local_run_dir": str(self.local_run_dir) if self.local_run_dir else None,
            "local_failed_run_dir": str(self.local_failed_run_dir)
            if self.local_failed_run_dir
            else None,
            "remote_cpu_run_dirs": [str(path) for path in self.remote_cpu_run_dirs],
            "remote_gpu_run_dir": str(self.remote_gpu_run_dir) if self.remote_gpu_run_dir else None,
            "parity_report_path": str(self.parity_report_path) if self.parity_report_path else None,
            "steps": [
                {
                    "name": step.name,
                    "status": step.status,
                    "command": step.command,
                    "return_code": step.return_code,
                    "details": step.details,
                }
                for step in self.steps
            ],
        }


def build_acceptance_plan(options: AcceptanceOptions) -> list[CommandStep]:
    steps: list[CommandStep] = []
    if options.run_clean_checks:
        steps.extend(
            [
                CommandStep(
                    name="dependency_installation",
                    description="Install all dependency groups from the lock file.",
                    command=["uv", "sync", "--all-groups"],
                ),
                CommandStep(
                    name="lint",
                    description="Run Ruff lint checks.",
                    command=["uv", "run", "--group", "development", "ruff", "check", "."],
                ),
                CommandStep(
                    name="format_check",
                    description="Run Ruff format check.",
                    command=[
                        "uv",
                        "run",
                        "--group",
                        "development",
                        "ruff",
                        "format",
                        "--check",
                        ".",
                    ],
                ),
                CommandStep(
                    name="type_check",
                    description="Run mypy over source and tests.",
                    command=["uv", "run", "--group", "development", "mypy", "src", "tests"],
                ),
                CommandStep(
                    name="unit_tests",
                    description="Run offline unit tests.",
                    command=[
                        "uv",
                        "run",
                        "--group",
                        "development",
                        "pytest",
                        "tests/unit",
                        "-m",
                        "unit",
                    ],
                ),
            ]
        )
    steps.extend(
        [
            CommandStep(
                name="local_successful_run",
                description="Run the local Hugging Face classification smoke experiment.",
                command=[
                    "uv",
                    "run",
                    "--group",
                    "ml",
                    "nlp-lab",
                    "run",
                    "--experiment",
                    "hf-text-classification",
                    "--config",
                    str(options.config),
                    "--output-root",
                    str(options.local_output_root),
                ],
            ),
            CommandStep(
                name="local_failed_run",
                description="Run an intentional local failure and preserve failure artifacts.",
                command=[
                    "uv",
                    "run",
                    "nlp-lab",
                    "run",
                    "--config",
                    str(options.config),
                    "--dummy-experiment",
                    "failure",
                    "--output-root",
                    str(options.failed_output_root),
                ],
                expected_exit_code=EXPERIMENT_FAILURE_EXIT_CODE,
            ),
        ]
    )
    if options.run_remote:
        steps.extend(
            [
                CommandStep(
                    name="modal_cpu_run_first",
                    description="Run classification smoke on Modal CPU.",
                    command=[
                        "uv",
                        "run",
                        "--group",
                        "remote",
                        "modal",
                        "run",
                        "modal_apps/classification.py",
                        "--config",
                        str(options.config),
                    ],
                ),
                CommandStep(
                    name="modal_cpu_run_second",
                    description=(
                        "Run classification smoke on Modal CPU again for cache reuse inspection."
                    ),
                    command=[
                        "uv",
                        "run",
                        "--group",
                        "remote",
                        "modal",
                        "run",
                        "modal_apps/classification.py",
                        "--config",
                        str(options.config),
                    ],
                ),
            ]
        )
        if options.run_gpu:
            steps.append(
                CommandStep(
                    name="modal_gpu_run",
                    description=(
                        "Run classification smoke on Modal GPU and capture device metadata."
                    ),
                    command=[
                        "uv",
                        "run",
                        "--group",
                        "remote",
                        "modal",
                        "run",
                        "modal_apps/classification.py",
                        "--config",
                        str(options.gpu_config),
                        "--gpu",
                    ],
                )
            )
        steps.append(
            CommandStep(
                name="modal_storage_report",
                description="Inspect Modal Volume storage and cache paths.",
                command=[
                    "uv",
                    "run",
                    "--group",
                    "remote",
                    "modal",
                    "run",
                    "modal_apps/storage.py",
                ],
            )
        )
    return steps


def run_acceptance(options: AcceptanceOptions) -> AcceptanceReport:
    steps = build_acceptance_plan(options)
    if options.dry_run:
        return write_acceptance_report(
            options.report_path,
            AcceptanceReport(
                status="SKIPPED",
                steps=[
                    AcceptanceStepResult(
                        name=step.name,
                        status="SKIPPED",
                        command=step.command,
                        details={"description": step.description, "dry_run": True},
                    )
                    for step in steps
                ],
            ),
        )

    results: list[AcceptanceStepResult] = []
    local_run_dir: Path | None = None
    local_failed_run_dir: Path | None = None
    remote_cpu_run_ids: list[str] = []
    remote_gpu_run_id: str | None = None

    for step in steps:
        completed = subprocess.run(step.command, text=True, capture_output=True, check=False)
        details: dict[str, Any] = {
            "description": step.description,
            "stdout_tail": tail(completed.stdout),
            "stderr_tail": tail(completed.stderr),
        }
        status: AcceptanceStatus = (
            "PASSED" if completed.returncode == step.expected_exit_code else "FAILED"
        )
        if status == "PASSED":
            if step.name == "local_successful_run":
                local_run_dir = parse_local_completed_run(completed.stdout)
                details["run_dir"] = str(local_run_dir)
                status = verify_run_status(
                    local_run_dir,
                    expected_status="COMPLETED",
                    details=details,
                )
            elif step.name == "local_failed_run":
                local_failed_run_dir = parse_local_failed_run(completed.stderr)
                details["run_dir"] = str(local_failed_run_dir)
                status = verify_run_status(
                    local_failed_run_dir,
                    expected_status="FAILED",
                    details=details,
                )
            elif step.name.startswith("modal_cpu_run"):
                run_id = parse_remote_run_id(completed.stdout)
                remote_cpu_run_ids.append(run_id)
                details["run_id"] = run_id
            elif step.name == "modal_gpu_run":
                remote_gpu_run_id = parse_remote_run_id(completed.stdout)
                details["run_id"] = remote_gpu_run_id
        results.append(
            AcceptanceStepResult(
                name=step.name,
                status=status,
                command=step.command,
                return_code=completed.returncode,
                details=details,
            )
        )
        if status == "FAILED":
            return write_acceptance_report(
                options.report_path,
                AcceptanceReport(
                    status="FAILED",
                    steps=results,
                    local_run_dir=local_run_dir,
                    local_failed_run_dir=local_failed_run_dir,
                ),
            )

    remote_cpu_run_dirs: list[Path] = []
    remote_gpu_run_dir: Path | None = None
    if options.run_remote:
        for index, run_id in enumerate(remote_cpu_run_ids, start=1):
            exported = export_modal_run(run_id, options.remote_download_root)
            remote_cpu_run_dirs.append(exported)
            results.append(
                AcceptanceStepResult(
                    name=f"export_modal_cpu_run_{index}",
                    status="PASSED",
                    details={"run_id": run_id, "run_dir": str(exported)},
                )
            )
        if remote_gpu_run_id is not None:
            exported = export_modal_run(remote_gpu_run_id, options.remote_download_root)
            remote_gpu_run_dir = exported
            results.append(
                AcceptanceStepResult(
                    name="export_modal_gpu_run",
                    status="PASSED",
                    details={"run_id": remote_gpu_run_id, "run_dir": str(exported)},
                )
            )
        if len(remote_cpu_run_dirs) >= 2:
            results.append(inspect_cache_reuse(remote_cpu_run_dirs[0], remote_cpu_run_dirs[1]))
        if local_run_dir is not None and remote_cpu_run_dirs:
            parity_report = compare_run_parity(local_run_dir, remote_cpu_run_dirs[-1])
            write_parity_report(options.parity_report_path, parity_report)
            results.append(
                AcceptanceStepResult(
                    name="local_remote_parity",
                    status="PASSED" if parity_report.passed else "FAILED",
                    command=[
                        "uv",
                        "run",
                        "nlp-lab",
                        "compare-runs",
                        "--local-run-dir",
                        str(local_run_dir),
                        "--remote-run-dir",
                        str(remote_cpu_run_dirs[-1]),
                        "--report",
                        str(options.parity_report_path),
                    ],
                    details={"parity_report": str(options.parity_report_path)},
                )
            )

    final_status: AcceptanceStatus = (
        "PASSED" if all(step.status == "PASSED" for step in results) else "FAILED"
    )
    return write_acceptance_report(
        options.report_path,
        AcceptanceReport(
            status=final_status,
            steps=results,
            local_run_dir=local_run_dir,
            local_failed_run_dir=local_failed_run_dir,
            remote_cpu_run_dirs=remote_cpu_run_dirs,
            remote_gpu_run_dir=remote_gpu_run_dir,
            parity_report_path=options.parity_report_path if options.run_remote else None,
        ),
    )


def export_modal_run(run_id: str, download_root: Path) -> Path:
    destination = download_root / run_id
    command = [
        "uv",
        "run",
        "--group",
        "remote",
        "modal",
        "volume",
        "get",
        "nlp-lab-storage",
        f"/experiments/{run_id}",
        str(destination),
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != SUCCESS_EXIT_CODE:
        msg = (
            f"failed to export Modal run {run_id}: "
            f"{tail(completed.stderr) or tail(completed.stdout)}"
        )
        raise RuntimeError(msg)
    return destination


def inspect_cache_reuse(first_run_dir: Path, second_run_dir: Path) -> AcceptanceStepResult:
    first_runtime = read_json(first_run_dir / "runtime.json")
    second_runtime = read_json(second_run_dir / "runtime.json")
    first_model_load = first_runtime.get("model_load_seconds")
    second_model_load = second_runtime.get("model_load_seconds")
    status: AcceptanceStatus = (
        "PASSED"
        if isinstance(first_model_load, int | float) and isinstance(second_model_load, int | float)
        else "FAILED"
    )
    return AcceptanceStepResult(
        name="cache_reuse_inspection",
        status=status,
        details={
            "first_run_dir": str(first_run_dir),
            "second_run_dir": str(second_run_dir),
            "first_model_load_seconds": first_model_load,
            "second_model_load_seconds": second_model_load,
            "note": (
                "Use these timings with Modal logs to confirm the second run reused Volume cache."
            ),
        },
    )


def verify_run_status(
    run_dir: Path,
    *,
    expected_status: str,
    details: dict[str, Any],
) -> AcceptanceStatus:
    metadata = read_json(run_dir / "run.json")
    actual_status = metadata.get("status")
    details["actual_status"] = actual_status
    return "PASSED" if actual_status == expected_status else "FAILED"


def read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def parse_local_completed_run(stdout: str) -> Path:
    for line in stdout.splitlines():
        if line.startswith("Run completed: "):
            return Path(line.removeprefix("Run completed: ").strip())
    raise ValueError("local completed run path was not found in stdout")


def parse_local_failed_run(stderr: str) -> Path:
    for line in stderr.splitlines():
        if line.startswith("Run artifacts: "):
            return Path(line.removeprefix("Run artifacts: ").strip())
    raise ValueError("local failed run path was not found in stderr")


def parse_remote_run_id(stdout: str) -> str:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        run_id = payload.get("run_id")
        if isinstance(run_id, str) and run_id:
            return run_id
    raise ValueError("remote run_id was not found in Modal stdout")


def write_acceptance_report(path: Path, report: AcceptanceReport) -> AcceptanceReport:
    write_json(path, report.to_dict())
    return report


def tail(text: str, *, max_lines: int = 30) -> str:
    lines = text.splitlines()
    return "\n".join(lines[-max_lines:])
