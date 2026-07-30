import json
import platform
import socket
import subprocess
import sys
import traceback
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import yaml
from pydantic import Field

from nlp_lab.core.config import ExperimentConfig, generate_run_id
from nlp_lab.core.config.common import StrictConfigModel, ensure_non_empty

if TYPE_CHECKING:
    from nlp_lab.core.experiment_result import ExperimentResult

RunStatus = Literal["CREATED", "RUNNING", "COMPLETED", "FAILED", "CANCELLED", "INTERRUPTED"]
ExecutionMode = Literal["local", "modal", "ci"]

RESOLVED_CONFIG_FILENAME = "config.resolved.yaml"
RUN_METADATA_FILENAME = "run.json"
ENVIRONMENT_FILENAME = "environment.json"
METRICS_FILENAME = "metrics.json"
RUNTIME_FILENAME = "runtime.json"
PREDICTIONS_FILENAME = "predictions.jsonl"
ERRORS_FILENAME = "errors.jsonl"
CONSOLE_LOG_FILENAME = "console.log"
SUMMARY_FILENAME = "summary.md"
AMBIGUOUS_METRIC_NAMES = {"f1", "precision", "recall"}


class RunMetadata(StrictConfigModel):
    run_id: str
    experiment_name: str
    task: str
    status: RunStatus
    started_at: datetime
    completed_at: datetime | None = None
    failed_at: datetime | None = None
    execution_mode: ExecutionMode
    exception_type: str | None = None
    error_message: str | None = None
    traceback_log: str | None = None


class RuntimeMeasurements(StrictConfigModel):
    total_duration_seconds: float | None = Field(default=None, ge=0.0)
    model_load_seconds: float | None = Field(default=None, ge=0.0)
    inference_seconds: float | None = Field(default=None, ge=0.0)
    evaluation_seconds: float | None = Field(default=None, ge=0.0)
    samples_per_second: float | None = Field(default=None, ge=0.0)
    batch_size: int = Field(..., gt=0)


class PredictionRecord(StrictConfigModel):
    sample_id: str
    text: str | None = None
    true_label: int | str | None = None
    predicted_label: int | str
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    is_correct: bool | None = None


class ErrorRecord(StrictConfigModel):
    sample_id: str
    true_label: int | str | None = None
    predicted_label: int | str
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    error_type: str


class RunArtifactPaths(StrictConfigModel):
    run_dir: Path
    resolved_config: Path
    run_metadata: Path
    environment: Path
    metrics: Path
    runtime: Path
    predictions: Path
    errors: Path
    console_log: Path
    summary: Path


def build_run_artifact_paths(run_dir: Path) -> RunArtifactPaths:
    return RunArtifactPaths(
        run_dir=run_dir,
        resolved_config=run_dir / RESOLVED_CONFIG_FILENAME,
        run_metadata=run_dir / RUN_METADATA_FILENAME,
        environment=run_dir / ENVIRONMENT_FILENAME,
        metrics=run_dir / METRICS_FILENAME,
        runtime=run_dir / RUNTIME_FILENAME,
        predictions=run_dir / PREDICTIONS_FILENAME,
        errors=run_dir / ERRORS_FILENAME,
        console_log=run_dir / CONSOLE_LOG_FILENAME,
        summary=run_dir / SUMMARY_FILENAME,
    )


def create_run_directory(config: ExperimentConfig, run_id: str | None = None) -> RunArtifactPaths:
    resolved_run_id = run_id or generate_run_id(config)
    run_dir = config.runtime.output_root / resolved_run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return build_run_artifact_paths(run_dir)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def append_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, sort_keys=True, ensure_ascii=True) + "\n")


def write_resolved_config(path: Path, config: ExperimentConfig) -> None:
    payload = config.model_dump(mode="json")
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def write_run_metadata(path: Path, metadata_: RunMetadata) -> None:
    write_json(path, metadata_.model_dump(mode="json"))


def write_environment(path: Path, environment: dict[str, Any] | None = None) -> None:
    write_json(path, environment or collect_environment_info())


def write_metrics(path: Path, metrics: dict[str, float]) -> None:
    ambiguous_names = sorted(set(metrics) & AMBIGUOUS_METRIC_NAMES)
    if ambiguous_names:
        msg = f"metric names must be explicit, found ambiguous names: {', '.join(ambiguous_names)}"
        raise ValueError(msg)
    write_json(path, metrics)


def write_runtime(path: Path, runtime: RuntimeMeasurements) -> None:
    write_json(path, runtime.model_dump(mode="json"))


def write_predictions(path: Path, records: list[PredictionRecord]) -> None:
    append_jsonl(path, [record.model_dump(mode="json") for record in records])


def write_errors(path: Path, records: list[ErrorRecord]) -> None:
    append_jsonl(path, [record.model_dump(mode="json") for record in records])


def write_summary(path: Path, title: str, lines: list[str]) -> None:
    safe_title = ensure_non_empty(title)
    content = "\n".join([f"# {safe_title}", "", *lines, ""])
    path.write_text(content, encoding="utf-8")


def write_experiment_result(paths: RunArtifactPaths, result: "ExperimentResult") -> None:
    write_metrics(paths.metrics, result.metrics)
    write_runtime(paths.runtime, result.runtime)
    if result.predictions:
        write_predictions(paths.predictions, result.predictions)
    if result.errors:
        write_errors(paths.errors, result.errors)
    if result.notes or result.artifacts:
        write_summary(paths.summary, "Experiment Summary", result.summary_lines())


def build_run_metadata(
    config: ExperimentConfig,
    run_id: str,
    status: RunStatus,
    started_at: datetime,
    execution_mode: ExecutionMode | None = None,
) -> RunMetadata:
    return RunMetadata(
        run_id=run_id,
        experiment_name=config.experiment.name,
        task=config.experiment.task,
        status=status,
        started_at=started_at,
        execution_mode=execution_mode or config.runtime.environment,
    )


def initialize_run_artifacts(
    config: ExperimentConfig,
    run_id: str | None = None,
    started_at: datetime | None = None,
    execution_mode: ExecutionMode | None = None,
) -> RunArtifactPaths:
    run_started_at = started_at or datetime.now().astimezone()
    paths = create_run_directory(config, run_id or generate_run_id(config, run_started_at))
    write_resolved_config(paths.resolved_config, config)
    write_run_metadata(
        paths.run_metadata,
        build_run_metadata(
            config=config,
            run_id=paths.run_dir.name,
            status="RUNNING",
            started_at=run_started_at,
            execution_mode=execution_mode,
        ),
    )
    write_environment(paths.environment)
    return paths


def mark_run_completed(
    path: Path,
    metadata_: RunMetadata,
    completed_at: datetime | None = None,
) -> RunMetadata:
    completed = metadata_.model_copy(
        update={
            "status": "COMPLETED",
            "completed_at": completed_at or datetime.now().astimezone(),
        }
    )
    write_run_metadata(path, completed)
    return completed


def mark_run_failed(
    path: Path,
    metadata_: RunMetadata,
    exception: BaseException,
    failed_at: datetime | None = None,
) -> RunMetadata:
    failed = metadata_.model_copy(
        update={
            "status": "FAILED",
            "failed_at": failed_at or datetime.now().astimezone(),
            "exception_type": type(exception).__name__,
            "error_message": safe_error_message(exception),
            "traceback_log": "".join(
                traceback.format_exception(type(exception), exception, exception.__traceback__)
            ),
        }
    )
    write_run_metadata(path, failed)
    return failed


def safe_error_message(exception: BaseException) -> str:
    message = str(exception)
    if not message:
        return type(exception).__name__
    return message.replace("\n", " ").strip()


def collect_environment_info() -> dict[str, Any]:
    torch_info = _collect_torch_info()
    return {
        "python_version": sys.version,
        "os": platform.system(),
        "os_release": platform.release(),
        "platform": platform.platform(),
        "pytorch_version": _package_version("torch"),
        "transformers_version": _package_version("transformers"),
        "cuda_available": torch_info["cuda_available"],
        "cuda_version": torch_info["cuda_version"],
        "gpu_name": torch_info["gpu_name"],
        "cpu": platform.processor() or platform.machine(),
        "git_commit": _git_output(["git", "rev-parse", "HEAD"]),
        "git_dirty": bool(_git_output(["git", "status", "--porcelain"])),
        "hostname": socket.gethostname(),
    }


def _package_version(package_name: str) -> str | None:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return None


def _collect_torch_info() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"cuda_available": False, "cuda_version": None, "gpu_name": None}

    cuda_available = bool(torch.cuda.is_available())
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
    return {
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "gpu_name": gpu_name,
    }


def _git_output(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    output = completed.stdout.strip()
    return output or None
