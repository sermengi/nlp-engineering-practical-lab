import traceback
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field

from nlp_lab.core.artifacts.paths import (
    CONSOLE_LOG_FILENAME as CONSOLE_LOG_FILENAME,
)
from nlp_lab.core.artifacts.paths import (
    ENVIRONMENT_FILENAME as ENVIRONMENT_FILENAME,
)
from nlp_lab.core.artifacts.paths import (
    ERRORS_FILENAME as ERRORS_FILENAME,
)
from nlp_lab.core.artifacts.paths import (
    METRICS_FILENAME as METRICS_FILENAME,
)
from nlp_lab.core.artifacts.paths import (
    PREDICTIONS_FILENAME as PREDICTIONS_FILENAME,
)
from nlp_lab.core.artifacts.paths import (
    RESOLVED_CONFIG_FILENAME as RESOLVED_CONFIG_FILENAME,
)
from nlp_lab.core.artifacts.paths import (
    RUN_METADATA_FILENAME as RUN_METADATA_FILENAME,
)
from nlp_lab.core.artifacts.paths import (
    RUNTIME_FILENAME as RUNTIME_FILENAME,
)
from nlp_lab.core.artifacts.paths import (
    SUMMARY_FILENAME as SUMMARY_FILENAME,
)
from nlp_lab.core.artifacts.paths import (
    RunArtifactPaths as RunArtifactPaths,
)
from nlp_lab.core.artifacts.paths import (
    build_run_artifact_paths,
)
from nlp_lab.core.artifacts.serializers import (
    ArtifactSerializationError as ArtifactSerializationError,
)
from nlp_lab.core.artifacts.serializers import (
    append_jsonl as append_jsonl_file,
)
from nlp_lab.core.artifacts.serializers import (
    write_json,
    write_text_atomic,
    write_yaml,
)
from nlp_lab.core.config import ExperimentConfig, generate_run_id
from nlp_lab.core.config.common import StrictConfigModel, ensure_non_empty
from nlp_lab.core.environment import EnvironmentInfo
from nlp_lab.core.environment import collect_environment_info as collect_environment
from nlp_lab.core.observability import categorize_exception

if TYPE_CHECKING:
    from nlp_lab.core.experiment_result import ExperimentResult

RunStatus = Literal["CREATED", "RUNNING", "COMPLETED", "FAILED", "CANCELLED", "INTERRUPTED"]
ExecutionMode = Literal["local", "modal", "ci"]

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
    error_stage: str | None = None
    error_category: str | None = None


class RuntimeMeasurements(StrictConfigModel):
    total_duration_seconds: float | None = Field(default=None, ge=0.0)
    config_loading_seconds: float | None = Field(default=None, ge=0.0)
    data_loading_seconds: float | None = Field(default=None, ge=0.0)
    model_load_seconds: float | None = Field(default=None, ge=0.0)
    preprocessing_seconds: float | None = Field(default=None, ge=0.0)
    inference_seconds: float | None = Field(default=None, ge=0.0)
    evaluation_seconds: float | None = Field(default=None, ge=0.0)
    artifact_writing_seconds: float | None = Field(default=None, ge=0.0)
    sample_count: int | None = Field(default=None, ge=0)
    batch_count: int | None = Field(default=None, ge=0)
    samples_per_second: float | None = Field(default=None, ge=0.0)
    average_batch_latency_seconds: float | None = Field(default=None, ge=0.0)
    process_peak_memory_mb: float | None = Field(default=None, ge=0.0)
    process_current_memory_mb: float | None = Field(default=None, ge=0.0)
    cuda_peak_allocated_mb: float | None = Field(default=None, ge=0.0)
    cuda_peak_reserved_mb: float | None = Field(default=None, ge=0.0)
    memory_device: str | None = None
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


def create_run_directory(config: ExperimentConfig, run_id: str | None = None) -> RunArtifactPaths:
    resolved_run_id = run_id or generate_run_id(config)
    run_dir = config.runtime.output_root / resolved_run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return build_run_artifact_paths(run_dir)


def append_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    append_jsonl_file(path, records)


def write_resolved_config(path: Path, config: ExperimentConfig) -> None:
    payload = config.model_dump(mode="json")
    write_yaml(path, payload)


def write_run_metadata(path: Path, metadata_: RunMetadata) -> None:
    write_json(path, metadata_.model_dump(mode="json"))


def write_environment(
    path: Path, environment: dict[str, Any] | EnvironmentInfo | None = None
) -> None:
    payload = environment or collect_environment_info()
    if isinstance(payload, EnvironmentInfo):
        write_json(path, payload.model_dump(mode="json"))
    else:
        write_json(path, payload)


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
    write_text_atomic(path, content)


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
    metadata = build_run_metadata(
        config=config,
        run_id=paths.run_dir.name,
        status="RUNNING",
        started_at=run_started_at,
        execution_mode=execution_mode,
    )
    write_resolved_config(paths.resolved_config, config)
    write_run_metadata(paths.run_metadata, metadata)
    write_environment(
        paths.environment,
        collect_environment_info(execution_mode=metadata.execution_mode),
    )
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
    error_stage, error_category = categorize_exception(exception, "experiment")
    failed = metadata_.model_copy(
        update={
            "status": "FAILED",
            "failed_at": failed_at or datetime.now().astimezone(),
            "exception_type": type(exception).__name__,
            "error_message": safe_error_message(exception),
            "traceback_log": redact_sensitive_text(
                "".join(
                    traceback.format_exception(type(exception), exception, exception.__traceback__)
                )
            ),
            "error_stage": error_stage,
            "error_category": error_category,
        }
    )
    write_run_metadata(path, failed)
    return failed


def mark_run_interrupted(
    path: Path,
    metadata_: RunMetadata,
    exception: BaseException,
    interrupted_at: datetime | None = None,
) -> RunMetadata:
    interrupted = metadata_.model_copy(
        update={
            "status": "INTERRUPTED",
            "failed_at": interrupted_at or datetime.now().astimezone(),
            "exception_type": type(exception).__name__,
            "error_message": safe_error_message(exception),
            "traceback_log": redact_sensitive_text(
                "".join(
                    traceback.format_exception(type(exception), exception, exception.__traceback__)
                )
            ),
            "error_stage": getattr(exception, "stage", "experiment"),
            "error_category": "interrupted",
        }
    )
    write_run_metadata(path, interrupted)
    return interrupted


def safe_error_message(exception: BaseException) -> str:
    message = str(exception)
    if not message:
        return type(exception).__name__
    return redact_sensitive_text(message.replace("\n", " ").strip())


def redact_sensitive_text(text: str) -> str:
    from nlp_lab.core.observability import redact_sensitive_text as redact

    return redact(text)


def collect_environment_info(
    execution_mode: ExecutionMode = "local",
    worker_id: str | None = None,
) -> dict[str, Any]:
    return collect_environment(execution_mode=execution_mode, worker_id=worker_id).model_dump(
        mode="json"
    )
