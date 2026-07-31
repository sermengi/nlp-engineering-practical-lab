from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import ValidationError

from nlp_lab.core.artifacts.writer import LocalFilesystemArtifactWriter, build_local_artifact_writer
from nlp_lab.core.config import ConfigOverrides, load_layered_experiment_config
from nlp_lab.core.config.common import PathLike, RawConfig
from nlp_lab.core.experiment_result import ExperimentResult
from nlp_lab.core.observability import (
    StageTimer,
    collect_memory_measurements,
    reset_cuda_peak_memory,
    wrap_stage_error,
)
from nlp_lab.core.run_artifacts import ExecutionMode, RunArtifactPaths, RunMetadata
from nlp_lab.core.run_context import RunContext
from nlp_lab.core.seed import SeedSetupResult, set_global_seed
from nlp_lab.experiments.protocol import ExperimentFn

CONFIG_VALIDATION_EXIT_CODE = 2
EXPERIMENT_FAILURE_EXIT_CODE = 1
SUCCESS_EXIT_CODE = 0

ExperimentRunFailureKind = Literal["experiment", "artifact", "interrupted"]


@dataclass(frozen=True)
class ExperimentRun:
    context: RunContext
    paths: RunArtifactPaths
    metadata: RunMetadata
    result: ExperimentResult
    seed: SeedSetupResult


class ExperimentRunFailedError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        kind: ExperimentRunFailureKind,
        paths: RunArtifactPaths,
        metadata: RunMetadata,
        original: BaseException,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.paths = paths
        self.metadata = metadata
        self.original = original


class ExperimentRunner:
    def __init__(self, artifact_writer: LocalFilesystemArtifactWriter | None = None) -> None:
        self.artifact_writer = artifact_writer or build_local_artifact_writer()

    def run(
        self,
        *,
        common_config_path: PathLike,
        experiment_config_path: PathLike,
        experiment_fn: ExperimentFn,
        overrides: ConfigOverrides | RawConfig | None = None,
        execution_mode: ExecutionMode = "local",
        started_at: datetime | None = None,
        run_id: str | None = None,
    ) -> ExperimentRun:
        timer = StageTimer()
        with timer.measure("config_loading"):
            config = load_layered_experiment_config(
                common_config_path,
                experiment_config_path,
                overrides=overrides,
            )
        seed = set_global_seed(config.runtime.seed, deterministic=config.runtime.deterministic)
        context = RunContext.create(
            config,
            started_at=started_at,
            run_id=run_id,
            execution_mode=execution_mode,
        )
        paths, metadata = self.artifact_writer.initialize_run(
            config,
            run_id=context.run_id,
            started_at=context.started_at,
            execution_mode=execution_mode,
        )
        running_context = context.with_status("RUNNING")
        self.artifact_writer.log_event(
            paths,
            metadata,
            stage="run_start",
            level="INFO",
            message="experiment run started",
        )
        reset_cuda_peak_memory()

        try:
            self.artifact_writer.log_event(
                paths,
                metadata,
                stage="experiment",
                level="INFO",
                message="experiment function started",
            )
            result = experiment_fn(running_context)
            memory_measurements = collect_memory_measurements()
            self._merge_runtime_measurements(
                result,
                timer=timer,
                artifact_writing_seconds=None,
                memory_measurements=memory_measurements,
            )
            try:
                with timer.measure("artifact_writing"):
                    self.artifact_writer.write_result(paths, result)
            except Exception as exc:
                artifact_exception = wrap_stage_error(
                    "artifact_writing", "artifact_writing_error", exc
                )
                failed_metadata = self.artifact_writer.fail_run(
                    paths,
                    metadata,
                    artifact_exception,
                )
                self.artifact_writer.log_event(
                    paths,
                    failed_metadata,
                    stage="artifact_writing",
                    level="ERROR",
                    message="artifact writing failed",
                    extra={"error": str(exc)},
                )
                raise ExperimentRunFailedError(
                    f"experiment artifact writing failed: {exc}",
                    kind="artifact",
                    paths=paths,
                    metadata=failed_metadata,
                    original=exc,
                ) from exc
            self._merge_runtime_measurements(
                result,
                timer=timer,
                artifact_writing_seconds=timer.stage_seconds.get("artifact_writing"),
                memory_measurements=memory_measurements,
            )
            self.artifact_writer.write_runtime(paths, result)
        except KeyboardInterrupt as exc:
            interrupted_metadata = self.artifact_writer.interrupt_run(paths, metadata, exc)
            self.artifact_writer.log_event(
                paths,
                interrupted_metadata,
                stage="experiment",
                level="ERROR",
                message="experiment run interrupted",
                extra={"error": str(exc) or type(exc).__name__},
            )
            raise ExperimentRunFailedError(
                "experiment run interrupted",
                kind="interrupted",
                paths=paths,
                metadata=interrupted_metadata,
                original=exc,
            ) from exc
        except Exception as exc:
            failed_metadata = self.artifact_writer.fail_run(paths, metadata, exc)
            self.artifact_writer.log_event(
                paths,
                failed_metadata,
                stage=getattr(exc, "stage", "experiment"),
                level="ERROR",
                message="experiment run failed",
                extra={"error": str(exc)},
            )
            raise ExperimentRunFailedError(
                f"experiment run failed: {exc}",
                kind="experiment",
                paths=paths,
                metadata=failed_metadata,
                original=exc,
            ) from exc

        completed_metadata = self.artifact_writer.complete_run(paths, metadata)
        self.artifact_writer.log_event(
            paths,
            completed_metadata,
            stage="run_complete",
            level="INFO",
            message="experiment run completed",
            extra={"total_duration_seconds": result.runtime.total_duration_seconds},
        )
        completed_context = running_context.with_status("COMPLETED")
        return ExperimentRun(
            context=completed_context,
            paths=paths,
            metadata=completed_metadata,
            result=result,
            seed=seed,
        )

    def _merge_runtime_measurements(
        self,
        result: ExperimentResult,
        *,
        timer: StageTimer,
        artifact_writing_seconds: float | None,
        memory_measurements: dict[str, float | str | None],
    ) -> None:
        runtime = result.runtime
        updates = {
            "config_loading_seconds": runtime.config_loading_seconds
            or timer.stage_seconds.get("config_loading"),
            "artifact_writing_seconds": artifact_writing_seconds
            if artifact_writing_seconds is not None
            else runtime.artifact_writing_seconds,
            "total_duration_seconds": timer.total_duration_seconds,
            **memory_measurements,
        }
        result.runtime = runtime.model_copy(update=updates)


def is_config_validation_error(exception: Exception) -> bool:
    return isinstance(exception, (ValidationError, ValueError, FileNotFoundError))


def default_common_config_path() -> Path:
    return Path("configs/common/default.yaml")
