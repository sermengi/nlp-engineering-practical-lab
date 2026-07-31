from datetime import datetime

from nlp_lab.core.artifacts.paths import RunArtifactPaths, build_run_artifact_paths
from nlp_lab.core.config import ExperimentConfig, generate_run_id
from nlp_lab.core.experiment_result import ExperimentResult
from nlp_lab.core.observability import log_run_event
from nlp_lab.core.run_artifacts import (
    ExecutionMode,
    RunMetadata,
    build_run_metadata,
    collect_environment_info,
    mark_run_completed,
    mark_run_failed,
    mark_run_interrupted,
    write_environment,
    write_experiment_result,
    write_resolved_config,
    write_run_metadata,
    write_runtime,
)


class ArtifactRunExistsError(FileExistsError):
    pass


class LocalFilesystemArtifactWriter:
    def create_run_directory(
        self, config: ExperimentConfig, run_id: str | None = None
    ) -> RunArtifactPaths:
        resolved_run_id = run_id or generate_run_id(config)
        run_dir = config.runtime.output_root / resolved_run_id
        if run_dir.exists():
            msg = f"run artifact directory already exists: {run_dir}"
            raise ArtifactRunExistsError(msg)
        run_dir.mkdir(parents=True)
        return build_run_artifact_paths(run_dir)

    def initialize_run(
        self,
        config: ExperimentConfig,
        run_id: str | None = None,
        started_at: datetime | None = None,
        execution_mode: ExecutionMode | None = None,
    ) -> tuple[RunArtifactPaths, RunMetadata]:
        run_started_at = started_at or datetime.now().astimezone()
        paths = self.create_run_directory(config, run_id or generate_run_id(config, run_started_at))
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
        return paths, metadata

    def write_result(self, paths: RunArtifactPaths, result: ExperimentResult) -> None:
        write_experiment_result(paths, result)

    def write_runtime(self, paths: RunArtifactPaths, result: ExperimentResult) -> None:
        write_runtime(paths.runtime, result.runtime)

    def complete_run(
        self,
        paths: RunArtifactPaths,
        metadata: RunMetadata,
        completed_at: datetime | None = None,
    ) -> RunMetadata:
        return mark_run_completed(paths.run_metadata, metadata, completed_at)

    def fail_run(
        self,
        paths: RunArtifactPaths,
        metadata: RunMetadata,
        exception: BaseException,
        failed_at: datetime | None = None,
    ) -> RunMetadata:
        return mark_run_failed(paths.run_metadata, metadata, exception, failed_at)

    def interrupt_run(
        self,
        paths: RunArtifactPaths,
        metadata: RunMetadata,
        exception: BaseException,
        interrupted_at: datetime | None = None,
    ) -> RunMetadata:
        return mark_run_interrupted(paths.run_metadata, metadata, exception, interrupted_at)

    def log_event(
        self,
        paths: RunArtifactPaths,
        metadata: RunMetadata,
        *,
        stage: str,
        level: str,
        message: str,
        extra: dict[str, object] | None = None,
    ) -> None:
        log_run_event(
            path=paths.console_log,
            run_id=metadata.run_id,
            experiment_name=metadata.experiment_name,
            execution_mode=metadata.execution_mode,
            stage=stage,
            level=level,
            message=message,
            extra=extra,
            emit_console=True,
            save_file=True,
        )


def build_local_artifact_writer() -> LocalFilesystemArtifactWriter:
    return LocalFilesystemArtifactWriter()
