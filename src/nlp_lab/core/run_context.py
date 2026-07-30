import subprocess
from datetime import datetime
from pathlib import Path

from pydantic import field_validator

from nlp_lab.core.config import ExperimentConfig, generate_run_id
from nlp_lab.core.config.common import StrictConfigModel, ensure_non_empty
from nlp_lab.core.run_artifacts import (
    ExecutionMode,
    RunArtifactPaths,
    RunMetadata,
    RunStatus,
    build_run_artifact_paths,
)


class GitState(StrictConfigModel):
    commit: str | None
    dirty: bool


class RunContext(StrictConfigModel):
    run_id: str
    started_at: datetime
    output_dir: Path
    config: ExperimentConfig
    execution_mode: ExecutionMode
    worker_id: str | None = None
    remote_provider: str | None = None
    git: GitState
    status: RunStatus = "CREATED"

    @field_validator("run_id")
    @classmethod
    def validate_run_id(cls, value: str) -> str:
        return ensure_non_empty(value)

    @field_validator("worker_id", "remote_provider")
    @classmethod
    def validate_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return ensure_non_empty(value)

    @classmethod
    def create(
        cls,
        config: ExperimentConfig,
        started_at: datetime | None = None,
        run_id: str | None = None,
        execution_mode: ExecutionMode | None = None,
        worker_id: str | None = None,
        remote_provider: str | None = None,
        git: GitState | None = None,
    ) -> "RunContext":
        context_started_at = started_at or datetime.now().astimezone()
        context_run_id = run_id or generate_run_id(config, context_started_at)
        return cls(
            run_id=context_run_id,
            started_at=context_started_at,
            output_dir=config.runtime.output_root / context_run_id,
            config=config,
            execution_mode=execution_mode or config.runtime.environment,
            worker_id=worker_id,
            remote_provider=remote_provider,
            git=git or collect_git_state(),
        )

    @property
    def artifact_paths(self) -> RunArtifactPaths:
        return build_run_artifact_paths(self.output_dir)

    def to_run_metadata(
        self,
        completed_at: datetime | None = None,
        failed_at: datetime | None = None,
        exception_type: str | None = None,
        error_message: str | None = None,
        traceback_log: str | None = None,
    ) -> RunMetadata:
        return RunMetadata(
            run_id=self.run_id,
            experiment_name=self.config.experiment.name,
            task=self.config.experiment.task,
            status=self.status,
            started_at=self.started_at,
            completed_at=completed_at,
            failed_at=failed_at,
            execution_mode=self.execution_mode,
            exception_type=exception_type,
            error_message=error_message,
            traceback_log=traceback_log,
        )

    def with_status(self, status: RunStatus) -> "RunContext":
        return self.model_copy(update={"status": status})


def collect_git_state() -> GitState:
    return GitState(
        commit=_git_output(["git", "rev-parse", "HEAD"]),
        dirty=bool(_git_output(["git", "status", "--porcelain"])),
    )


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
