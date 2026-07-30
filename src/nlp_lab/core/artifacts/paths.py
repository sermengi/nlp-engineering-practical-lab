from pathlib import Path

from nlp_lab.core.config.common import StrictConfigModel

RESOLVED_CONFIG_FILENAME = "config.resolved.yaml"
RUN_METADATA_FILENAME = "run.json"
ENVIRONMENT_FILENAME = "environment.json"
METRICS_FILENAME = "metrics.json"
RUNTIME_FILENAME = "runtime.json"
PREDICTIONS_FILENAME = "predictions.jsonl"
ERRORS_FILENAME = "errors.jsonl"
CONSOLE_LOG_FILENAME = "console.log"
SUMMARY_FILENAME = "summary.md"


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
