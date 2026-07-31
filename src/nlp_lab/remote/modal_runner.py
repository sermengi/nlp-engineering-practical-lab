from pathlib import Path
from typing import Any

from nlp_lab.core.config import ConfigOverrides
from nlp_lab.experiments.local import resolve_local_experiment
from nlp_lab.experiments.runner import ExperimentRun, ExperimentRunner


def run_modal_experiment(
    *,
    experiment_config_path: str | Path,
    common_config_path: str | Path = "configs/common/default.yaml",
    experiment: str = "hf-text-classification",
    output_root: str | Path = "/artifacts/experiments",
    seed: int | None = None,
    max_samples: int | None = None,
    batch_size: int | None = None,
) -> dict[str, Any]:
    overrides = build_modal_overrides(
        output_root=output_root,
        seed=seed,
        max_samples=max_samples,
        batch_size=batch_size,
    )
    run = ExperimentRunner().run(
        common_config_path=common_config_path,
        experiment_config_path=experiment_config_path,
        experiment_fn=resolve_local_experiment(experiment),
        overrides=overrides,
        execution_mode="modal",
    )
    return summarize_modal_run(run)


def build_modal_overrides(
    *,
    output_root: str | Path,
    seed: int | None = None,
    max_samples: int | None = None,
    batch_size: int | None = None,
) -> ConfigOverrides:
    return ConfigOverrides.model_validate(
        {
            "output_root": Path(output_root),
            "seed": seed,
            "max_samples": max_samples,
            "batch_size": batch_size,
        }
    )


def summarize_modal_run(run: ExperimentRun) -> dict[str, Any]:
    return {
        "run_id": run.metadata.run_id,
        "status": run.metadata.status,
        "experiment_name": run.metadata.experiment_name,
        "task": run.metadata.task,
        "execution_mode": run.metadata.execution_mode,
        "run_dir": str(run.paths.run_dir),
        "metrics": run.result.metrics,
        "artifact_paths": {
            "run_metadata": str(run.paths.run_metadata),
            "environment": str(run.paths.environment),
            "metrics": str(run.paths.metrics),
            "runtime": str(run.paths.runtime),
            "predictions": str(run.paths.predictions),
            "errors": str(run.paths.errors),
        },
    }
