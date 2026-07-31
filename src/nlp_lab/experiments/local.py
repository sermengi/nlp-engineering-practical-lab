from nlp_lab.core.config import ConfigOverrides
from nlp_lab.core.config.common import PathLike
from nlp_lab.experiments.dummy import failing_dummy_experiment, successful_dummy_experiment
from nlp_lab.experiments.protocol import ExperimentFn
from nlp_lab.experiments.runner import ExperimentRun, ExperimentRunner, default_common_config_path

DummyExperimentName = str


def resolve_dummy_experiment(name: DummyExperimentName) -> ExperimentFn:
    if name == "success":
        return successful_dummy_experiment
    if name == "failure":
        return failing_dummy_experiment
    msg = f"unknown dummy experiment: {name}"
    raise ValueError(msg)


def run_local_experiment(
    *,
    experiment_config_path: PathLike,
    common_config_path: PathLike | None = None,
    overrides: ConfigOverrides | None = None,
    experiment_fn: ExperimentFn | None = None,
    dummy_experiment: DummyExperimentName = "success",
) -> ExperimentRun:
    selected_experiment = experiment_fn or resolve_dummy_experiment(dummy_experiment)
    return ExperimentRunner().run(
        common_config_path=common_config_path or default_common_config_path(),
        experiment_config_path=experiment_config_path,
        experiment_fn=selected_experiment,
        overrides=overrides,
        execution_mode="local",
    )
