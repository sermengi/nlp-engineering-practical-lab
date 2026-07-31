from nlp_lab.core.config import ConfigOverrides
from nlp_lab.core.config.common import PathLike
from nlp_lab.experiments.dummy import failing_dummy_experiment, successful_dummy_experiment
from nlp_lab.experiments.protocol import ExperimentFn
from nlp_lab.experiments.runner import ExperimentRun, ExperimentRunner, default_common_config_path
from nlp_lab.experiments.text_classification import hf_text_classification_experiment

DummyExperimentName = str
LocalExperimentName = str


def resolve_dummy_experiment(name: DummyExperimentName) -> ExperimentFn:
    if name == "success":
        return successful_dummy_experiment
    if name == "failure":
        return failing_dummy_experiment
    msg = f"unknown dummy experiment: {name}"
    raise ValueError(msg)


def resolve_local_experiment(name: LocalExperimentName) -> ExperimentFn:
    if name == "dummy-success":
        return successful_dummy_experiment
    if name == "dummy-failure":
        return failing_dummy_experiment
    if name == "hf-text-classification":
        return hf_text_classification_experiment
    msg = f"unknown local experiment: {name}"
    raise ValueError(msg)


def run_local_experiment(
    *,
    experiment_config_path: PathLike,
    common_config_path: PathLike | None = None,
    overrides: ConfigOverrides | None = None,
    experiment_fn: ExperimentFn | None = None,
    experiment: LocalExperimentName = "dummy-success",
    dummy_experiment: DummyExperimentName | None = None,
) -> ExperimentRun:
    selected_experiment = experiment_fn
    if selected_experiment is None and dummy_experiment is not None:
        selected_experiment = resolve_dummy_experiment(dummy_experiment)
    if selected_experiment is None:
        selected_experiment = resolve_local_experiment(experiment)
    return ExperimentRunner().run(
        common_config_path=common_config_path or default_common_config_path(),
        experiment_config_path=experiment_config_path,
        experiment_fn=selected_experiment,
        overrides=overrides,
        execution_mode="local",
    )
