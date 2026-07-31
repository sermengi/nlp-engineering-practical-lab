"""Experiment orchestration."""

from nlp_lab.experiments.dummy import failing_dummy_experiment, successful_dummy_experiment
from nlp_lab.experiments.local import run_local_experiment
from nlp_lab.experiments.protocol import ExperimentFn
from nlp_lab.experiments.runner import (
    CONFIG_VALIDATION_EXIT_CODE,
    EXPERIMENT_FAILURE_EXIT_CODE,
    SUCCESS_EXIT_CODE,
    ExperimentRun,
    ExperimentRunFailedError,
    ExperimentRunner,
)

__all__ = [
    "CONFIG_VALIDATION_EXIT_CODE",
    "EXPERIMENT_FAILURE_EXIT_CODE",
    "SUCCESS_EXIT_CODE",
    "ExperimentFn",
    "ExperimentRun",
    "ExperimentRunFailedError",
    "ExperimentRunner",
    "failing_dummy_experiment",
    "run_local_experiment",
    "successful_dummy_experiment",
]
