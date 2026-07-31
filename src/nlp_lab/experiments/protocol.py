from typing import Protocol

from nlp_lab.core.experiment_result import ExperimentResult
from nlp_lab.core.run_context import RunContext


class ExperimentFn(Protocol):
    def __call__(self, context: RunContext) -> ExperimentResult:
        """Run an experiment using an already initialized run context."""
