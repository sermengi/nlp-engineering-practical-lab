from time import perf_counter

from nlp_lab.core.experiment_result import ExperimentResult
from nlp_lab.core.run_artifacts import PredictionRecord, RuntimeMeasurements
from nlp_lab.core.run_context import RunContext


def successful_dummy_experiment(context: RunContext) -> ExperimentResult:
    started = perf_counter()
    sample_count = context.config.dataset.max_samples or 3
    bounded_sample_count = min(sample_count, 3)
    predictions = [
        PredictionRecord(
            sample_id=f"dummy-{index + 1:03d}",
            text=f"Dummy input {index + 1}",
            true_label=index % 2,
            predicted_label=index % 2,
            confidence=0.9 - (index * 0.1),
            is_correct=True,
        )
        for index in range(bounded_sample_count)
    ]
    duration = perf_counter() - started
    return ExperimentResult(
        metrics={
            "accuracy": 1.0,
            "macro_precision": 1.0,
            "macro_recall": 1.0,
            "macro_f1": 1.0,
        },
        runtime=RuntimeMeasurements(
            total_duration_seconds=duration,
            inference_seconds=duration,
            samples_per_second=bounded_sample_count / duration if duration > 0 else None,
            batch_size=context.config.inference.batch_size,
        ),
        predictions=predictions if context.config.evaluation.save_predictions else [],
        notes=["Dummy experiment completed without loading external models."],
    )


def failing_dummy_experiment(context: RunContext) -> ExperimentResult:
    raise RuntimeError(f"intentional dummy failure for run {context.run_id}")
