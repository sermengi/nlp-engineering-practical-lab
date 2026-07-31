import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from nlp_lab.core.config import ConfigOverrides
from nlp_lab.experiments.local import resolve_local_experiment
from nlp_lab.experiments.runner import ExperimentRunner


@pytest.mark.integration
def test_local_hf_smoke_writes_expected_artifacts_for_two_runs(tmp_path: Path) -> None:
    runner = ExperimentRunner()
    experiment_fn = resolve_local_experiment("hf-text-classification")
    started_at = datetime(2026, 7, 31, 12, 0, tzinfo=UTC)

    runs = [
        runner.run(
            common_config_path="configs/common/default.yaml",
            experiment_config_path="configs/experiments/local_smoke_tiny_sst2.yaml",
            experiment_fn=experiment_fn,
            overrides=ConfigOverrides(output_root=tmp_path),
            started_at=started_at + timedelta(seconds=index),
        )
        for index in range(2)
    ]

    assert runs[0].paths.run_dir != runs[1].paths.run_dir

    for run in runs:
        assert run.metadata.status == "COMPLETED"
        assert run.paths.resolved_config.exists()
        assert run.paths.environment.exists()

        run_metadata = json.loads(run.paths.run_metadata.read_text(encoding="utf-8"))
        assert run_metadata["status"] == "COMPLETED"

        predictions = read_jsonl(run.paths.predictions)
        assert len(predictions) == 4
        assert [prediction["sample_id"] for prediction in predictions] == [
            "smoke-001",
            "smoke-002",
            "smoke-003",
            "smoke-004",
        ]
        assert all("predicted_label" in prediction for prediction in predictions)
        assert all("confidence" in prediction for prediction in predictions)

        errors = read_jsonl(run.paths.errors)
        assert errors
        incorrect_prediction_ids = {
            prediction["sample_id"]
            for prediction in predictions
            if prediction["is_correct"] is False
        }
        assert {error["sample_id"] for error in errors} == incorrect_prediction_ids

        metrics = json.loads(run.paths.metrics.read_text(encoding="utf-8"))
        assert set(metrics) >= {
            "accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "weighted_f1",
            "class_support_negative",
            "class_support_positive",
        }

        runtime = json.loads(run.paths.runtime.read_text(encoding="utf-8"))
        assert runtime["model_load_seconds"] is not None
        assert runtime["inference_seconds"] is not None
        assert runtime["model_load_seconds"] >= 0
        assert runtime["inference_seconds"] >= 0

        model_metadata = json.loads((run.paths.run_dir / "model_metadata.json").read_text())
        assert (
            model_metadata["model_id"]
            == "peft-internal-testing/tiny-random-BertForSequenceClassification"
        )
        assert model_metadata["device"] == "cpu"

        dataset_metadata = json.loads((run.paths.run_dir / "dataset_metadata.json").read_text())
        assert dataset_metadata["local_path"] == "data/raw/smoke_sst2.csv"
        assert dataset_metadata["sample_count"] == 4


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
