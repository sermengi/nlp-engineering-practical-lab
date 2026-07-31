import json
from pathlib import Path
from typing import Any

import yaml


def create_run_dir(
    run_dir: Path,
    *,
    confidence: float = 0.9,
    predicted_label: str = "positive",
    accuracy: float = 1.0,
    model_revision: str = "main",
    execution_mode: str = "local",
    include_errors: bool = True,
) -> None:
    run_dir.mkdir(parents=True)
    write_yaml(
        run_dir / "config.resolved.yaml",
        {
            "model": {
                "model_id": "tiny-model",
                "revision": model_revision,
                "dtype": "float32",
            },
            "dataset": {
                "local_path": "data/raw/smoke_sst2.csv",
                "split": "test",
                "max_samples": 1,
            },
            "preprocessing": {"max_length": 64, "truncation": True, "padding": "dynamic"},
            "inference": {"batch_size": 2, "threshold": 0.5, "device": "cpu"},
            "evaluation": {"metrics": ["accuracy"], "save_predictions": True},
            "runtime": {"seed": 42, "output_root": "ignored-for-parity"},
        },
    )
    write_json(
        run_dir / "run.json",
        {"run_id": run_dir.name, "status": "COMPLETED", "execution_mode": execution_mode},
    )
    write_json(
        run_dir / "environment.json",
        {
            "execution_mode": execution_mode,
            "python": {"version": "3.11.8", "implementation": "CPython"},
            "package_versions": {"torch": "2.0.0", "transformers": "4.0.0"},
            "pytorch_version": "2.0.0",
            "cuda": {"available": execution_mode == "modal", "version": None, "gpu_name": None},
        },
    )
    write_json(run_dir / "metrics.json", {"accuracy": accuracy})
    write_json(run_dir / "runtime.json", {"batch_size": 2, "sample_count": 1})
    write_jsonl(
        run_dir / "predictions.jsonl",
        [
            {
                "sample_id": "sample-001",
                "text": "Great result.",
                "true_label": "positive",
                "predicted_label": predicted_label,
                "confidence": confidence,
                "is_correct": predicted_label == "positive",
            }
        ],
    )
    if include_errors:
        write_jsonl(run_dir / "errors.jsonl", [])
    (run_dir / "console.log").write_text(
        json.dumps({"run_id": run_dir.name, "message": "done"}) + "\n",
        encoding="utf-8",
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
