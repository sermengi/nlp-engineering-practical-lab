import hashlib
import json
import re
from datetime import datetime
from typing import Any

from nlp_lab.core.config.experiment import ExperimentConfig

RUN_ID_TIMESTAMP_FORMAT = "%Y%m%d-%H%M%S"


def slugify_run_component(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", value.strip().lower())
    slug = re.sub(r"-+", "-", slug).strip("-")
    if not slug:
        msg = "run ID component must not be empty"
        raise ValueError(msg)
    return slug


def config_hash_payload(config: ExperimentConfig) -> dict[str, Any]:
    return {
        "model": config.model.model_dump(mode="json"),
        "dataset": config.dataset.model_dump(mode="json"),
        "preprocessing": config.preprocessing.model_dump(mode="json"),
        "inference": config.inference.model_dump(mode="json"),
        "evaluation": config.evaluation.model_dump(mode="json"),
        "seed": config.runtime.seed,
    }


def compute_config_hash(config: ExperimentConfig, length: int = 8) -> str:
    if length <= 0:
        msg = "hash length must be positive"
        raise ValueError(msg)

    payload = config_hash_payload(config)
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:length]


def generate_run_id(config: ExperimentConfig, timestamp: datetime | None = None) -> str:
    run_timestamp = timestamp or datetime.now()
    timestamp_part = run_timestamp.strftime(RUN_ID_TIMESTAMP_FORMAT)
    experiment_part = slugify_run_component(config.experiment.name)
    hash_part = compute_config_hash(config)
    return f"{timestamp_part}_{experiment_part}_{hash_part}"
