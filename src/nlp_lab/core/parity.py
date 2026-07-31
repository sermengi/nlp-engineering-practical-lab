import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field

from nlp_lab.core.artifacts.paths import (
    CONSOLE_LOG_FILENAME,
    ENVIRONMENT_FILENAME,
    ERRORS_FILENAME,
    METRICS_FILENAME,
    PREDICTIONS_FILENAME,
    RESOLVED_CONFIG_FILENAME,
    RUN_METADATA_FILENAME,
    RUNTIME_FILENAME,
)
from nlp_lab.core.artifacts.serializers import write_json
from nlp_lab.core.config.common import StrictConfigModel

ParityStatus = Literal["PASSED", "FAILED"]

REQUIRED_ARTIFACTS = {
    "resolved_config": RESOLVED_CONFIG_FILENAME,
    "run_metadata": RUN_METADATA_FILENAME,
    "environment": ENVIRONMENT_FILENAME,
    "metrics": METRICS_FILENAME,
    "runtime": RUNTIME_FILENAME,
    "predictions": PREDICTIONS_FILENAME,
    "errors": ERRORS_FILENAME,
    "log": CONSOLE_LOG_FILENAME,
}

REQUIRED_PREDICTION_FIELDS = {
    "sample_id",
    "true_label",
    "predicted_label",
    "confidence",
    "is_correct",
}

CONFIG_PARITY_PATHS = (
    ("model",),
    ("dataset",),
    ("preprocessing",),
    ("inference",),
    ("evaluation",),
    ("runtime", "seed"),
)

ENVIRONMENT_DIFF_PATHS = (
    ("execution_mode",),
    ("python", "version"),
    ("pytorch_version",),
    ("cuda", "available"),
    ("cuda", "version"),
    ("cuda", "gpu_name"),
)


class ParityCheck(StrictConfigModel):
    name: str
    status: ParityStatus
    details: dict[str, Any] = Field(default_factory=dict)


class ParityReport(StrictConfigModel):
    status: ParityStatus
    local_run_dir: Path
    remote_run_dir: Path
    checks: list[ParityCheck]
    metric_tolerance: float
    confidence_tolerance: float

    @property
    def passed(self) -> bool:
        return self.status == "PASSED"


def compare_run_parity(
    local_run_dir: str | Path,
    remote_run_dir: str | Path,
    *,
    metric_tolerance: float = 1e-6,
    confidence_tolerance: float = 1e-4,
) -> ParityReport:
    local_dir = Path(local_run_dir)
    remote_dir = Path(remote_run_dir)
    validate_run_directory(local_dir, label="local")
    validate_run_directory(remote_dir, label="remote")
    checks = [
        safe_check(
            "artifact_completeness",
            lambda: compare_artifact_completeness(local_dir, remote_dir),
        ),
        safe_check("resolved_config", lambda: compare_resolved_config(local_dir, remote_dir)),
        safe_check("prediction_schema", lambda: compare_prediction_schema(local_dir, remote_dir)),
        safe_check(
            "predictions",
            lambda: compare_predictions(
                local_dir,
                remote_dir,
                confidence_tolerance=confidence_tolerance,
            ),
        ),
        safe_check(
            "metrics",
            lambda: compare_metrics(local_dir, remote_dir, metric_tolerance=metric_tolerance),
        ),
        safe_check("environment", lambda: compare_environment(local_dir, remote_dir)),
    ]
    status: ParityStatus = (
        "PASSED" if all(check.status == "PASSED" for check in checks) else "FAILED"
    )
    return ParityReport(
        status=status,
        local_run_dir=local_dir,
        remote_run_dir=remote_dir,
        checks=checks,
        metric_tolerance=metric_tolerance,
        confidence_tolerance=confidence_tolerance,
    )


def write_parity_report(path: str | Path, report: ParityReport) -> None:
    write_json(Path(path), report.model_dump(mode="json"))


def safe_check(name: str, check_fn: Callable[[], ParityCheck]) -> ParityCheck:
    try:
        return check_fn()
    except (OSError, ValueError, json.JSONDecodeError, yaml.YAMLError) as exc:
        return ParityCheck(name=name, status="FAILED", details={"error": str(exc)})


def validate_run_directory(run_dir: Path, *, label: str) -> None:
    if not run_dir.is_dir():
        msg = f"{label} run directory does not exist: {run_dir}"
        raise ValueError(msg)


def compare_artifact_completeness(local_dir: Path, remote_dir: Path) -> ParityCheck:
    local_missing = sorted(
        name
        for name, filename in REQUIRED_ARTIFACTS.items()
        if not (local_dir / filename).exists()
    )
    remote_missing = sorted(
        name
        for name, filename in REQUIRED_ARTIFACTS.items()
        if not (remote_dir / filename).exists()
    )
    status: ParityStatus = "PASSED" if not local_missing and not remote_missing else "FAILED"
    return ParityCheck(
        name="artifact_completeness",
        status=status,
        details={
            "required_artifacts": sorted(REQUIRED_ARTIFACTS),
            "local_missing": local_missing,
            "remote_missing": remote_missing,
        },
    )


def compare_resolved_config(local_dir: Path, remote_dir: Path) -> ParityCheck:
    local_config = read_yaml(local_dir / RESOLVED_CONFIG_FILENAME)
    remote_config = read_yaml(remote_dir / RESOLVED_CONFIG_FILENAME)
    local_subset = select_config_parity_payload(local_config)
    remote_subset = select_config_parity_payload(remote_config)
    mismatched_paths = [
        ".".join(path)
        for path in CONFIG_PARITY_PATHS
        if get_nested(local_config, path) != get_nested(remote_config, path)
    ]
    status: ParityStatus = "PASSED" if not mismatched_paths else "FAILED"
    return ParityCheck(
        name="resolved_config",
        status=status,
        details={
            "local_hash": stable_hash(local_subset),
            "remote_hash": stable_hash(remote_subset),
            "mismatched_paths": mismatched_paths,
        },
    )


def compare_prediction_schema(local_dir: Path, remote_dir: Path) -> ParityCheck:
    local_records = read_jsonl(local_dir / PREDICTIONS_FILENAME)
    remote_records = read_jsonl(remote_dir / PREDICTIONS_FILENAME)
    local_fields = sorted({field for record in local_records for field in record})
    remote_fields = sorted({field for record in remote_records for field in record})
    local_missing = sorted(REQUIRED_PREDICTION_FIELDS - set(local_fields))
    remote_missing = sorted(REQUIRED_PREDICTION_FIELDS - set(remote_fields))
    status: ParityStatus = (
        "PASSED"
        if local_fields == remote_fields and not local_missing and not remote_missing
        else "FAILED"
    )
    return ParityCheck(
        name="prediction_schema",
        status=status,
        details={
            "local_fields": local_fields,
            "remote_fields": remote_fields,
            "local_missing_required_fields": local_missing,
            "remote_missing_required_fields": remote_missing,
        },
    )


def compare_predictions(
    local_dir: Path,
    remote_dir: Path,
    *,
    confidence_tolerance: float,
) -> ParityCheck:
    local_records = read_jsonl(local_dir / PREDICTIONS_FILENAME)
    remote_records = read_jsonl(remote_dir / PREDICTIONS_FILENAME)
    local_ids = [str(record.get("sample_id")) for record in local_records]
    remote_ids = [str(record.get("sample_id")) for record in remote_records]
    remote_by_id = {str(record.get("sample_id")): record for record in remote_records}
    label_mismatches: list[str] = []
    correctness_mismatches: list[str] = []
    confidence_mismatches: list[dict[str, Any]] = []

    for local_record in local_records:
        sample_id = str(local_record.get("sample_id"))
        remote_record = remote_by_id.get(sample_id)
        if remote_record is None:
            continue
        if local_record.get("predicted_label") != remote_record.get("predicted_label"):
            label_mismatches.append(sample_id)
        if local_record.get("is_correct") != remote_record.get("is_correct"):
            correctness_mismatches.append(sample_id)
        local_confidence = local_record.get("confidence")
        remote_confidence = remote_record.get("confidence")
        if isinstance(local_confidence, int | float) and isinstance(remote_confidence, int | float):
            difference = abs(float(local_confidence) - float(remote_confidence))
            if difference > confidence_tolerance:
                confidence_mismatches.append(
                    {
                        "sample_id": sample_id,
                        "local_confidence": local_confidence,
                        "remote_confidence": remote_confidence,
                        "difference": difference,
                    }
                )
        elif local_confidence != remote_confidence:
            confidence_mismatches.append(
                {
                    "sample_id": sample_id,
                    "local_confidence": local_confidence,
                    "remote_confidence": remote_confidence,
                    "difference": None,
                }
            )

    status: ParityStatus = (
        "PASSED"
        if local_ids == remote_ids
        and not label_mismatches
        and not correctness_mismatches
        and not confidence_mismatches
        else "FAILED"
    )
    return ParityCheck(
        name="predictions",
        status=status,
        details={
            "sample_count": len(local_records),
            "local_sample_ids": local_ids,
            "remote_sample_ids": remote_ids,
            "sample_order_matches": local_ids == remote_ids,
            "missing_remote_sample_ids": sorted(set(local_ids) - set(remote_ids)),
            "extra_remote_sample_ids": sorted(set(remote_ids) - set(local_ids)),
            "label_mismatches": label_mismatches,
            "correctness_mismatches": correctness_mismatches,
            "confidence_mismatches": confidence_mismatches,
        },
    )


def compare_metrics(local_dir: Path, remote_dir: Path, *, metric_tolerance: float) -> ParityCheck:
    local_metrics = read_json(local_dir / METRICS_FILENAME)
    remote_metrics = read_json(remote_dir / METRICS_FILENAME)
    local_keys = set(local_metrics)
    remote_keys = set(remote_metrics)
    value_mismatches = []
    for metric_name in sorted(local_keys & remote_keys):
        local_value = local_metrics[metric_name]
        remote_value = remote_metrics[metric_name]
        if not isinstance(local_value, int | float) or not isinstance(remote_value, int | float):
            if local_value != remote_value:
                value_mismatches.append(
                    {"metric": metric_name, "local": local_value, "remote": remote_value}
                )
            continue
        difference = abs(float(local_value) - float(remote_value))
        if difference > metric_tolerance:
            value_mismatches.append(
                {
                    "metric": metric_name,
                    "local": local_value,
                    "remote": remote_value,
                    "difference": difference,
                }
            )
    status: ParityStatus = (
        "PASSED"
        if local_keys == remote_keys and not value_mismatches
        else "FAILED"
    )
    return ParityCheck(
        name="metrics",
        status=status,
        details={
            "local_metric_keys": sorted(local_keys),
            "remote_metric_keys": sorted(remote_keys),
            "missing_remote_metrics": sorted(local_keys - remote_keys),
            "extra_remote_metrics": sorted(remote_keys - local_keys),
            "value_mismatches": value_mismatches,
        },
    )


def compare_environment(local_dir: Path, remote_dir: Path) -> ParityCheck:
    local_environment = read_json(local_dir / ENVIRONMENT_FILENAME)
    remote_environment = read_json(remote_dir / ENVIRONMENT_FILENAME)
    differences: dict[str, dict[str, Any]] = {}
    for path in ENVIRONMENT_DIFF_PATHS:
        local_value = get_nested(local_environment, path)
        remote_value = get_nested(remote_environment, path)
        if local_value != remote_value:
            differences[".".join(path)] = {"local": local_value, "remote": remote_value}

    package_differences: dict[str, dict[str, Any]] = {}
    local_packages = local_environment.get("package_versions", {})
    remote_packages = remote_environment.get("package_versions", {})
    if isinstance(local_packages, dict) and isinstance(remote_packages, dict):
        for package_name in sorted(set(local_packages) | set(remote_packages)):
            local_version = local_packages.get(package_name)
            remote_version = remote_packages.get(package_name)
            if local_version != remote_version:
                package_differences[package_name] = {
                    "local": local_version,
                    "remote": remote_version,
                }

    return ParityCheck(
        name="environment",
        status="PASSED",
        details={
            "differences": differences,
            "package_version_differences": package_differences,
        },
    )


def select_config_parity_payload(config: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for path in CONFIG_PARITY_PATHS:
        assign_nested(payload, path, get_nested(config, path))
    return payload


def assign_nested(payload: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    cursor = payload
    for key in path[:-1]:
        cursor = cursor.setdefault(key, {})
    cursor[path[-1]] = value


def get_nested(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    cursor: Any = payload
    for key in path:
        if not isinstance(cursor, dict):
            return None
        cursor = cursor.get(key)
    return cursor


def stable_hash(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:12]


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        msg = f"expected JSON object: {path}"
        raise ValueError(msg)
    return payload


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        payload = yaml.safe_load(file)
    if not isinstance(payload, dict):
        msg = f"expected YAML object: {path}"
        raise ValueError(msg)
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                msg = f"expected JSONL object at {path}:{line_number}"
                raise ValueError(msg)
            records.append(payload)
    return records
