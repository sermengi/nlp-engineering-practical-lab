import json
import os
import re
import resource
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter
from typing import TYPE_CHECKING, Any

from nlp_lab.core.artifacts.serializers import append_jsonl

if TYPE_CHECKING:
    from pathlib import Path


SECRET_KEY_MARKERS = (
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "API_KEY",
    "APIKEY",
    "AUTHORIZATION",
    "CREDENTIAL",
)
SECRET_PATTERNS = (
    re.compile(r"(?i)(authorization\s*[:=]\s*)(bearer\s+)?[^\s,;]+"),
    re.compile(r"(?i)\b(bearer)\s+[a-z0-9._\-]+"),
    re.compile(r"\bhf_[A-Za-z0-9_]{12,}\b"),
    re.compile(r"(?i)([?&](?:token|access_token|api_key|signature|secret)=)[^&\s]+"),
)


class StageTimer:
    def __init__(self) -> None:
        self._total_started_at = perf_counter()
        self.stage_seconds: dict[str, float] = {}

    @contextmanager
    def measure(self, stage: str) -> Iterator[None]:
        started_at = perf_counter()
        try:
            yield
        finally:
            self.record(stage, perf_counter() - started_at)

    def record(self, stage: str, duration_seconds: float) -> None:
        self.stage_seconds[stage] = self.stage_seconds.get(stage, 0.0) + duration_seconds

    @property
    def total_duration_seconds(self) -> float:
        return perf_counter() - self._total_started_at


class StageExecutionError(RuntimeError):
    def __init__(
        self,
        stage: str,
        category: str,
        original: BaseException,
    ) -> None:
        super().__init__(str(original) or type(original).__name__)
        self.stage = stage
        self.category = category
        self.original = original


def categorize_exception(exception: BaseException, fallback_stage: str) -> tuple[str, str]:
    stage = getattr(exception, "stage", fallback_stage)
    category = getattr(exception, "category", None)
    if isinstance(category, str):
        return stage, category
    if stage == "config_loading":
        return stage, "configuration_error"
    if stage in {"data_loading", "preprocessing"}:
        return stage, "data_error"
    if stage == "model_loading":
        return stage, "model_loading_error"
    if stage == "inference":
        return stage, "inference_error"
    if stage == "artifact_writing":
        return stage, "artifact_writing_error"
    if stage == "remote_execution":
        return stage, "remote_execution_error"
    return stage, "experiment_error"


def wrap_stage_error(stage: str, category: str, exception: BaseException) -> StageExecutionError:
    if isinstance(exception, StageExecutionError):
        return exception
    return StageExecutionError(stage, category, exception)


def redact_sensitive_text(text: str) -> str:
    redacted = text
    for key, value in os.environ.items():
        if not value or len(value) < 4:
            continue
        if any(marker in key.upper() for marker in SECRET_KEY_MARKERS):
            redacted = redacted.replace(value, "[REDACTED]")
    for pattern in SECRET_PATTERNS:
        redacted = pattern.sub(redact_secret_match, redacted)
    return redacted


def redact_secret_match(match: re.Match[str]) -> str:
    prefix = match.group(1) if match.lastindex else ""
    return f"{prefix}[REDACTED]"


def redacted_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: redact_value(key, value) for key, value in payload.items()}


def redact_value(key: str, value: Any) -> Any:
    if value is None:
        return None
    if any(marker in key.upper() for marker in SECRET_KEY_MARKERS):
        return "[REDACTED]"
    if isinstance(value, str):
        return redact_sensitive_text(value)
    if isinstance(value, dict):
        return {
            nested_key: redact_value(str(nested_key), nested_value)
            for nested_key, nested_value in value.items()
        }
    if isinstance(value, list):
        return [redact_value(key, item) for item in value]
    return value


def log_run_event(
    *,
    path: "Path | None",
    run_id: str,
    experiment_name: str,
    execution_mode: str,
    stage: str,
    level: str,
    message: str,
    extra: dict[str, Any] | None = None,
    emit_console: bool = True,
    save_file: bool = True,
) -> None:
    payload = redacted_payload(
        {
            "run_id": run_id,
            "experiment_name": experiment_name,
            "execution_mode": execution_mode,
            "stage": stage,
            "level": level.upper(),
            "message": message,
            **(extra or {}),
        }
    )
    line = json.dumps(payload, sort_keys=True, default=str)
    if emit_console:
        print(line, file=sys.stderr)
    if save_file and path is not None:
        append_jsonl(path, [payload])


def reset_cuda_peak_memory() -> None:
    torch = import_torch_if_available()
    if torch is None:
        return
    try:
        if bool(torch.cuda.is_available()):
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        return


def collect_memory_measurements() -> dict[str, float | str | None]:
    measurements: dict[str, float | str | None] = {
        "process_peak_memory_mb": process_peak_memory_mb(),
        "process_current_memory_mb": process_current_memory_mb(),
        "cuda_peak_allocated_mb": None,
        "cuda_peak_reserved_mb": None,
        "memory_device": "cpu",
    }
    torch = import_torch_if_available()
    if torch is None:
        return measurements
    try:
        if bool(torch.cuda.is_available()):
            measurements["cuda_peak_allocated_mb"] = bytes_to_mb(torch.cuda.max_memory_allocated())
            measurements["cuda_peak_reserved_mb"] = bytes_to_mb(torch.cuda.max_memory_reserved())
            measurements["memory_device"] = torch.cuda.get_device_name(0)
    except Exception:
        return measurements
    return measurements


def process_peak_memory_mb() -> float | None:
    try:
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except Exception:
        return None
    if sys.platform == "darwin":
        return bytes_to_mb(peak)
    return peak / 1024


def process_current_memory_mb() -> float | None:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        with open("/proc/self/statm", encoding="utf-8") as file:
            resident_pages = int(file.read().split()[1])
    except Exception:
        return None
    return bytes_to_mb(resident_pages * page_size)


def bytes_to_mb(value: int | float) -> float:
    return float(value) / (1024 * 1024)


def import_torch_if_available() -> Any | None:
    try:
        import torch
    except Exception:
        return None
    return torch
