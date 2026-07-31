from pathlib import Path
from typing import Any

try:
    from modal_apps.common import CPU_COUNT, MEMORY_MB, TIMEOUT_SECONDS, app, reload_storage
except ModuleNotFoundError:
    from common import (  # type: ignore[no-redef]
        CPU_COUNT,
        MEMORY_MB,
        TIMEOUT_SECONDS,
        app,
        reload_storage,
    )

from nlp_lab.remote.storage import (
    CACHE_ROOT,
    DATASETS_CACHE_DIR,
    HF_CACHE_DIR,
    REMOTE_CHECKPOINTS_ROOT,
    REMOTE_EXPERIMENTS_ROOT,
    REMOTE_MODELS_ROOT,
    STORAGE_MOUNT_PATH,
    TORCH_CACHE_DIR,
    TRANSFORMERS_CACHE_DIR,
)

REPORT_PATHS = [
    CACHE_ROOT,
    HF_CACHE_DIR,
    DATASETS_CACHE_DIR,
    TRANSFORMERS_CACHE_DIR,
    TORCH_CACHE_DIR,
    REMOTE_EXPERIMENTS_ROOT,
    REMOTE_CHECKPOINTS_ROOT,
    REMOTE_MODELS_ROOT,
]


@app.function(cpu=CPU_COUNT, memory=MEMORY_MB, timeout=TIMEOUT_SECONDS)
def storage_report(path: str = str(STORAGE_MOUNT_PATH)) -> dict[str, Any]:
    reload_storage()
    selected_path = Path(path)
    return {
        "storage_root": str(STORAGE_MOUNT_PATH),
        "selected_path": str(selected_path),
        "selected_entries": list_entries(selected_path),
        "paths": {str(report_path): describe_path(report_path) for report_path in REPORT_PATHS},
    }


def describe_path(path: Path) -> dict[str, Any]:
    return {
        "exists": path.exists(),
        "bytes": directory_size(path) if path.exists() else 0,
        "entries": list_entries(path),
    }


def list_entries(path: Path, limit: int = 50) -> list[str]:
    if not path.exists() or not path.is_dir():
        return []
    return sorted(child.name for child in path.iterdir())[:limit]


def directory_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(child.stat().st_size for child in path.rglob("*") if child.is_file())


@app.local_entrypoint(name="storage")
def main(path: str = str(STORAGE_MOUNT_PATH)) -> None:
    print(storage_report.remote(path=path))
