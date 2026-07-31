from pathlib import Path

import modal

from nlp_lab.remote.storage import (
    REMOTE_EXPERIMENTS_ROOT,
    STORAGE_MOUNT_PATH,
    STORAGE_VOLUME_NAME,
    modal_cache_environment,
)

APP_NAME = "nlp-lab-remote"
PROJECT_ROOT = Path(__file__).resolve().parents[1]

REMOTE_OUTPUT_ROOT = str(REMOTE_EXPERIMENTS_ROOT)

CPU_COUNT = 2
MEMORY_MB = 4096
TIMEOUT_SECONDS = 900
GPU_TYPE = "T4"

storage_volume = modal.Volume.from_name(STORAGE_VOLUME_NAME, create_if_missing=True)

base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_sync(str(PROJECT_ROOT), groups=["ml", "remote"], frozen=True)
    .env(modal_cache_environment())
    .workdir("/root")
    .add_local_python_source("nlp_lab")
    .add_local_dir(PROJECT_ROOT / "modal_apps", remote_path="/root/modal_apps")
    .add_local_dir(PROJECT_ROOT / "configs", remote_path="/root/configs")
    .add_local_dir(PROJECT_ROOT / "data", remote_path="/root/data")
)

app = modal.App(
    APP_NAME,
    image=base_image,
    volumes={str(STORAGE_MOUNT_PATH): storage_volume},
)


def commit_storage() -> None:
    storage_volume.commit()


def reload_storage() -> None:
    storage_volume.reload()


def optional_positive_int(value: int) -> int | None:
    return None if value < 0 else value


def raise_modal_safe_experiment_error(exc: BaseException) -> None:
    paths = getattr(exc, "paths", None)
    run_dir = getattr(paths, "run_dir", None)
    original = getattr(exc, "original", exc)
    message = f"experiment failed: {original}"
    if run_dir is not None:
        message = f"{message}; artifacts: {run_dir}"
    raise RuntimeError(message) from None
