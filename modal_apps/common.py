from pathlib import Path

import modal

APP_NAME = "nlp-lab-remote"
PROJECT_ROOT = Path(__file__).resolve().parents[1]

ARTIFACT_VOLUME_NAME = "nlp-lab-artifacts"
ARTIFACT_MOUNT_PATH = "/artifacts"
REMOTE_OUTPUT_ROOT = f"{ARTIFACT_MOUNT_PATH}/experiments"
HF_CACHE_DIR = f"{ARTIFACT_MOUNT_PATH}/.cache/huggingface"

CPU_COUNT = 2
MEMORY_MB = 4096
TIMEOUT_SECONDS = 900
GPU_TYPE = "T4"

artifact_volume = modal.Volume.from_name(ARTIFACT_VOLUME_NAME, create_if_missing=True)

base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_sync(str(PROJECT_ROOT), groups=["ml", "remote"], frozen=True)
    .env(
        {
            "HF_HOME": HF_CACHE_DIR,
            "TRANSFORMERS_CACHE": HF_CACHE_DIR,
        }
    )
    .workdir("/root")
    .add_local_python_source("nlp_lab")
    .add_local_dir(PROJECT_ROOT / "modal_apps", remote_path="/root/modal_apps")
    .add_local_dir(PROJECT_ROOT / "configs", remote_path="/root/configs")
    .add_local_dir(PROJECT_ROOT / "data", remote_path="/root/data")
)

app = modal.App(
    APP_NAME,
    image=base_image,
    volumes={ARTIFACT_MOUNT_PATH: artifact_volume},
)


def commit_artifacts() -> None:
    artifact_volume.commit()


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
