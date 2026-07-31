from pathlib import Path

STORAGE_VOLUME_NAME = "nlp-lab-storage"
STORAGE_MOUNT_PATH = Path("/storage")

CACHE_ROOT = STORAGE_MOUNT_PATH / "cache"
HF_CACHE_DIR = CACHE_ROOT / "huggingface"
HF_HUB_CACHE_DIR = HF_CACHE_DIR / "hub"
DATASETS_CACHE_DIR = CACHE_ROOT / "datasets"
TRANSFORMERS_CACHE_DIR = CACHE_ROOT / "transformers"
TORCH_CACHE_DIR = CACHE_ROOT / "torch"

REMOTE_EXPERIMENTS_ROOT = STORAGE_MOUNT_PATH / "experiments"
REMOTE_CHECKPOINTS_ROOT = STORAGE_MOUNT_PATH / "checkpoints"
REMOTE_MODELS_ROOT = STORAGE_MOUNT_PATH / "models"


def modal_cache_environment() -> dict[str, str]:
    return {
        "NLP_LAB_STORAGE_ROOT": str(STORAGE_MOUNT_PATH),
        "HF_HOME": str(HF_CACHE_DIR),
        "HF_HUB_CACHE": str(HF_HUB_CACHE_DIR),
        "HUGGINGFACE_HUB_CACHE": str(HF_HUB_CACHE_DIR),
        "HF_DATASETS_CACHE": str(DATASETS_CACHE_DIR),
        "TRANSFORMERS_CACHE": str(TRANSFORMERS_CACHE_DIR),
        "TORCH_HOME": str(TORCH_CACHE_DIR),
    }
