from pathlib import Path

import pytest

from nlp_lab.remote.storage import (
    CACHE_ROOT,
    DATASETS_CACHE_DIR,
    HF_CACHE_DIR,
    HF_HUB_CACHE_DIR,
    REMOTE_CHECKPOINTS_ROOT,
    REMOTE_EXPERIMENTS_ROOT,
    REMOTE_MODELS_ROOT,
    STORAGE_MOUNT_PATH,
    STORAGE_VOLUME_NAME,
    TORCH_CACHE_DIR,
    TRANSFORMERS_CACHE_DIR,
    modal_cache_environment,
)


@pytest.mark.unit
def test_modal_storage_layout_uses_single_persistent_volume() -> None:
    assert STORAGE_VOLUME_NAME == "nlp-lab-storage"
    assert STORAGE_MOUNT_PATH == Path("/storage")
    assert CACHE_ROOT == Path("/storage/cache")
    assert HF_CACHE_DIR == Path("/storage/cache/huggingface")
    assert DATASETS_CACHE_DIR == Path("/storage/cache/datasets")
    assert TRANSFORMERS_CACHE_DIR == Path("/storage/cache/transformers")
    assert TORCH_CACHE_DIR == Path("/storage/cache/torch")
    assert REMOTE_EXPERIMENTS_ROOT == Path("/storage/experiments")
    assert REMOTE_CHECKPOINTS_ROOT == Path("/storage/checkpoints")
    assert REMOTE_MODELS_ROOT == Path("/storage/models")


@pytest.mark.unit
def test_modal_cache_environment_points_tool_caches_at_storage_volume() -> None:
    cache_env = modal_cache_environment()

    assert cache_env == {
        "NLP_LAB_STORAGE_ROOT": "/storage",
        "HF_HOME": "/storage/cache/huggingface",
        "HF_HUB_CACHE": str(HF_HUB_CACHE_DIR),
        "HUGGINGFACE_HUB_CACHE": str(HF_HUB_CACHE_DIR),
        "HF_DATASETS_CACHE": "/storage/cache/datasets",
        "TRANSFORMERS_CACHE": "/storage/cache/transformers",
        "TORCH_HOME": "/storage/cache/torch",
    }
