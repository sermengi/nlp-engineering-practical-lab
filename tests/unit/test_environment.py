import os

import pytest

from nlp_lab.core.environment import (
    KEY_PACKAGE_NAMES,
    EnvironmentInfo,
    collect_environment_info,
    collect_key_package_versions,
)


@pytest.mark.unit
def test_collect_environment_info_returns_safe_reproducibility_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "should-not-be-captured")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "should-not-be-captured")
    monkeypatch.setenv("MODAL_TASK_ID", "ta-safe-worker")
    monkeypatch.setenv("MODAL_CLOUD_PROVIDER", "aws")
    monkeypatch.setenv("MODAL_IMAGE_ID", "im-safe-image")
    monkeypatch.setenv("MODAL_REGION", "us-east-1")
    monkeypatch.setenv("MODAL_ENVIRONMENT", "main")
    monkeypatch.setenv("MODAL_IS_REMOTE", "1")
    monkeypatch.setenv("NLP_LAB_STORAGE_ROOT", "/storage")
    monkeypatch.setenv("HF_HOME", "/storage/cache/huggingface")
    monkeypatch.setenv("HF_HUB_CACHE", "/storage/cache/huggingface/hub")
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", "/storage/cache/huggingface/hub")
    monkeypatch.setenv("HF_DATASETS_CACHE", "/storage/cache/datasets")
    monkeypatch.setenv("TRANSFORMERS_CACHE", "/storage/cache/transformers")
    monkeypatch.setenv("TORCH_HOME", "/storage/cache/torch")

    environment = collect_environment_info(execution_mode="modal")
    payload = environment.model_dump(mode="json")

    assert isinstance(environment, EnvironmentInfo)
    assert payload["python"]["version"]
    assert payload["os"]["architecture"]
    assert payload["package_versions"]["pydantic"] is not None
    assert "torch" in payload["package_versions"]
    assert "cuda" in payload
    assert "git" in payload
    assert payload["execution_mode"] == "modal"
    assert payload["worker_id"] == "ta-safe-worker"
    assert payload["modal"] == {
        "task_id": "ta-safe-worker",
        "cloud_provider": "aws",
        "image_id": "im-safe-image",
        "region": "us-east-1",
        "environment_name": "main",
        "is_remote": True,
        "storage_root": "/storage",
        "hf_home": "/storage/cache/huggingface",
        "hf_hub_cache": "/storage/cache/huggingface/hub",
        "huggingface_hub_cache": "/storage/cache/huggingface/hub",
        "hf_datasets_cache": "/storage/cache/datasets",
        "transformers_cache": "/storage/cache/transformers",
        "torch_home": "/storage/cache/torch",
    }
    assert "environment_variables" not in payload
    assert "pip_freeze" not in payload
    assert "username" not in payload
    assert "hostname" not in payload
    assert "HF_TOKEN" not in str(payload)
    assert "MODAL_TOKEN_SECRET" not in str(payload)
    assert os.environ["HF_TOKEN"] == "should-not-be-captured"


@pytest.mark.unit
def test_collect_key_package_versions_is_limited_to_main_packages() -> None:
    versions = collect_key_package_versions()

    assert set(versions) == set(KEY_PACKAGE_NAMES)
    assert "pip" not in versions
