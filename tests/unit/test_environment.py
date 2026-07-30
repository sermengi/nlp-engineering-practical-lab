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

    environment = collect_environment_info(execution_mode="modal", worker_id="worker-safe-id")
    payload = environment.model_dump(mode="json")

    assert isinstance(environment, EnvironmentInfo)
    assert payload["python"]["version"]
    assert payload["os"]["architecture"]
    assert payload["package_versions"]["pydantic"] is not None
    assert "torch" in payload["package_versions"]
    assert "cuda" in payload
    assert "git" in payload
    assert payload["execution_mode"] == "modal"
    assert payload["worker_id"] == "worker-safe-id"
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
