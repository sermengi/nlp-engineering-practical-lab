import json
from pathlib import Path

import pytest

from nlp_lab.core import observability
from nlp_lab.core.observability import (
    collect_memory_measurements,
    log_run_event,
    redact_sensitive_text,
)


@pytest.mark.unit
def test_redact_sensitive_text_masks_common_secret_shapes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_envsecret123456")

    redacted = redact_sensitive_text(
        "Authorization: Bearer hf_abcd1234abcd1234 "
        "standalone=hf_standalone123456 "
        "env=hf_envsecret123456 "
        "url=https://example.test/model?token=plain-secret&ok=1"
    )

    assert "hf_envsecret123456" not in redacted
    assert "hf_abcd1234abcd1234" not in redacted
    assert "hf_standalone123456" not in redacted
    assert "plain-secret" not in redacted
    assert "[REDACTED]" in redacted


@pytest.mark.unit
def test_log_run_event_writes_redacted_jsonl_to_file_and_console(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    log_path = tmp_path / "console.log"

    log_run_event(
        path=log_path,
        run_id="run-001",
        experiment_name="classification-baseline",
        execution_mode="local",
        stage="inference",
        level="info",
        message="using Authorization: Bearer hf_abcd1234abcd1234",
        extra={"api_key": "visible-secret"},
    )

    file_record = json.loads(log_path.read_text(encoding="utf-8"))
    console_record = json.loads(capsys.readouterr().err)

    assert file_record == console_record
    assert file_record["stage"] == "inference"
    assert file_record["run_id"] == "run-001"
    assert file_record["api_key"] == "[REDACTED]"
    assert "hf_abcd1234abcd1234" not in file_record["message"]


@pytest.mark.unit
def test_memory_collector_handles_missing_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(observability, "import_torch_if_available", lambda: None)

    measurements = collect_memory_measurements()

    assert measurements["memory_device"] == "cpu"
    assert measurements["cuda_peak_allocated_mb"] is None
    assert measurements["cuda_peak_reserved_mb"] is None
    assert "process_peak_memory_mb" in measurements


class FakeCuda:
    def is_available(self) -> bool:
        return True

    def max_memory_allocated(self) -> int:
        return 2 * 1024 * 1024

    def max_memory_reserved(self) -> int:
        return 4 * 1024 * 1024

    def get_device_name(self, index: int) -> str:
        return "Fake GPU"


class FakeTorch:
    cuda = FakeCuda()


@pytest.mark.unit
def test_memory_collector_records_cuda_peak_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(observability, "import_torch_if_available", lambda: FakeTorch())

    measurements = collect_memory_measurements()

    assert measurements["cuda_peak_allocated_mb"] == 2.0
    assert measurements["cuda_peak_reserved_mb"] == 4.0
    assert measurements["memory_device"] == "Fake GPU"
