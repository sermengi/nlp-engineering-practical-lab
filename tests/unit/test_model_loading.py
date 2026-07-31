from types import SimpleNamespace

import pytest

from nlp_lab.models.loading import select_device, select_torch_dtype


class FakeCuda:
    @staticmethod
    def is_available() -> bool:
        return False


class FakeMps:
    @staticmethod
    def is_available() -> bool:
        return False


@pytest.mark.unit
def test_select_device_uses_cpu_for_auto_when_accelerators_are_unavailable() -> None:
    fake_torch = SimpleNamespace(cuda=FakeCuda(), backends=SimpleNamespace(mps=FakeMps()))

    assert select_device("auto", fake_torch) == "cpu"


@pytest.mark.unit
def test_select_device_rejects_unavailable_cuda_request() -> None:
    fake_torch = SimpleNamespace(cuda=FakeCuda(), backends=SimpleNamespace(mps=FakeMps()))

    with pytest.raises(RuntimeError, match="CUDA device requested"):
        select_device("cuda", fake_torch)


@pytest.mark.unit
def test_select_torch_dtype_maps_supported_dtype_names() -> None:
    fake_torch = SimpleNamespace(float32="float32", float16="float16", bfloat16="bfloat16")

    assert select_torch_dtype("float32", fake_torch) == "float32"
    assert select_torch_dtype("auto", fake_torch) is None
