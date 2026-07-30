import random
from types import SimpleNamespace

import pytest

from nlp_lab.core.seed import set_global_seed


@pytest.mark.unit
def test_set_global_seed_seeds_python_random() -> None:
    set_global_seed(42)
    first = random.random()

    set_global_seed(42)
    second = random.random()

    assert first == second


@pytest.mark.unit
def test_set_global_seed_keeps_seed_and_determinism_separate() -> None:
    result = set_global_seed(42, deterministic=False)

    assert result.seed == 42
    assert result.deterministic is False
    assert result.deterministic_algorithms_enabled is False
    assert result.python_random_seeded is True
    assert "does not guarantee identical results" in result.notes[0]


@pytest.mark.unit
def test_set_global_seed_rejects_negative_seed() -> None:
    with pytest.raises(ValueError, match="seed must be non-negative"):
        set_global_seed(-1)


@pytest.mark.unit
def test_set_global_seed_controls_numpy_and_torch_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, int | bool]] = []

    fake_numpy = SimpleNamespace(
        random=SimpleNamespace(seed=lambda seed: calls.append(("numpy.seed", seed)))
    )
    fake_torch = SimpleNamespace(
        manual_seed=lambda seed: calls.append(("torch.manual_seed", seed)),
        cuda=SimpleNamespace(
            is_available=lambda: True,
            manual_seed_all=lambda seed: calls.append(("torch.cuda.manual_seed_all", seed)),
        ),
        use_deterministic_algorithms=lambda enabled: calls.append(
            ("torch.use_deterministic_algorithms", enabled)
        ),
        backends=SimpleNamespace(
            cudnn=SimpleNamespace(deterministic=False, benchmark=True),
        ),
    )

    def fake_optional_import(module_name: str) -> object | None:
        return {"numpy": fake_numpy, "torch": fake_torch}.get(module_name)

    monkeypatch.setattr("nlp_lab.core.seed.optional_import", fake_optional_import)

    result = set_global_seed(7, deterministic=True)

    assert ("numpy.seed", 7) in calls
    assert ("torch.manual_seed", 7) in calls
    assert ("torch.cuda.manual_seed_all", 7) in calls
    assert ("torch.use_deterministic_algorithms", True) in calls
    assert fake_torch.backends.cudnn.deterministic is True
    assert fake_torch.backends.cudnn.benchmark is False
    assert result.numpy_seeded is True
    assert result.torch_cpu_seeded is True
    assert result.torch_cuda_seeded is True
    assert result.deterministic_algorithms_enabled is True
