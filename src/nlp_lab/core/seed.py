import importlib
import random
from typing import Any

from pydantic import Field

from nlp_lab.core.config.common import StrictConfigModel


class SeedSetupResult(StrictConfigModel):
    seed: int = Field(..., ge=0)
    deterministic: bool
    python_random_seeded: bool
    numpy_seeded: bool
    torch_cpu_seeded: bool
    torch_cuda_seeded: bool
    deterministic_algorithms_enabled: bool
    notes: list[str] = Field(default_factory=list)


def set_global_seed(seed: int, deterministic: bool = False) -> SeedSetupResult:
    if seed < 0:
        msg = "seed must be non-negative"
        raise ValueError(msg)

    notes = [
        "Using the same seed does not guarantee identical results across all hardware or kernels.",
    ]
    random.seed(seed)

    numpy_seeded = seed_numpy(seed)
    torch_result = seed_torch(seed, deterministic)
    if deterministic:
        notes.append(
            "Deterministic algorithms can reduce performance and may raise errors "
            "for unsupported operations."
        )

    return SeedSetupResult(
        seed=seed,
        deterministic=deterministic,
        python_random_seeded=True,
        numpy_seeded=numpy_seeded,
        torch_cpu_seeded=torch_result["cpu_seeded"],
        torch_cuda_seeded=torch_result["cuda_seeded"],
        deterministic_algorithms_enabled=torch_result["deterministic_algorithms_enabled"],
        notes=notes,
    )


def seed_numpy(seed: int) -> bool:
    numpy = optional_import("numpy")
    if numpy is None:
        return False
    numpy.random.seed(seed)
    return True


def seed_torch(seed: int, deterministic: bool) -> dict[str, bool]:
    torch = optional_import("torch")
    if torch is None:
        return {
            "cpu_seeded": False,
            "cuda_seeded": False,
            "deterministic_algorithms_enabled": False,
        }

    torch.manual_seed(seed)
    cuda_seeded = False
    if bool(torch.cuda.is_available()):
        torch.cuda.manual_seed_all(seed)
        cuda_seeded = True

    deterministic_enabled = False
    if deterministic:
        torch.use_deterministic_algorithms(True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        deterministic_enabled = True

    return {
        "cpu_seeded": True,
        "cuda_seeded": cuda_seeded,
        "deterministic_algorithms_enabled": deterministic_enabled,
    }


def optional_import(module_name: str) -> Any | None:
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return None
