import platform
import socket
import subprocess
import sys
from importlib import metadata
from typing import Literal

from nlp_lab.core.config.common import StrictConfigModel

ExecutionMode = Literal["local", "modal", "ci"]
KEY_PACKAGE_NAMES = [
    "accelerate",
    "datasets",
    "evaluate",
    "numpy",
    "pandas",
    "pydantic",
    "pyyaml",
    "scikit-learn",
    "torch",
    "transformers",
]


class PythonEnvironmentInfo(StrictConfigModel):
    version: str
    implementation: str


class OperatingSystemInfo(StrictConfigModel):
    name: str
    release: str
    platform: str
    architecture: str


class CudaInfo(StrictConfigModel):
    available: bool
    version: str | None = None
    gpu_name: str | None = None


class GitMetadata(StrictConfigModel):
    commit: str | None
    dirty: bool


class EnvironmentInfo(StrictConfigModel):
    python: PythonEnvironmentInfo
    os: OperatingSystemInfo
    package_versions: dict[str, str | None]
    pytorch_version: str | None
    cuda: CudaInfo
    git: GitMetadata
    execution_mode: ExecutionMode
    worker_id: str | None = None


def collect_environment_info(
    execution_mode: ExecutionMode = "local",
    worker_id: str | None = None,
) -> EnvironmentInfo:
    cuda = collect_cuda_info()
    return EnvironmentInfo(
        python=PythonEnvironmentInfo(
            version=sys.version,
            implementation=platform.python_implementation(),
        ),
        os=OperatingSystemInfo(
            name=platform.system(),
            release=platform.release(),
            platform=platform.platform(),
            architecture=platform.machine(),
        ),
        package_versions=collect_key_package_versions(),
        pytorch_version=package_version("torch"),
        cuda=cuda,
        git=collect_git_metadata(),
        execution_mode=execution_mode,
        worker_id=worker_id,
    )


def collect_key_package_versions(package_names: list[str] | None = None) -> dict[str, str | None]:
    return {
        package_name: package_version(package_name)
        for package_name in package_names or KEY_PACKAGE_NAMES
    }


def package_version(package_name: str) -> str | None:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return None


def collect_cuda_info() -> CudaInfo:
    try:
        import torch
    except ImportError:
        return CudaInfo(available=False)

    cuda_available = bool(torch.cuda.is_available())
    return CudaInfo(
        available=cuda_available,
        version=torch.version.cuda,
        gpu_name=torch.cuda.get_device_name(0) if cuda_available else None,
    )


def collect_git_metadata() -> GitMetadata:
    return GitMetadata(
        commit=git_output(["git", "rev-parse", "HEAD"]),
        dirty=bool(git_output(["git", "status", "--porcelain"])),
    )


def collect_hostname_worker_id() -> str:
    return socket.gethostname()


def git_output(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    output = completed.stdout.strip()
    return output or None
