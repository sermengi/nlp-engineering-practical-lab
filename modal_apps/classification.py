from typing import Any

CPU_SMOKE_CONFIG = "configs/experiments/local_smoke_tiny_sst2.yaml"
GPU_SMOKE_CONFIG = "configs/experiments/modal_smoke_tiny_sst2_gpu.yaml"

try:
    from modal_apps.common import (
        CPU_COUNT,
        GPU_TYPE,
        MEMORY_MB,
        REMOTE_OUTPUT_ROOT,
        TIMEOUT_SECONDS,
        app,
        commit_storage,
        optional_positive_int,
        raise_modal_safe_experiment_error,
    )
except ModuleNotFoundError:
    from common import (  # type: ignore[no-redef]
        CPU_COUNT,
        GPU_TYPE,
        MEMORY_MB,
        REMOTE_OUTPUT_ROOT,
        TIMEOUT_SECONDS,
        app,
        commit_storage,
        optional_positive_int,
        raise_modal_safe_experiment_error,
    )


@app.function(cpu=CPU_COUNT, memory=MEMORY_MB, timeout=TIMEOUT_SECONDS)
def run_classification_cpu(
    config: str = CPU_SMOKE_CONFIG,
    common_config: str = "configs/common/default.yaml",
    output_root: str = REMOTE_OUTPUT_ROOT,
    seed: int | None = None,
    max_samples: int | None = None,
    batch_size: int | None = None,
) -> dict[str, Any]:
    return run_remote_classification(
        config=config,
        common_config=common_config,
        output_root=output_root,
        seed=seed,
        max_samples=max_samples,
        batch_size=batch_size,
    )


@app.function(
    gpu=GPU_TYPE,
    cpu=CPU_COUNT,
    memory=MEMORY_MB,
    timeout=TIMEOUT_SECONDS,
)
def run_classification_gpu(
    config: str = GPU_SMOKE_CONFIG,
    common_config: str = "configs/common/default.yaml",
    output_root: str = REMOTE_OUTPUT_ROOT,
    seed: int | None = None,
    max_samples: int | None = None,
    batch_size: int | None = None,
) -> dict[str, Any]:
    return run_remote_classification(
        config=config,
        common_config=common_config,
        output_root=output_root,
        seed=seed,
        max_samples=max_samples,
        batch_size=batch_size,
    )


def run_remote_classification(
    *,
    config: str,
    common_config: str,
    output_root: str,
    seed: int | None,
    max_samples: int | None,
    batch_size: int | None,
) -> dict[str, Any]:
    from nlp_lab.remote import run_modal_experiment

    try:
        return run_modal_experiment(
            experiment_config_path=config,
            common_config_path=common_config,
            experiment="hf-text-classification",
            output_root=output_root,
            seed=seed,
            max_samples=max_samples,
            batch_size=batch_size,
        )
    except Exception as exc:
        if type(exc).__name__ == "ExperimentRunFailedError":
            raise_modal_safe_experiment_error(exc)
        raise
    finally:
        commit_storage()


@app.local_entrypoint(name="classification")
def main(
    config: str = CPU_SMOKE_CONFIG,
    common_config: str = "configs/common/default.yaml",
    output_root: str = REMOTE_OUTPUT_ROOT,
    seed: int = -1,
    max_samples: int = -1,
    batch_size: int = -1,
    gpu: bool = False,
) -> None:
    run_function = run_classification_gpu if gpu else run_classification_cpu
    selected_config = GPU_SMOKE_CONFIG if gpu and config == CPU_SMOKE_CONFIG else config
    summary = run_function.remote(
        config=selected_config,
        common_config=common_config,
        output_root=output_root,
        seed=optional_positive_int(seed),
        max_samples=optional_positive_int(max_samples),
        batch_size=optional_positive_int(batch_size),
    )
    print(summary)
