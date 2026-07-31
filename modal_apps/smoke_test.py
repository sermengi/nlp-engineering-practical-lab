import json
from typing import Any

DEFAULT_SMOKE_CONFIG = "configs/experiments/local_smoke_tiny_sst2.yaml"

try:
    from modal_apps.common import (
        CPU_COUNT,
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
        MEMORY_MB,
        REMOTE_OUTPUT_ROOT,
        TIMEOUT_SECONDS,
        app,
        commit_storage,
        optional_positive_int,
        raise_modal_safe_experiment_error,
    )


@app.function(cpu=CPU_COUNT, memory=MEMORY_MB, timeout=TIMEOUT_SECONDS)
def run_dummy_smoke(
    config: str = DEFAULT_SMOKE_CONFIG,
    common_config: str = "configs/common/default.yaml",
    output_root: str = REMOTE_OUTPUT_ROOT,
    seed: int | None = None,
    max_samples: int | None = None,
    batch_size: int | None = None,
) -> dict[str, Any]:
    from nlp_lab.remote import run_modal_experiment

    try:
        return run_modal_experiment(
            experiment_config_path=config,
            common_config_path=common_config,
            experiment="dummy-success",
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


@app.function(cpu=CPU_COUNT, memory=MEMORY_MB, timeout=TIMEOUT_SECONDS)
def run_dummy_failure(
    config: str = DEFAULT_SMOKE_CONFIG,
    common_config: str = "configs/common/default.yaml",
    output_root: str = REMOTE_OUTPUT_ROOT,
    seed: int | None = None,
    max_samples: int | None = None,
    batch_size: int | None = None,
) -> dict[str, Any]:
    from nlp_lab.remote import run_modal_experiment

    try:
        return run_modal_experiment(
            experiment_config_path=config,
            common_config_path=common_config,
            experiment="dummy-failure",
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


@app.local_entrypoint(name="dummy_smoke")
def main(
    config: str = DEFAULT_SMOKE_CONFIG,
    common_config: str = "configs/common/default.yaml",
    output_root: str = REMOTE_OUTPUT_ROOT,
    seed: int = -1,
    max_samples: int = -1,
    batch_size: int = -1,
    fail: bool = False,
) -> None:
    run_function = run_dummy_failure if fail else run_dummy_smoke
    summary = run_function.remote(
        config=config,
        common_config=common_config,
        output_root=output_root,
        seed=optional_positive_int(seed),
        max_samples=optional_positive_int(max_samples),
        batch_size=optional_positive_int(batch_size),
    )
    print(json.dumps(summary, sort_keys=True))
