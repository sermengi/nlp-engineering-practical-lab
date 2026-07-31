import importlib
from dataclasses import dataclass
from typing import Any

from nlp_lab.core.config import ExperimentConfig


@dataclass(frozen=True)
class LoadedSequenceClassifier:
    tokenizer: Any
    model: Any
    device: str
    dtype: str | None


def load_sequence_classifier(config: ExperimentConfig) -> LoadedSequenceClassifier:
    transformers = import_optional_dependency("transformers")
    torch = import_optional_dependency("torch")

    device = select_device(config.inference.device, torch)
    dtype = select_torch_dtype(config.model.dtype, torch)
    cache_dir = config.cache.huggingface or config.cache.models
    model_kwargs: dict[str, Any] = {
        "revision": config.model.revision,
        "trust_remote_code": config.model.trust_remote_code,
        "cache_dir": cache_dir,
    }
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.model.model_id,
        revision=config.model.revision,
        trust_remote_code=config.model.trust_remote_code,
        cache_dir=cache_dir,
    )
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        config.model.model_id,
        **model_kwargs,
    )
    model.to(device)
    model.eval()
    return LoadedSequenceClassifier(
        tokenizer=tokenizer,
        model=model,
        device=device,
        dtype=str(dtype).replace("torch.", "") if dtype is not None else None,
    )


def select_device(device_preference: str, torch: Any) -> str:
    if device_preference == "cpu":
        return "cpu"
    if device_preference == "cuda":
        if not bool(torch.cuda.is_available()):
            msg = "CUDA device requested but torch.cuda is unavailable"
            raise RuntimeError(msg)
        return "cuda"
    if device_preference == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is None or not bool(mps_backend.is_available()):
            msg = "MPS device requested but torch MPS backend is unavailable"
            raise RuntimeError(msg)
        return "mps"
    if device_preference == "auto":
        if bool(torch.cuda.is_available()):
            return "cuda"
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and bool(mps_backend.is_available()):
            return "mps"
        return "cpu"
    msg = f"unsupported device preference: {device_preference}"
    raise ValueError(msg)


def select_torch_dtype(dtype: str | None, torch: Any) -> Any | None:
    if dtype is None or dtype == "auto":
        return None
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    try:
        return dtype_map[dtype]
    except KeyError as exc:
        msg = f"unsupported torch dtype: {dtype}"
        raise ValueError(msg) from exc


def import_optional_dependency(module_name: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        msg = (
            f"missing optional dependency {module_name!r}; "
            "run with the project's ml dependency group enabled"
        )
        raise RuntimeError(msg) from exc
