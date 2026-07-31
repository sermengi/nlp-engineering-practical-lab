from typing import TYPE_CHECKING

from nlp_lab.core.artifacts.paths import (
    CONSOLE_LOG_FILENAME,
    ENVIRONMENT_FILENAME,
    ERRORS_FILENAME,
    METRICS_FILENAME,
    PREDICTIONS_FILENAME,
    RESOLVED_CONFIG_FILENAME,
    RUN_METADATA_FILENAME,
    RUNTIME_FILENAME,
    SUMMARY_FILENAME,
    RunArtifactPaths,
    build_run_artifact_paths,
)
from nlp_lab.core.artifacts.serializers import (
    ArtifactSerializationError,
    append_jsonl,
    serialize_json,
    serialize_jsonl,
    serialize_yaml,
    write_json,
    write_jsonl,
    write_text_atomic,
    write_yaml,
)

if TYPE_CHECKING:
    from nlp_lab.core.artifacts.writer import (
        ArtifactRunExistsError,
        LocalFilesystemArtifactWriter,
        build_local_artifact_writer,
    )

__all__ = [
    "CONSOLE_LOG_FILENAME",
    "ENVIRONMENT_FILENAME",
    "ERRORS_FILENAME",
    "METRICS_FILENAME",
    "PREDICTIONS_FILENAME",
    "RESOLVED_CONFIG_FILENAME",
    "RUN_METADATA_FILENAME",
    "RUNTIME_FILENAME",
    "SUMMARY_FILENAME",
    "ArtifactSerializationError",
    "ArtifactRunExistsError",
    "LocalFilesystemArtifactWriter",
    "RunArtifactPaths",
    "append_jsonl",
    "build_run_artifact_paths",
    "build_local_artifact_writer",
    "serialize_json",
    "serialize_jsonl",
    "serialize_yaml",
    "write_json",
    "write_jsonl",
    "write_text_atomic",
    "write_yaml",
]


def __getattr__(name: str) -> object:
    if name in {
        "ArtifactRunExistsError",
        "LocalFilesystemArtifactWriter",
        "build_local_artifact_writer",
    }:
        from nlp_lab.core.artifacts.writer import (
            ArtifactRunExistsError,
            LocalFilesystemArtifactWriter,
            build_local_artifact_writer,
        )

        exports = {
            "ArtifactRunExistsError": ArtifactRunExistsError,
            "LocalFilesystemArtifactWriter": LocalFilesystemArtifactWriter,
            "build_local_artifact_writer": build_local_artifact_writer,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
