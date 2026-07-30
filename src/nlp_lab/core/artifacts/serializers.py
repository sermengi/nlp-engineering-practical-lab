import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import yaml


class ArtifactSerializationError(RuntimeError):
    pass


def write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            delete=False,
        ) as temp_file:
            temp_file.write(content)
            temp_path = Path(temp_file.name)
        os.replace(temp_path, path)
    except Exception as exc:
        if "temp_path" in locals() and temp_path.exists():
            temp_path.unlink()
        msg = f"failed to write artifact atomically: {path}"
        raise ArtifactSerializationError(msg) from exc


def serialize_json(payload: dict[str, Any]) -> str:
    try:
        return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    except (TypeError, ValueError) as exc:
        msg = "failed to serialize JSON artifact"
        raise ArtifactSerializationError(msg) from exc


def serialize_yaml(payload: dict[str, Any]) -> str:
    try:
        return yaml.safe_dump(payload, sort_keys=False)
    except yaml.YAMLError as exc:
        msg = "failed to serialize YAML artifact"
        raise ArtifactSerializationError(msg) from exc


def serialize_jsonl(records: list[dict[str, Any]]) -> str:
    try:
        return "".join(
            json.dumps(record, sort_keys=True, ensure_ascii=True) + "\n" for record in records
        )
    except (TypeError, ValueError) as exc:
        msg = "failed to serialize JSONL artifact"
        raise ArtifactSerializationError(msg) from exc


def write_json(path: Path, payload: dict[str, Any]) -> None:
    write_text_atomic(path, serialize_json(payload))


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    write_text_atomic(path, serialize_yaml(payload))


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    write_text_atomic(path, serialize_jsonl(records))


def append_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("a", encoding="utf-8") as file:
            file.write(serialize_jsonl(records))
    except OSError as exc:
        msg = f"failed to append JSONL artifact: {path}"
        raise ArtifactSerializationError(msg) from exc
