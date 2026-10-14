"""Versioned JSONL datasets for Confidence Lab."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path

from pydantic import ValidationError

from action_sentinel.contracts import DatasetExample


class DatasetError(ValueError):
    pass


def load_dataset(path: str | Path, splits: Iterable[str] | None = None) -> tuple[list[DatasetExample], str]:
    """Load and validate every line. Returns (examples, sha256 of the file). Mixed versions are rejected."""
    p = Path(path)
    if not p.is_file():
        raise DatasetError("dataset not found: %s" % p)
    data = p.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    examples: list[DatasetExample] = []
    seen: set[str] = set()
    for lineno, line in enumerate(data.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            ex = DatasetExample.model_validate_json(line)
        except ValidationError as exc:
            raise DatasetError("%s:%d: %s" % (p.name, lineno, exc.errors()[0]["msg"])) from exc
        if ex.example_id in seen:
            raise DatasetError("%s:%d: duplicate example_id %s" % (p.name, lineno, ex.example_id))
        seen.add(ex.example_id)
        examples.append(ex)
    versions = {e.dataset_version for e in examples}
    if len(versions) > 1:
        raise DatasetError("dataset mixes versions %s" % sorted(versions))
    if splits is not None:
        wanted = set(splits)
        examples = [e for e in examples if e.split in wanted]
    return examples, digest
