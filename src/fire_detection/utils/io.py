"""Small atomic file-writing helpers used by command-line tools."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """Write text atomically in the destination directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding=encoding, newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, payload: Any) -> None:
    """Write deterministic, strict JSON atomically."""

    text = json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
    atomic_write_text(path, f"{text}\n")


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    """Write a DataFrame as UTF-8 CSV with stable newlines and no index."""

    atomic_write_text(path, frame.to_csv(index=False, lineterminator="\n"))
