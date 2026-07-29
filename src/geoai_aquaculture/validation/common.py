"""Shared deterministic helpers for the authoritative validation protocol."""

from __future__ import annotations

import ctypes
import gc
import hashlib
import json
import logging
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


class ValidationError(ValueError):
    """Raised when fixed folds, predictions, or diagnostics violate the contract."""


def dataframe_fingerprint(frame: pd.DataFrame, *, columns: Sequence[str] | None = None) -> str:
    """Hash a deterministically ordered table including its schema and missing values."""

    selected = frame.loc[:, list(columns)] if columns is not None else frame
    payload = selected.to_csv(index=False, lineterminator="\n", na_rep="<NA>", float_format="%.17g")
    digest = hashlib.sha256()
    digest.update("\x1f".join(selected.columns.astype(str)).encode())
    digest.update("\x1f".join(map(str, selected.dtypes)).encode())
    digest.update(payload.encode())
    return digest.hexdigest()


def json_fingerprint(value: Mapping[str, Any] | Sequence[Any]) -> str:
    """Return a stable SHA-256 hash for JSON-compatible scientific metadata."""

    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def validate_probabilities(probabilities: np.ndarray, *, name: str = "probabilities") -> None:
    """Reject non-vector, non-finite, or out-of-range probabilities."""

    if probabilities.ndim != 1 or probabilities.size == 0:
        raise ValidationError(f"{name} must be a non-empty one-dimensional array")
    if not np.isfinite(probabilities).all():
        raise ValidationError(f"{name} must be finite")
    if ((probabilities < 0.0) | (probabilities > 1.0)).any():
        raise ValidationError(f"{name} must remain within [0, 1]")


def ensure_directory(path: Path) -> Path:
    """Create an artifact directory and return its resolved path."""

    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def release_temporary_memory() -> None:
    """Collect temporary arrays and return allocator pages where the platform supports it."""

    gc.collect()
    if sys.platform.startswith("linux"):
        try:
            ctypes.CDLL(None).malloc_trim(0)
        except (AttributeError, OSError) as exc:
            LOGGER.debug("allocator trimming is unavailable: %s", exc)
