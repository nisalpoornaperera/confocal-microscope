"""On-disk model registry: ``<models_dir>/<name>/model.joblib`` + ``metadata.json``.

``metadata.json`` is an :class:`~confocal.models.ml.MLModelInfo`; it is read
without unpickling anything, so models can be listed and checked for
compatibility cheaply and safely. ``model.joblib`` is the fitted scikit-learn
estimator.

SECURITY: joblib files are Python pickles -- loading one can execute
arbitrary code. Only deploy model files you trained yourself or obtained from
a trusted source, and keep ``models_dir`` writable only by the operator.
Model names are restricted to a safe character set (:data:`MODEL_NAME_PATTERN`)
so a name can never escape ``models_dir``.

A model is *compatible* when its ``feature_names`` equal
:data:`confocal.ml.features.FEATURE_NAMES` exactly (same features, same
order) and its task is supported by inference: a model trained on another
feature definition would silently receive the wrong columns.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
from pydantic import ValidationError

from confocal.errors import ModelNotAvailableError
from confocal.ml.features import FEATURE_NAMES
from confocal.models.ml import MLModelInfo, MLTask

logger = logging.getLogger(__name__)

MODEL_FILE = "model.joblib"
METADATA_FILE = "metadata.json"

#: Allowed model names: a letter or digit, then letters, digits, ``_ . -``.
MODEL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")

#: Tasks the inference service can run.
SUPPORTED_TASKS: frozenset[MLTask] = frozenset({MLTask.BAD_POINT, MLTask.CONFIDENCE})


def is_valid_model_name(name: str) -> bool:
    return MODEL_NAME_PATTERN.fullmatch(name) is not None and ".." not in name


def compatibility_problem(info: MLModelInfo) -> str | None:
    """Why the model cannot be used by this software version (None if it can)."""
    if info.task not in SUPPORTED_TASKS:
        return f"task '{info.task.value}' is not supported for inference"
    if tuple(info.feature_names) != FEATURE_NAMES:
        expected, got = set(FEATURE_NAMES), set(info.feature_names)
        detail = []
        if missing := sorted(expected - got):
            detail.append(f"missing {missing}")
        if extra := sorted(got - expected):
            detail.append(f"unknown {extra}")
        return "feature names differ from this version's features" + (
            f" ({'; '.join(detail)})" if detail else " (order differs)"
        )
    return None


@dataclass(frozen=True, slots=True)
class LoadedModel:
    info: MLModelInfo
    estimator: Any  # fitted scikit-learn estimator (untyped library)


class ModelRegistry:
    """Save, list and load models under one directory (blocking file I/O)."""

    def __init__(self, models_dir: Path) -> None:
        self._root = Path(models_dir)

    @property
    def root(self) -> Path:
        return self._root

    def model_dir(self, name: str) -> Path:
        """Directory of model ``name``.

        Raises:
            ValueError: if the name is not a valid model name.
        """
        if not is_valid_model_name(name):
            raise ValueError(f"invalid model name {name!r}")
        return self._root / name

    def save(self, estimator: object, info: MLModelInfo, *, overwrite: bool = False) -> Path:
        """Write ``model.joblib`` and ``metadata.json`` for ``info.name``; returns the directory.

        Raises:
            ValueError: invalid name, or the model exists and ``overwrite`` is false.
        """
        directory = self.model_dir(info.name)
        if directory.exists() and not overwrite:
            raise ValueError(f"model {info.name!r} already exists in {self._root}")
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(estimator, directory / MODEL_FILE)
        (directory / METADATA_FILE).write_text(info.model_dump_json(indent=2), encoding="utf-8")
        return directory

    def load_info(self, name: str) -> MLModelInfo:
        """Metadata of model ``name`` (no unpickling).

        Raises:
            ModelNotAvailableError: unknown / invalid name or unreadable metadata.
        """
        if not is_valid_model_name(name):
            raise ModelNotAvailableError(f"invalid model name {name!r}")
        directory = self._root / name
        try:
            raw = json.loads((directory / METADATA_FILE).read_text(encoding="utf-8"))
            info = MLModelInfo.model_validate(raw)
        except FileNotFoundError as exc:
            raise ModelNotAvailableError(f"model {name!r} not found in {self._root}") from exc
        except (OSError, ValueError, ValidationError) as exc:
            raise ModelNotAvailableError(f"model {name!r} has invalid metadata: {exc}") from exc
        if info.name != name:
            raise ModelNotAvailableError(
                f"metadata of model directory {name!r} names a different model {info.name!r}"
            )
        if not (directory / MODEL_FILE).is_file():
            raise ModelNotAvailableError(f"model {name!r} has no {MODEL_FILE}")
        return info

    def list_models(self) -> list[MLModelInfo]:
        """Every model with readable metadata, sorted by name; broken entries are logged."""
        if not self._root.is_dir():
            return []
        infos: list[MLModelInfo] = []
        for entry in sorted(self._root.iterdir()):
            if not entry.is_dir() or not is_valid_model_name(entry.name):
                continue
            try:
                infos.append(self.load_info(entry.name))
            except ModelNotAvailableError as exc:
                logger.warning("skipping model directory %s: %s", entry, exc)
        return infos

    def load(self, name: str) -> LoadedModel:
        """Metadata and estimator of a compatible model (unpickles ``model.joblib``).

        Raises:
            ModelNotAvailableError: unknown, incompatible or unloadable model.
        """
        info = self.load_info(name)
        problem = compatibility_problem(info)
        if problem is not None:
            raise ModelNotAvailableError(f"model {name!r} is not usable: {problem}")
        try:
            estimator = joblib.load(self._root / name / MODEL_FILE)
        except Exception as exc:  # unpickling can raise anything
            raise ModelNotAvailableError(f"model {name!r} could not be loaded: {exc}") from exc
        required = "predict_proba" if info.task is MLTask.BAD_POINT else "predict"
        if not callable(getattr(estimator, required, None)):
            raise ModelNotAvailableError(
                f"model {name!r} ({info.task.value}) has no {required}() method"
            )
        return LoadedModel(info=info, estimator=estimator)
