"""Shared fixtures. Subsystem-specific fixtures live in each test package's conftest."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from confocal.config import Settings, SimulationConfig, StorageConfig


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    path = tmp_path / "data"
    path.mkdir()
    return path


@pytest.fixture
def sim_settings(data_dir: Path) -> Settings:
    """Full simulation, instantaneous motion, deterministic noise, isolated data dir."""
    return Settings(
        simulation=SimulationConfig(time_scale=0.0, seed=42),
        storage=StorageConfig(data_dir=data_dir),
    )


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20261001)
