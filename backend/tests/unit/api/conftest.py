"""Fixtures for the API unit tests: a running app on the full simulation."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from tests.unit.api.helpers import running_client, with_simulation

from confocal.config import Settings


@pytest.fixture
def client(sim_settings: Settings) -> Iterator[TestClient]:
    """Instantaneous simulation (time_scale 0): scans finish in about a second."""
    with running_client(sim_settings) as test_client:
        yield test_client


@pytest.fixture
def slow_settings(sim_settings: Settings) -> Settings:
    """Simulation at 5 % of real time: a small scan stays active for a few seconds."""
    return with_simulation(sim_settings, time_scale=0.05)


@pytest.fixture
def slow_client(slow_settings: Settings) -> Iterator[TestClient]:
    with running_client(slow_settings) as test_client:
        yield test_client
