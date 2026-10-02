"""HIL: ArduinoStage on a flashed Uno - a small coordinated move, STOP and home.

Moves stay within 20 um of where the stage is when the test starts (that place
becomes the origin when the port opens and the Uno resets).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest
from tests.hil.helpers import hil_settings, requires_arduino

from confocal.errors import MotionAbortedError
from confocal.hardware.arduino.stage import ArduinoStage
from confocal.models.common import Position
from confocal.models.hardware import StageState

pytestmark = requires_arduino

ORIGIN = Position(x_um=0.0, y_um=0.0, z_um=0.0)


@pytest.fixture
async def stage() -> AsyncIterator[ArduinoStage]:
    driver = ArduinoStage.from_settings(hil_settings())
    await driver.connect()
    try:
        yield driver
    finally:
        try:
            await driver.stop()
            await driver.release()
        finally:
            await driver.close()


async def test_handshake_status(stage: ArduinoStage) -> None:
    status = await stage.status()
    assert status.connected
    assert status.state is StageState.IDLE
    assert status.position == ORIGIN
    assert status.firmware_version
    assert await stage.ping() < 0.1


async def test_small_coordinated_move_and_back(stage: ArduinoStage) -> None:
    tolerance = hil_settings().motion.position_tolerance_um
    for target in (
        Position(x_um=5.0, y_um=0.0, z_um=0.0),
        Position(x_um=5.0, y_um=-5.0, z_um=3.0),
        ORIGIN,
    ):
        reported = await stage.move_to(target)
        assert reported.max_axis_error(target) <= stage.mapper.resolution_um() + 1e-9
        assert reported.max_axis_error(target) <= tolerance
        assert await stage.get_position() == reported
        assert stage.last_steps == stage.mapper.to_motor_steps(target)


async def test_stop_mid_move(stage: ArduinoStage) -> None:
    target = Position(x_um=0.0, y_um=0.0, z_um=20.0)
    move = asyncio.create_task(stage.move_to(target))
    await asyncio.sleep(0.3)
    await stage.stop()
    with pytest.raises(MotionAbortedError):
        await move
    partial = await stage.get_position()
    assert 0.0 < partial.z_um < target.z_um
    assert (await stage.status()).state is StageState.STOPPED
    assert await stage.home() == ORIGIN
    assert stage.homed
