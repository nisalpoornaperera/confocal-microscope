"""``/api/v1/adc``: detector status, single readings and gain configuration."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query

from confocal.api.deps import ControllerDep, NoActiveScan
from confocal.api.errors import ERROR_RESPONSES
from confocal.models.calibration import ADCCalibrateRequest, ADCCalibrateResponse
from confocal.models.hardware import ADCStatus, SamplingMethod
from confocal.models.measurement import IntensityMeasurement

router = APIRouter(prefix="/api/v1/adc", tags=["adc"], responses=ERROR_RESPONSES)


@router.get("/status", response_model=ADCStatus, summary="ADC gain, data rate, last reading")
async def get_status(controller: ControllerDep) -> ADCStatus:
    return (await controller.status()).adc


@router.get(
    "/read",
    response_model=IntensityMeasurement,
    dependencies=[NoActiveScan],
    summary="Calibrated intensity reading at the current position",
)
async def read(
    controller: ControllerDep,
    n_samples: Annotated[int, Query(ge=1, le=1024)] = 16,
    method: SamplingMethod = SamplingMethod.MEDIAN,
) -> IntensityMeasurement:
    return await controller.measure_intensity(n_samples, method)


@router.post(
    "/calibrate",
    response_model=ADCCalibrateResponse,
    dependencies=[NoActiveScan],
    summary="Set the ADC gain explicitly or auto-range it",
)
async def calibrate(
    request: ADCCalibrateRequest, controller: ControllerDep
) -> ADCCalibrateResponse:
    return await controller.configure_adc(request)
