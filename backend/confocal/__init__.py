"""Confocal surface scanner for the OpenFlexure microscope.

Layering (each layer may only depend on the layers below it):

    api        HTTP / WebSocket transport (FastAPI)
    services   application wiring and lifecycle
    scanning   ScanPlan, ScanManager, state machine, per-point acquisition
    surface    surface reconstruction from (X, Y, Z, confidence)
    ml         optional, advisory post-processing (inference only on the Pi)
    processing pure signal processing of I(Z) profiles
    microscope MicroscopeController: safety, calibration, measurement
    hardware   Stage / ADC / Laser / Camera drivers and simulations
    storage    SQLite metadata + HDF5 measurement arrays
    models     Pydantic models and plain data containers shared by all layers

All application-level coordinates are micrometres. Only the Arduino layer
converts micrometres to motor steps.
"""

from confocal._version import __version__

__all__ = ["__version__"]
