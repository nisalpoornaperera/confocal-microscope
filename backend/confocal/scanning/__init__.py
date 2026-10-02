"""Scan engine: plan, state machine, adaptive Z, per-point acquisition and the ScanManager.

The engine works in Cartesian micrometres and depends only on the
``MicroscopeController`` ABC, the ``ScanRepository`` protocol and the injected
physics callables of :mod:`confocal.scanning.protocols`.
"""

from confocal.scanning.control import ScanControl, ScanStopRequested, StopKind, StopRequest
from confocal.scanning.events import DROPPABLE_EVENT_TYPES, ScanEventBroker, Subscription
from confocal.scanning.executor import ScanExecutor
from confocal.scanning.manager import SHUTDOWN_MESSAGE, ScanManager
from confocal.scanning.plan import (
    MOTOR_RESOLUTION_UM,
    GridPoint,
    ScanPlan,
    axis_positions,
    grid_axes,
    order_points,
    sweep_is_clipped,
    sweep_length,
    z_sweep,
)
from confocal.scanning.profile_buffer import CalibrationSnapshot, ProfileBuffer
from confocal.scanning.progress import EventThrottle, LiveEventConfig, ProgressTracker
from confocal.scanning.protocols import (
    CoarsePeakFinder,
    MLAnalyser,
    ProfileAnalyser,
    SurfaceReconstructor,
)
from confocal.scanning.state import ALLOWED_TRANSITIONS, ScanStateMachine, StateTransition
from confocal.scanning.z_estimation import AdaptiveZEstimator, ZEstimate

__all__ = [
    "ALLOWED_TRANSITIONS",
    "DROPPABLE_EVENT_TYPES",
    "MOTOR_RESOLUTION_UM",
    "SHUTDOWN_MESSAGE",
    "AdaptiveZEstimator",
    "CalibrationSnapshot",
    "CoarsePeakFinder",
    "EventThrottle",
    "GridPoint",
    "LiveEventConfig",
    "MLAnalyser",
    "ProfileAnalyser",
    "ProfileBuffer",
    "ProgressTracker",
    "ScanControl",
    "ScanEventBroker",
    "ScanExecutor",
    "ScanManager",
    "ScanPlan",
    "ScanStateMachine",
    "ScanStopRequested",
    "StateTransition",
    "StopKind",
    "StopRequest",
    "Subscription",
    "SurfaceReconstructor",
    "ZEstimate",
    "axis_positions",
    "grid_axes",
    "order_points",
    "sweep_is_clipped",
    "sweep_length",
    "z_sweep",
]
