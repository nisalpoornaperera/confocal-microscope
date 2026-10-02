"""Application wiring and lifecycle (dependency container)."""

from confocal.services.container import ServiceContainer
from confocal.services.system import (
    build_system_info,
    build_system_status,
    component_errors,
    overall_status,
)

__all__ = [
    "ServiceContainer",
    "build_system_info",
    "build_system_status",
    "component_errors",
    "overall_status",
]
