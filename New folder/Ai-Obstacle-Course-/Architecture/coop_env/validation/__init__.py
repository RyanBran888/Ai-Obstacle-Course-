"""Room validation: structural rules plus a solvability proof.

The analysis here is static and runs once, at generation time. It decides
whether a room *can* be completed; it never plans, routes, or moves anything.
"""

from .connectivity import ConnectivityModel, DoorCluster, build_connectivity
from .solvability import AGENT_SLOTS, SolvabilityReport, analyse
from .validator import (
    STRUCTURAL_RULES,
    Issue,
    Severity,
    ValidationReport,
    validate_room,
)

__all__ = [
    "validate_room",
    "ValidationReport",
    "Issue",
    "Severity",
    "STRUCTURAL_RULES",
    "build_connectivity",
    "ConnectivityModel",
    "DoorCluster",
    "analyse",
    "SolvabilityReport",
    "AGENT_SLOTS",
]
