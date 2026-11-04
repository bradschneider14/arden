"""Attention gate interface and the M4 HeartbeatGate placeholder."""

from arden.agent.loop import (
    DEFAULT_HEARTBEAT_INTERVAL,
    Gate,
    GateContext,
    HeartbeatGate,
    WakeDecision,
)

__all__ = [
    "DEFAULT_HEARTBEAT_INTERVAL",
    "Gate",
    "GateContext",
    "HeartbeatGate",
    "WakeDecision",
]
