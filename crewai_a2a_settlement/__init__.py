"""
crewai-a2a-settlement — Bridge CrewAI agents with the A2A Settlement Exchange.

Public API:
    A2AConfig              — Environment-driven configuration
    A2ASettlementClient    — Singleton HTTP client for the exchange
    A2ASettlementError     — Base exception
    A2AAuthError           — Authentication failures
    A2AEscrowError         — Escrow operation failures
    A2AReleaseError        — Release/cancel failures
    A2ARegistrationError   — Agent registration failures
    A2ANetworkError        — Transient network failures
    EscrowReceipt          — Locked-funds receipt
    SettlementResult       — Release/cancel confirmation
    SessionSummary         — Aggregated session metrics
    AgentRegistration      — Agent registration result
"""

__version__ = "0.1.0"

from .client import (
    A2AAuthError,
    A2AEscrowError,
    A2ANetworkError,
    A2ARegistrationError,
    A2AReleaseError,
    A2ASettlementClient,
    A2ASettlementError,
)
from .config import A2AConfig
from .models import (
    AgentRegistration,
    EscrowReceipt,
    SessionSummary,
    SettlementResult,
)

__all__ = [
    "__version__",
    "A2AConfig",
    "A2ASettlementClient",
    "A2ASettlementError",
    "A2AAuthError",
    "A2AEscrowError",
    "A2AReleaseError",
    "A2ARegistrationError",
    "A2ANetworkError",
    "AgentRegistration",
    "EscrowReceipt",
    "SettlementResult",
    "SessionSummary",
]
