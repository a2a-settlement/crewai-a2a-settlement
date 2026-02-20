"""
crewai-a2a-settlement — Bridge CrewAI agents with the A2A Settlement Exchange.

Public API:
    A2AConfig                — Environment-driven configuration
    A2ASettlementClient      — Singleton HTTP client for the exchange
    A2ASettlementError       — Base exception
    A2AAuthError             — Authentication failures
    A2AEscrowError           — Escrow operation failures
    A2AReleaseError          — Release/cancel failures
    A2ARegistrationError     — Agent registration failures
    A2ANetworkError          — Transient network failures
    EscrowReceipt            — Locked-funds receipt
    SettlementResult         — Release/cancel confirmation
    BatchSettlementResult    — Aggregated batch-release result
    SessionSummary           — Aggregated session metrics
    AgentRegistration        — Agent registration result
    NegotiationEntry         — Single dialogue turn in a negotiation
    NegotiationTranscript    — Hashed, tamper-evident negotiation log
    build_transcript         — Assemble dialogue into a signed transcript
    hash_transcript          — SHA-256 of canonical JSON
    verify_transcript        — Verify transcript integrity
    validate_no_execution_authority — Guard against execution commands
    TranscriptValidationError — Forbidden phrase detected
    TranscriptIntegrityError  — Hash mismatch on verification
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
    BatchSettlementResult,
    EscrowReceipt,
    NegotiationEntry,
    NegotiationTranscript,
    SessionSummary,
    SettlementResult,
)
from .transcript import (
    TranscriptIntegrityError,
    TranscriptValidationError,
    build_transcript,
    hash_transcript,
    validate_no_execution_authority,
    verify_transcript,
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
    "BatchSettlementResult",
    "EscrowReceipt",
    "SettlementResult",
    "SessionSummary",
    "NegotiationEntry",
    "NegotiationTranscript",
    "build_transcript",
    "hash_transcript",
    "verify_transcript",
    "validate_no_execution_authority",
    "TranscriptValidationError",
    "TranscriptIntegrityError",
]
