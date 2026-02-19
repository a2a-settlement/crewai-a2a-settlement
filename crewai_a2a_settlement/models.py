"""
models.py — Shared dataclasses for crewai-a2a-settlement.

These are the data structures passed between the client, tasks,
agents, and crew. Keeping them in one file avoids circular imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AgentRegistration:
    """Result of registering an agent with the exchange."""
    wallet_address: str
    agent_id: str
    name: str


@dataclass
class EscrowReceipt:
    """
    Returned by client.escrow(). Represents locked funds
    that haven't been released or cancelled yet.
    """
    escrow_id: str
    task_id: str
    payer_address: str
    payee_address: str
    amount: float
    status: str          # "escrowed" | "released" | "cancelled"
    created_at: str


@dataclass
class SettlementResult:
    """
    Returned by client.release() or client.cancel().
    Represents a completed (finalized) escrow operation.
    """
    escrow_id: str
    status: str          # "released" | "cancelled"
    tx_hash: str
    settled_at: str


@dataclass
class SessionSummary:
    """
    Aggregated settlement data for an entire SettledCrew session.
    Attached to CrewOutput as .settlement_receipts after kickoff().
    """
    receipts: list[EscrowReceipt] = field(default_factory=list)
    results: list[SettlementResult] = field(default_factory=list)
    total_escrowed: float = 0.0
    total_released: float = 0.0
    total_cancelled: float = 0.0
    cancelled_count: int = 0

    def __str__(self) -> str:
        lines = [
            "=== A2A-SE Settlement Summary ===",
            f"  Escrow operations : {len(self.receipts)}",
            f"  Total escrowed    : {self.total_escrowed:.4f} tokens",
            f"  Total released    : {self.total_released:.4f} tokens",
            f"  Total cancelled   : {self.total_cancelled:.4f} tokens",
            f"  Cancelled tasks   : {self.cancelled_count}",
        ]
        if self.receipts:
            lines.append("\n  Per-task breakdown:")
            for r in self.receipts:
                result_status = next(
                    (s.status for s in self.results if s.escrow_id == r.escrow_id),
                    "pending"
                )
                lines.append(
                    f"    task={r.task_id[:8]}... "
                    f"amount={r.amount:.4f} "
                    f"status={result_status}"
                )
        return "\n".join(lines)
