"""
client.py — A2A-SE SDK wrapper for CrewAI integration

Uses the official a2a-settlement SDK (SettlementExchangeClient) instead of
raw httpx, ensuring API path compatibility with the core exchange.

Design goals:
  - Singleton pattern so SettledCrew, SettledTask, and SettledAgent
    can all share one authenticated session without passing state around
  - Every method raises a typed A2ASettlementError on failure so callers
    don't need to inspect raw HTTP responses
  - Retry logic built in for transient failures (escrow operations
    are idempotent with the same task_id / idempotency_key)
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import httpx
from a2a_settlement.client import SettlementExchangeClient

from .config import A2AConfig
from .models import BatchSettlementResult, EscrowReceipt, SessionSummary, SettlementResult

logger = logging.getLogger("crewai_a2a_settlement.client")


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class A2ASettlementError(Exception):
    """Base exception for all A2A-SE client errors."""


class A2AAuthError(A2ASettlementError):
    """API key missing, invalid, or expired."""


class A2AEscrowError(A2ASettlementError):
    """Escrow operation failed — insufficient balance, invalid address, etc."""


class A2AReleaseError(A2ASettlementError):
    """Release or cancel of escrow failed."""


class A2ARegistrationError(A2ASettlementError):
    """Agent registration with the exchange failed."""


class A2ANetworkError(A2ASettlementError):
    """Unrecoverable network error after retries exhausted."""


# ---------------------------------------------------------------------------
# Internal retry helper
# ---------------------------------------------------------------------------

def _with_retries(fn, *, retries: int = 3, backoff: float = 1.0, label: str = ""):
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except (httpx.TimeoutException, httpx.NetworkError, A2ANetworkError) as exc:
            last_exc = exc
            wait = backoff * (2 ** (attempt - 1))
            logger.warning(
                "A2A-SE %s: retriable error on attempt %d/%d, retrying in %.1fs: %s",
                label, attempt, retries, wait, exc,
            )
            if attempt < retries:
                time.sleep(wait)
    raise A2ANetworkError(
        f"{label} failed after {retries} attempts: {last_exc}"
    ) from last_exc


# ---------------------------------------------------------------------------
# Main client
# ---------------------------------------------------------------------------

class A2ASettlementClient:
    """
    Singleton client for the A2A Settlement Exchange.

    Uses the official SettlementExchangeClient SDK for all exchange calls.
    """

    _instance: Optional["A2ASettlementClient"] = None

    def __init__(self, config: A2AConfig):
        self._config = config
        self._sdk = SettlementExchangeClient(
            base_url=config.exchange_url,
            api_key=config.api_key,
            timeout_s=float(config.timeout_seconds),
        )
        self._session_receipts: list[EscrowReceipt] = []
        self._session_results: list[SettlementResult] = []
        self._pending_releases: list[str] = []
        logger.info(
            "A2ASettlementClient initialized: exchange=%s network=%s",
            config.exchange_url,
            config.network,
        )

    # ------------------------------------------------------------------
    # Singleton lifecycle
    # ------------------------------------------------------------------

    @classmethod
    def initialize(
        cls,
        config: Optional[A2AConfig] = None,
    ) -> "A2ASettlementClient":
        resolved_config = config or A2AConfig()
        if not resolved_config.api_key:
            raise A2AAuthError(
                "A2A_API_KEY is not set. Export it or pass it via A2AConfig(api_key=...). "
                "Get a sandbox key at https://sandbox.a2a-settlement.org"
            )
        cls._instance = cls(resolved_config)
        return cls._instance

    @classmethod
    def get_instance(cls) -> "A2ASettlementClient":
        if cls._instance is None:
            raise A2ASettlementError(
                "A2ASettlementClient not initialized. Use SettledCrew (not bare Crew) "
                "to ensure the client is set up before tasks execute."
            )
        return cls._instance

    @classmethod
    def _clear_instance(cls) -> None:
        cls._instance = None

    # ------------------------------------------------------------------
    # Agent registration
    # ------------------------------------------------------------------

    def register_agent(
        self,
        name: str,
        capabilities: list[str],
        metadata: Optional[dict] = None,
    ) -> str:
        def _call():
            result = self._sdk.register_account(
                bot_name=name,
                developer_id=metadata.get("developer_id", "crewai") if metadata else "crewai",
                developer_name=(
                    metadata.get("developer_name", "CrewAI Agent")
                    if metadata
                    else "CrewAI Agent"
                ),
                contact_email=(
                    metadata.get("contact_email", "noreply@localhost")
                    if metadata
                    else "noreply@localhost"
                ),
                skills=capabilities,
            )
            return result

        try:
            data = _with_retries(_call, label="register_agent")
        except (A2ANetworkError, A2ASettlementError):
            raise
        except Exception as exc:
            raise A2ARegistrationError(
                f"Unexpected error registering agent '{name}': {exc}"
            ) from exc

        account_id = data.get("account", {}).get("id", "")
        logger.info("Agent registered: name=%s id=%s", name, account_id)
        return account_id

    # ------------------------------------------------------------------
    # Escrow lifecycle
    # ------------------------------------------------------------------

    def escrow(
        self,
        payer_address: str,
        payee_address: str,
        amount: float,
        task_id: str,
        description: str = "",
        idempotency_key: Optional[str] = None,
    ) -> EscrowReceipt:
        def _call():
            return self._sdk.create_escrow(
                provider_id=payee_address,
                amount=int(amount),
                task_id=task_id,
                task_type=description[:64] if description else "crewai-task",
                idempotency_key=idempotency_key,
            )

        try:
            data = _with_retries(_call, label="escrow")
        except (A2ANetworkError, A2ASettlementError):
            raise
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status == 401:
                raise A2AAuthError(f"[escrow] Unauthorized: {exc}") from exc
            if status in (400, 402):
                raise A2AEscrowError(f"[escrow] Failed: {exc}") from exc
            if 500 <= status < 600:
                raise A2ANetworkError(f"[escrow] Server error: {exc}") from exc
            raise A2AEscrowError(f"[escrow] Unexpected error: {exc}") from exc
        except Exception as exc:
            raise A2AEscrowError(
                f"Unexpected error creating escrow for task {task_id}: {exc}"
            ) from exc

        receipt = EscrowReceipt(
            escrow_id=data["escrow_id"],
            task_id=task_id,
            payer_address=payer_address,
            payee_address=payee_address,
            amount=amount,
            status="escrowed",
            expires_at=str(data.get("expires_at", "")),
        )
        self._session_receipts.append(receipt)
        logger.info("Escrow created: id=%s task=%s amount=%.4f", receipt.escrow_id, task_id, amount)
        return receipt

    def release(self, escrow_id: str) -> SettlementResult:
        if self._config.batch_settlements:
            self.defer_release(escrow_id)
            return SettlementResult(
                escrow_id=escrow_id, status="deferred", tx_hash="", settled_at=""
            )

        def _call():
            return self._sdk.release_escrow(escrow_id=escrow_id)

        try:
            _with_retries(_call, label="release")
        except (A2ANetworkError, A2ASettlementError):
            raise
        except Exception as exc:
            raise A2AReleaseError(f"Unexpected error releasing escrow {escrow_id}: {exc}") from exc

        result = SettlementResult(
            escrow_id=escrow_id,
            status="released",
            tx_hash="",
            settled_at="",
        )
        self._session_results.append(result)
        logger.info("Escrow released: id=%s", escrow_id)
        return result

    def cancel(self, escrow_id: str, reason: str = "") -> SettlementResult:
        def _call():
            return self._sdk.refund_escrow(escrow_id=escrow_id, reason=reason)

        try:
            _with_retries(_call, label="cancel")
        except (A2ANetworkError, A2ASettlementError):
            raise
        except Exception as exc:
            raise A2AReleaseError(f"Unexpected error cancelling escrow {escrow_id}: {exc}") from exc

        result = SettlementResult(
            escrow_id=escrow_id, status="cancelled", tx_hash="", settled_at=""
        )
        self._session_results.append(result)
        logger.info("Escrow cancelled: id=%s reason=%s", escrow_id, reason or "(none)")
        return result

    # ------------------------------------------------------------------
    # Batch settlement
    # ------------------------------------------------------------------

    def defer_release(self, escrow_id: str) -> None:
        self._pending_releases.append(escrow_id)
        logger.info("Escrow deferred for batch release: id=%s", escrow_id)

    def flush_settlements(self) -> BatchSettlementResult:
        if not self._pending_releases:
            return BatchSettlementResult()

        escrow_ids = list(self._pending_releases)
        results: list[SettlementResult] = []
        failed_ids: list[str] = []

        for eid in escrow_ids:
            try:
                self._sdk.release_escrow(escrow_id=eid)
                result = SettlementResult(
                    escrow_id=eid, status="released", tx_hash="", settled_at=""
                )
                results.append(result)
                self._session_results.append(result)
            except Exception as exc:
                logger.warning("Batch release failed for %s: %s", eid, exc)
                failed_ids.append(eid)

        receipt_amounts = {r.escrow_id: r.amount for r in self._session_receipts}
        total_released = sum(receipt_amounts.get(r.escrow_id, 0.0) for r in results)
        self._pending_releases.clear()

        return BatchSettlementResult(
            results=results,
            batch_tx_hash="",
            settled_at="",
            total_released=total_released,
            escrow_count=len(results),
            failed_escrow_ids=failed_ids,
        )

    def get_pending_count(self) -> int:
        return len(self._pending_releases)

    def get_pending_escrow_ids(self) -> list[str]:
        return list(self._pending_releases)

    # ------------------------------------------------------------------
    # Status and account queries
    # ------------------------------------------------------------------

    def get_escrow_status(self, escrow_id: str) -> dict:
        def _call():
            return self._sdk.get_escrow(escrow_id=escrow_id)
        return _with_retries(_call, label="get_escrow_status")

    def get_balance(self) -> float:
        def _call():
            return self._sdk.get_balance()
        data = _with_retries(_call, label="get_balance")
        return float(data.get("available", 0))

    def get_account_history(self, limit: int = 50, offset: int = 0) -> list[dict]:
        def _call():
            return self._sdk.get_transactions(limit=limit, offset=offset)
        data = _with_retries(_call, label="get_account_history")
        return data.get("transactions", [])

    # ------------------------------------------------------------------
    # Session summary
    # ------------------------------------------------------------------

    def get_session_receipts(self) -> SessionSummary:
        released_ids = {s.escrow_id for s in self._session_results if s.status == "released"}
        cancelled_ids = {s.escrow_id for s in self._session_results if s.status == "cancelled"}
        total_released = sum(
            r.amount for r in self._session_receipts if r.escrow_id in released_ids
        )
        total_cancelled = sum(
            r.amount for r in self._session_receipts if r.escrow_id in cancelled_ids
        )
        total_escrowed = sum(r.amount for r in self._session_receipts)

        return SessionSummary(
            receipts=list(self._session_receipts),
            results=list(self._session_results),
            total_escrowed=total_escrowed,
            total_released=total_released,
            total_cancelled=total_cancelled,
            cancelled_count=len(cancelled_ids),
        )

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass
