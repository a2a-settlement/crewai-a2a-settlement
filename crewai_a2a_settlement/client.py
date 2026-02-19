"""
client.py — A2A-SE SDK wrapper for CrewAI integration

Thin abstraction layer between crewai-a2a-settlement and the core
A2A Settlement Exchange API. All network calls go through this class.

Design goals:
  - Singleton pattern so SettledCrew, SettledTask, and SettledAgent
    can all share one authenticated session without passing state around
  - Every method raises a typed A2ASettlementError on failure so callers
    don't need to inspect raw HTTP responses
  - Retry logic built in for transient failures (escrow operations
    are idempotent with the same task_id / idempotency_key)
  - Pluggable transport for testing (inject a mock httpx.Client)
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import httpx

from .config import A2AConfig
from .models import EscrowReceipt, SessionSummary, SettlementResult

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
    """
    Execute fn(), retrying up to `retries` times on transient network errors
    or 5xx server errors. Raises A2ANetworkError if all attempts fail.

    Retried: httpx transport errors AND A2ANetworkError (raised by _raise_for_status
    for 5xx responses). Not retried: auth errors, validation errors, or any other
    typed A2ASettlementError subclass — those indicate a problem with the request
    itself, not a transient failure.
    """
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

    Usage:
        # In SettledCrew.kickoff():
        client = A2ASettlementClient.initialize(config)

        # In SettledTask.execute_sync():
        client = A2ASettlementClient.get_instance()
        receipt = client.escrow(...)
        client.release(receipt.escrow_id)

    Never instantiate directly — always use initialize() or get_instance().
    """

    _instance: Optional["A2ASettlementClient"] = None

    def __init__(
        self,
        config: A2AConfig,
        http_client: Optional[httpx.Client] = None,
    ):
        self._config = config
        self._http = http_client or httpx.Client(
            base_url=config.exchange_url,
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
                "X-Client": "crewai-a2a-settlement/0.1.0",
            },
            timeout=config.timeout_seconds,
        )
        # Track every escrow created this session for the settlement summary
        self._session_receipts: list[EscrowReceipt] = []
        self._session_results: list[SettlementResult] = []
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
        http_client: Optional[httpx.Client] = None,
    ) -> "A2ASettlementClient":
        """
        Create (or recreate) the singleton client. Call once in SettledCrew.kickoff().
        Pass http_client in tests to inject a mock transport.
        """
        resolved_config = config or A2AConfig()
        if not resolved_config.api_key:
            raise A2AAuthError(
                "A2ASE_API_KEY is not set. Export it or pass it via A2AConfig(api_key=...). "
                "Get a sandbox key at https://sandbox.a2a-se.dev"
            )
        cls._instance = cls(resolved_config, http_client=http_client)
        return cls._instance

    @classmethod
    def get_instance(cls) -> "A2ASettlementClient":
        """
        Return the active singleton. Raises if initialize() was never called.
        This is intentional — SettledTask should never silently skip settlement.
        """
        if cls._instance is None:
            raise A2ASettlementError(
                "A2ASettlementClient not initialized. Use SettledCrew (not bare Crew) "
                "to ensure the client is set up before tasks execute."
            )
        return cls._instance

    @classmethod
    def _clear_instance(cls) -> None:
        """Reset the singleton. Used in tests only — do not call in application code."""
        if cls._instance and cls._instance._http:
            try:
                cls._instance._http.close()
            except Exception:
                pass
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
        """
        Register an agent with the exchange and return its wallet address.

        Args:
            name:         Human-readable identifier (typically the agent's role).
            capabilities: List of capability strings (typically the agent's goals).
            metadata:     Optional dict of additional key-value metadata.

        Returns:
            wallet_address assigned by the exchange.
        """
        payload = {
            "name": name,
            "capabilities": capabilities,
            "metadata": metadata or {},
            "network": self._config.network,
        }

        def _call():
            resp = self._http.post("/v1/agents/register", json=payload)
            _raise_for_status(resp, "register_agent")
            return resp.json()

        try:
            data = _with_retries(_call, label="register_agent")
        except (A2ANetworkError, A2ASettlementError):
            raise
        except Exception as exc:
            raise A2ARegistrationError(
                f"Unexpected error registering agent '{name}': {exc}"
            ) from exc

        wallet = data["wallet_address"]
        logger.info("Agent registered: name=%s wallet=%s", name, wallet)
        return wallet

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
        """
        Lock `amount` tokens from payer into escrow, earmarked for payee.

        The idempotency_key defaults to task_id so retrying a failed
        escrow call with the same task_id is safe — the exchange deduplicates.

        Returns:
            EscrowReceipt with escrow_id, amount, addresses, and timestamp.

        Raises:
            A2AEscrowError:   Insufficient balance, invalid address, etc.
            A2AAuthError:     Invalid API key.
            A2ANetworkError:  Exchange unreachable after retries.
        """
        key = idempotency_key or f"task-{task_id}"
        payload = {
            "payer_address": payer_address,
            "payee_address": payee_address,
            "amount": amount,
            "task_id": task_id,
            "description": description,
            "idempotency_key": key,
            "network": self._config.network,
        }

        def _call():
            resp = self._http.post("/v1/escrow", json=payload)
            _raise_for_status(resp, "escrow")
            return resp.json()

        try:
            data = _with_retries(_call, label="escrow")
        except (A2ANetworkError, A2ASettlementError):
            raise
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
            created_at=data.get("created_at", ""),
        )
        self._session_receipts.append(receipt)
        logger.info(
            "Escrow created: id=%s task=%s amount=%.4f",
            receipt.escrow_id, task_id, amount,
        )
        return receipt

    def release(self, escrow_id: str) -> SettlementResult:
        """
        Release escrowed funds to the payee. Call this on task success.

        Returns:
            SettlementResult with tx_hash confirming the transfer.

        Raises:
            A2AReleaseError:  Escrow not found, already settled, etc.
            A2ANetworkError:  Exchange unreachable after retries.
        """

        def _call():
            resp = self._http.post(f"/v1/escrow/{escrow_id}/release")
            _raise_for_status(resp, "release", validation_error_class=A2AReleaseError)
            return resp.json()

        try:
            data = _with_retries(_call, label="release")
        except (A2ANetworkError, A2ASettlementError):
            raise
        except Exception as exc:
            raise A2AReleaseError(
                f"Unexpected error releasing escrow {escrow_id}: {exc}"
            ) from exc

        result = SettlementResult(
            escrow_id=escrow_id,
            status="released",
            tx_hash=data.get("tx_hash", ""),
            settled_at=data.get("settled_at", ""),
        )
        self._session_results.append(result)
        logger.info("Escrow released: id=%s tx=%s", escrow_id, result.tx_hash)
        return result

    def cancel(self, escrow_id: str, reason: str = "") -> SettlementResult:
        """
        Cancel escrow and return funds to payer. Call this on task failure.

        Args:
            escrow_id: The escrow to cancel.
            reason:    Optional string logged in the exchange audit trail.

        Returns:
            SettlementResult confirming the cancellation.

        Raises:
            A2AReleaseError:  Escrow not found, already settled, etc.
            A2ANetworkError:  Exchange unreachable after retries.
        """
        payload = {"reason": reason} if reason else {}

        def _call():
            resp = self._http.post(
                f"/v1/escrow/{escrow_id}/cancel", json=payload
            )
            _raise_for_status(resp, "cancel", validation_error_class=A2AReleaseError)
            return resp.json()

        try:
            data = _with_retries(_call, label="cancel")
        except (A2ANetworkError, A2ASettlementError):
            raise
        except Exception as exc:
            raise A2AReleaseError(
                f"Unexpected error cancelling escrow {escrow_id}: {exc}"
            ) from exc

        result = SettlementResult(
            escrow_id=escrow_id,
            status="cancelled",
            tx_hash=data.get("tx_hash", ""),
            settled_at=data.get("settled_at", ""),
        )
        self._session_results.append(result)
        logger.info("Escrow cancelled: id=%s reason=%s", escrow_id, reason or "(none)")
        return result

    # ------------------------------------------------------------------
    # Status and account queries
    # ------------------------------------------------------------------

    def get_escrow_status(self, escrow_id: str) -> dict:
        """
        Poll the exchange for the current status of an escrow.
        Useful for dashboards, the sandbox demo UI, or dispute workflows.
        """

        def _call():
            resp = self._http.get(f"/v1/escrow/{escrow_id}")
            _raise_for_status(resp, "get_escrow_status")
            return resp.json()

        return _with_retries(_call, label="get_escrow_status")

    def get_balance(self, wallet_address: str) -> float:
        """Return the available (non-escrowed) token balance for a wallet address."""

        def _call():
            resp = self._http.get(f"/v1/accounts/{wallet_address}/balance")
            _raise_for_status(resp, "get_balance")
            return resp.json()

        data = _with_retries(_call, label="get_balance")
        return float(data.get("available_balance", 0.0))

    def get_account_history(
        self,
        wallet_address: str,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict]:
        """Return paginated transaction history for a wallet address."""

        def _call():
            resp = self._http.get(
                f"/v1/accounts/{wallet_address}/history",
                params={"limit": limit, "offset": offset},
            )
            _raise_for_status(resp, "get_account_history")
            return resp.json()

        data = _with_retries(_call, label="get_account_history")
        return data.get("transactions", [])

    # ------------------------------------------------------------------
    # Session summary
    # ------------------------------------------------------------------

    def get_session_receipts(self) -> SessionSummary:
        """
        Return aggregated settlement data for the entire SettledCrew session.
        Called by SettledCrew.kickoff() and attached to CrewOutput.
        """
        released_ids = {
            s.escrow_id for s in self._session_results if s.status == "released"
        }
        total_released = sum(
            r.amount
            for r in self._session_receipts
            if r.escrow_id in released_ids
        )
        cancelled_count = sum(
            1 for s in self._session_results if s.status == "cancelled"
        )
        total_escrowed = sum(r.amount for r in self._session_receipts)

        return SessionSummary(
            receipts=list(self._session_receipts),
            results=list(self._session_results),
            total_escrowed=total_escrowed,
            total_released=total_released,
            total_cancelled=total_escrowed - total_released,
            cancelled_count=cancelled_count,
        )

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._http.close()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _raise_for_status(
    resp: httpx.Response,
    operation: str,
    validation_error_class: type = None,
) -> None:
    """
    Translate HTTP error codes into typed A2A-SE exceptions.

    Exchange error responses are expected to be JSON:
      {"error": "...", "code": "...", "detail": "..."}

    Args:
        resp:                   The httpx response to inspect.
        operation:              Name of the operation (for error messages).
        validation_error_class: Exception class to raise for 422 responses.
                                Defaults to A2AEscrowError. Pass A2AReleaseError
                                for release/cancel operations so callers get the
                                correct typed exception.
    """
    if resp.is_success:
        return

    try:
        body = resp.json()
        message = body.get("error") or body.get("detail") or resp.text
    except Exception:
        message = resp.text

    status = resp.status_code
    _422_class = validation_error_class or A2AEscrowError

    if status == 401:
        raise A2AAuthError(f"[{operation}] Unauthorized: {message}")
    if status == 402:
        raise A2AEscrowError(f"[{operation}] Insufficient balance: {message}")
    if status == 404:
        raise A2ASettlementError(f"[{operation}] Not found: {message}")
    if status == 409:
        # Idempotency collision — exchange already processed this request, safe to continue
        logger.warning("[%s] Idempotency collision (409): %s", operation, message)
        return
    if status == 422:
        raise _422_class(f"[{operation}] Validation error: {message}")
    if 500 <= status < 600:
        raise A2ANetworkError(f"[{operation}] Server error {status}: {message}")

    raise A2ASettlementError(f"[{operation}] Unexpected {status}: {message}")
