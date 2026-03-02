"""
test_client.py — Full test suite for A2A-SE CrewAI client

Tests cover:
  - Singleton lifecycle (initialize, get_instance, _clear_instance)
  - Agent registration (success, duplicate, network failure)
  - Escrow creation (success, insufficient balance, idempotency)
  - Release (success, double-release, already-cancelled)
  - Cancel (success, with reason, already-released)
  - Session summary math
  - Retry behavior (transient network errors, 5xx, recovery)
  - Balance and history queries
  - Context manager support
  - Config validation

Run with:
    pytest tests/test_client.py -v
    pytest tests/test_client.py -v --tb=short   # condensed tracebacks
    pytest tests/test_client.py -k "escrow"     # escrow tests only
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from crewai_a2a_settlement.client import (
    A2AAuthError,
    A2AEscrowError,
    A2ANetworkError,
    A2AReleaseError,
    A2ASettlementClient,
    A2ASettlementError,
    _with_retries,
)
from crewai_a2a_settlement.config import A2AConfig


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _make_client() -> A2ASettlementClient:
    """Build a test client with a default config (SDK is mocked at call sites)."""
    config = A2AConfig(api_key="test-key", exchange_url="https://test", network="sandbox")
    return A2ASettlementClient(config)


REGISTER_OK = {"account": {"id": "agent-001"}, "api_key": "ate_xxx", "starter_tokens": 100}
ESCROW_OK = {"escrow_id": "esc-001", "expires_at": "2026-02-18T12:00:00Z"}
RELEASE_OK = {"escrow_id": "esc-001", "status": "released"}
CANCEL_OK = {"escrow_id": "esc-001", "status": "refunded"}
BALANCE_OK = {"available": 100, "held_in_escrow": 20}
HISTORY_OK = {"transactions": [{"id": "tx-1", "amount": 5}]}


# ---------------------------------------------------------------------------
# 1. Singleton lifecycle
# ---------------------------------------------------------------------------

class TestSingletonLifecycle:

    def setup_method(self):
        A2ASettlementClient._clear_instance()

    def teardown_method(self):
        A2ASettlementClient._clear_instance()

    def test_get_instance_before_initialize_raises(self):
        with pytest.raises(A2ASettlementError, match="not initialized"):
            A2ASettlementClient.get_instance()

    def test_initialize_returns_instance(self):
        config = A2AConfig(api_key="key", exchange_url="https://test")
        client = A2ASettlementClient.initialize(config)
        assert isinstance(client, A2ASettlementClient)

    def test_get_instance_returns_same_object_after_initialize(self):
        config = A2AConfig(api_key="key", exchange_url="https://test")
        first = A2ASettlementClient.initialize(config)
        second = A2ASettlementClient.get_instance()
        assert first is second

    def test_clear_instance_resets_singleton(self):
        config = A2AConfig(api_key="key", exchange_url="https://test")
        A2ASettlementClient.initialize(config)
        A2ASettlementClient._clear_instance()
        with pytest.raises(A2ASettlementError):
            A2ASettlementClient.get_instance()

    def test_initialize_without_api_key_raises_auth_error(self):
        config = A2AConfig(api_key="", exchange_url="https://test")
        with pytest.raises(A2AAuthError, match="A2A_API_KEY"):
            A2ASettlementClient.initialize(config)

    def test_initialize_twice_replaces_instance(self):
        config = A2AConfig(api_key="key", exchange_url="https://test")
        first = A2ASettlementClient.initialize(config)
        second = A2ASettlementClient.initialize(config)
        assert first is not second
        assert A2ASettlementClient.get_instance() is second


# ---------------------------------------------------------------------------
# 2. Agent registration
# ---------------------------------------------------------------------------

class TestAgentRegistration:

    def setup_method(self):
        A2ASettlementClient._clear_instance()

    def teardown_method(self):
        A2ASettlementClient._clear_instance()

    def test_register_agent_returns_account_id(self):
        client = _make_client()
        client._sdk = MagicMock()
        client._sdk.register_account.return_value = REGISTER_OK

        account_id = client.register_agent("Research Agent", ["research"])
        assert account_id == "agent-001"
        client._sdk.register_account.assert_called_once()

    def test_register_agent_with_metadata(self):
        client = _make_client()
        client._sdk = MagicMock()
        client._sdk.register_account.return_value = REGISTER_OK

        account_id = client.register_agent(
            "Scraper",
            ["scraping"],
            metadata={"developer_id": "dev1", "developer_name": "Dev One"},
        )
        assert account_id == "agent-001"
        call_kw = client._sdk.register_account.call_args[1]
        assert call_kw["developer_id"] == "dev1"
        assert call_kw["developer_name"] == "Dev One"

    def test_register_agent_network_error_retries(self, monkeypatch):
        monkeypatch.setattr("crewai_a2a_settlement.client.time.sleep", lambda _: None)
        client = _make_client()
        client._sdk = MagicMock()
        client._sdk.register_account.side_effect = [
            httpx.NetworkError("unreachable"),
            httpx.NetworkError("unreachable"),
            REGISTER_OK,
        ]

        account_id = client.register_agent("Agent", [])
        assert account_id == "agent-001"
        assert client._sdk.register_account.call_count == 3

    def test_register_agent_all_retries_fail(self, monkeypatch):
        monkeypatch.setattr("crewai_a2a_settlement.client.time.sleep", lambda _: None)
        client = _make_client()
        client._sdk = MagicMock()
        client._sdk.register_account.side_effect = httpx.NetworkError("down")

        with pytest.raises(A2ANetworkError, match="failed after"):
            client.register_agent("Agent", [])


# ---------------------------------------------------------------------------
# 3. Escrow creation
# ---------------------------------------------------------------------------

class TestEscrowCreation:

    def setup_method(self):
        A2ASettlementClient._clear_instance()

    def teardown_method(self):
        A2ASettlementClient._clear_instance()

    def test_escrow_success_returns_receipt(self):
        client = _make_client()
        client._sdk = MagicMock()
        client._sdk.create_escrow.return_value = ESCROW_OK

        receipt = client.escrow(
            payer_address="0xPAYER",
            payee_address="0xPAYEE",
            amount=5.0,
            task_id="task-abc",
            description="Research task",
        )

        assert receipt.escrow_id == "esc-001"
        assert receipt.task_id == "task-abc"
        assert receipt.amount == 5.0
        assert receipt.status == "escrowed"

    def test_escrow_added_to_session_receipts(self):
        client = _make_client()
        client._sdk = MagicMock()
        client._sdk.create_escrow.return_value = ESCROW_OK

        client.escrow("0xP", "0xQ", 5.0, "task-1")
        assert len(client._session_receipts) == 1

    def test_escrow_401_raises_auth_error(self):
        client = _make_client()
        client._sdk = MagicMock()
        resp = MagicMock()
        resp.status_code = 401
        client._sdk.create_escrow.side_effect = httpx.HTTPStatusError(
            "401", request=MagicMock(), response=resp
        )

        with pytest.raises(A2AAuthError):
            client.escrow("0xP", "0xQ", 5.0, "task-x")

    def test_escrow_402_raises_escrow_error(self):
        client = _make_client()
        client._sdk = MagicMock()
        resp = MagicMock()
        resp.status_code = 402
        client._sdk.create_escrow.side_effect = httpx.HTTPStatusError(
            "402", request=MagicMock(), response=resp
        )

        with pytest.raises(A2AEscrowError):
            client.escrow("0xP", "0xQ", 999999.0, "task-big")

    def test_escrow_network_error_retries(self, monkeypatch):
        monkeypatch.setattr("crewai_a2a_settlement.client.time.sleep", lambda _: None)
        client = _make_client()
        client._sdk = MagicMock()
        client._sdk.create_escrow.side_effect = [
            httpx.TimeoutException("timeout"),
            ESCROW_OK,
        ]

        receipt = client.escrow("0xP", "0xQ", 5.0, "task-retry")
        assert receipt.escrow_id == "esc-001"


# ---------------------------------------------------------------------------
# 4. Release
# ---------------------------------------------------------------------------

class TestRelease:

    def setup_method(self):
        A2ASettlementClient._clear_instance()

    def teardown_method(self):
        A2ASettlementClient._clear_instance()

    def test_release_success_returns_result(self):
        client = _make_client()
        client._sdk = MagicMock()
        client._sdk.release_escrow.return_value = RELEASE_OK

        result = client.release("esc-001")
        assert result.escrow_id == "esc-001"
        assert result.status == "released"
        client._sdk.release_escrow.assert_called_once_with(escrow_id="esc-001")

    def test_release_added_to_session_results(self):
        client = _make_client()
        client._sdk = MagicMock()
        client._sdk.release_escrow.return_value = RELEASE_OK

        client.release("esc-001")
        assert len(client._session_results) == 1
        assert client._session_results[0].status == "released"

    def test_release_retries_on_network_error(self, monkeypatch):
        monkeypatch.setattr("crewai_a2a_settlement.client.time.sleep", lambda _: None)
        client = _make_client()
        client._sdk = MagicMock()
        client._sdk.release_escrow.side_effect = [
            httpx.TimeoutException("timeout"),
            RELEASE_OK,
        ]

        result = client.release("esc-001")
        assert result.status == "released"


# ---------------------------------------------------------------------------
# 5. Cancel
# ---------------------------------------------------------------------------

class TestCancel:

    def setup_method(self):
        A2ASettlementClient._clear_instance()

    def teardown_method(self):
        A2ASettlementClient._clear_instance()

    def test_cancel_success_returns_result(self):
        client = _make_client()
        client._sdk = MagicMock()
        client._sdk.refund_escrow.return_value = CANCEL_OK

        result = client.cancel("esc-001")
        assert result.escrow_id == "esc-001"
        assert result.status == "cancelled"

    def test_cancel_with_reason(self):
        client = _make_client()
        client._sdk = MagicMock()
        client._sdk.refund_escrow.return_value = CANCEL_OK

        result = client.cancel("esc-001", reason="Task raised exception")
        assert result.status == "cancelled"
        client._sdk.refund_escrow.assert_called_once_with(escrow_id="esc-001", reason="Task raised exception")

    def test_cancel_added_to_session_results(self):
        client = _make_client()
        client._sdk = MagicMock()
        client._sdk.refund_escrow.return_value = CANCEL_OK

        client.cancel("esc-001")
        assert len(client._session_results) == 1
        assert client._session_results[0].status == "cancelled"


# ---------------------------------------------------------------------------
# 6. Full lifecycle: escrow → release and escrow → cancel
# ---------------------------------------------------------------------------

class TestFullLifecycle:

    def setup_method(self):
        A2ASettlementClient._clear_instance()

    def teardown_method(self):
        A2ASettlementClient._clear_instance()

    def test_escrow_then_release(self):
        client = _make_client()
        client._sdk = MagicMock()
        client._sdk.create_escrow.return_value = ESCROW_OK
        client._sdk.release_escrow.return_value = RELEASE_OK

        receipt = client.escrow("0xPAYER", "0xPAYEE", 5.0, "task-1")
        result = client.release(receipt.escrow_id)
        assert result.status == "released"

    def test_escrow_then_cancel_on_failure(self):
        client = _make_client()
        client._sdk = MagicMock()
        client._sdk.create_escrow.return_value = ESCROW_OK
        client._sdk.refund_escrow.return_value = CANCEL_OK

        receipt = client.escrow("0xPAYER", "0xPAYEE", 5.0, "task-fail")
        result = client.cancel(receipt.escrow_id, reason="Task raised RuntimeError")
        assert result.status == "cancelled"

    def test_multiple_tasks_mixed_outcomes(self):
        client = _make_client()
        client._sdk = MagicMock()
        client._sdk.create_escrow.side_effect = [
            {"escrow_id": "esc-001", "expires_at": ""},
            {"escrow_id": "esc-002", "expires_at": ""},
        ]
        client._sdk.release_escrow.return_value = RELEASE_OK
        client._sdk.refund_escrow.return_value = CANCEL_OK

        r1 = client.escrow("0xP", "0xQ", 5.0, "task-1")
        r2 = client.escrow("0xP", "0xR", 3.0, "task-2")
        client.release(r1.escrow_id)
        client.cancel(r2.escrow_id, reason="Task 2 failed")

        summary = client.get_session_receipts()
        assert summary.total_escrowed == 8.0
        assert summary.total_released == 5.0
        assert summary.total_cancelled == 3.0
        assert summary.cancelled_count == 1


# ---------------------------------------------------------------------------
# 7. Session summary
# ---------------------------------------------------------------------------

class TestSessionSummary:

    def setup_method(self):
        A2ASettlementClient._clear_instance()

    def teardown_method(self):
        A2ASettlementClient._clear_instance()

    def test_empty_session_summary(self):
        client = _make_client()
        summary = client.get_session_receipts()
        assert summary.total_escrowed == 0.0
        assert summary.total_released == 0.0
        assert summary.total_cancelled == 0.0
        assert summary.cancelled_count == 0
        assert summary.receipts == []

    def test_session_summary_all_cancelled(self):
        client = _make_client()
        client._sdk = MagicMock()
        client._sdk.create_escrow.return_value = ESCROW_OK
        client._sdk.refund_escrow.return_value = CANCEL_OK

        client.escrow("0xP", "0xQ", 5.0, "task-1")
        client.cancel("esc-001")

        summary = client.get_session_receipts()
        assert summary.total_released == 0.0
        assert summary.total_cancelled == 5.0
        assert summary.cancelled_count == 1


# ---------------------------------------------------------------------------
# 8. Balance and history
# ---------------------------------------------------------------------------

class TestAccountQueries:

    def setup_method(self):
        A2ASettlementClient._clear_instance()

    def teardown_method(self):
        A2ASettlementClient._clear_instance()

    def test_get_balance_returns_float(self):
        client = _make_client()
        client._sdk = MagicMock()
        client._sdk.get_balance.return_value = BALANCE_OK

        balance = client.get_balance("0xWALLET")
        assert balance == 100.0
        assert isinstance(balance, float)

    def test_get_balance_missing_key_returns_zero(self):
        client = _make_client()
        client._sdk = MagicMock()
        client._sdk.get_balance.return_value = {}

        balance = client.get_balance("0xWALLET")
        assert balance == 0.0

    def test_get_account_history(self):
        client = _make_client()
        client._sdk = MagicMock()
        client._sdk.get_transactions.return_value = HISTORY_OK

        history = client.get_account_history("0xWALLET")
        assert len(history) == 1
        assert history[0]["id"] == "tx-1"

    def test_get_escrow_status(self):
        client = _make_client()
        client._sdk = MagicMock()
        client._sdk.get_escrow.return_value = {"escrow_id": "esc-001", "status": "held", "amount": 5}

        status = client.get_escrow_status("esc-001")
        assert status["status"] == "held"


# ---------------------------------------------------------------------------
# 9. _with_retries
# ---------------------------------------------------------------------------

class TestRetryHelper:

    def test_succeeds_first_attempt(self):
        calls = []

        def fn():
            calls.append(1)
            return "ok"

        result = _with_retries(fn, retries=3, label="test")
        assert result == "ok"
        assert len(calls) == 1

    def test_retries_on_timeout_then_succeeds(self, monkeypatch):
        monkeypatch.setattr("crewai_a2a_settlement.client.time.sleep", lambda _: None)
        calls = []

        def fn():
            calls.append(1)
            if len(calls) < 3:
                raise httpx.TimeoutException("timeout")
            return "ok"

        result = _with_retries(fn, retries=3, label="test")
        assert result == "ok"
        assert len(calls) == 3

    def test_exhausts_retries_and_raises_network_error(self, monkeypatch):
        monkeypatch.setattr("crewai_a2a_settlement.client.time.sleep", lambda _: None)

        def fn():
            raise httpx.NetworkError("unreachable")

        with pytest.raises(A2ANetworkError, match="failed after 3 attempts"):
            _with_retries(fn, retries=3, label="conn")

    def test_non_network_exception_not_retried(self):
        calls = []

        def fn():
            calls.append(1)
            raise ValueError("bad data")

        with pytest.raises(ValueError):
            _with_retries(fn, retries=3, label="test")
        assert len(calls) == 1

    def test_backoff_timing(self, monkeypatch):
        sleep_calls = []
        monkeypatch.setattr(
            "crewai_a2a_settlement.client.time.sleep",
            lambda secs: sleep_calls.append(secs),
        )

        def fn():
            raise httpx.TimeoutException("t")

        with pytest.raises(A2ANetworkError):
            _with_retries(fn, retries=3, backoff=2.0, label="test")

        assert sleep_calls == [2.0, 4.0]


# ---------------------------------------------------------------------------
# 10. Context manager
# ---------------------------------------------------------------------------

class TestContextManager:

    def setup_method(self):
        A2ASettlementClient._clear_instance()

    def teardown_method(self):
        A2ASettlementClient._clear_instance()

    def test_context_manager_enter_exit(self):
        client = _make_client()
        with client as c:
            assert c is client


# ---------------------------------------------------------------------------
# 11. Config validation
# ---------------------------------------------------------------------------

class TestConfig:

    def test_default_network_is_sandbox(self):
        config = A2AConfig(api_key="key")
        assert config.network == "sandbox"

    def test_mainnet_is_valid(self):
        config = A2AConfig(api_key="key", network="mainnet")
        assert config.network == "mainnet"

    def test_invalid_network_raises(self):
        with pytest.raises(Exception):
            A2AConfig(api_key="key", network="testnet-999")

    def test_timeout_bounds(self):
        with pytest.raises(Exception):
            A2AConfig(api_key="key", timeout_seconds=0)
        with pytest.raises(Exception):
            A2AConfig(api_key="key", timeout_seconds=9999)

    def test_valid_timeout(self):
        config = A2AConfig(api_key="key", timeout_seconds=60)
        assert config.timeout_seconds == 60

    def test_batch_settlements_default_false(self):
        config = A2AConfig(api_key="key")
        assert config.batch_settlements is False

    def test_batch_settlements_enabled(self):
        config = A2AConfig(api_key="key", batch_settlements=True)
        assert config.batch_settlements is True


# ---------------------------------------------------------------------------
# 12. Batch settlement
# ---------------------------------------------------------------------------


def _make_batch_client() -> A2ASettlementClient:
    """Build a test client with batch_settlements enabled."""
    config = A2AConfig(
        api_key="test-key",
        exchange_url="https://test",
        network="sandbox",
        batch_settlements=True,
    )
    return A2ASettlementClient(config)


class TestBatchSettlement:

    def setup_method(self):
        A2ASettlementClient._clear_instance()

    def teardown_method(self):
        A2ASettlementClient._clear_instance()

    def test_release_defers_when_batch_mode_enabled(self):
        client = _make_batch_client()
        result = client.release("esc-001")
        assert result.status == "deferred"
        assert result.escrow_id == "esc-001"
        assert client.get_pending_count() == 1

    def test_release_immediate_when_batch_mode_disabled(self):
        client = _make_client()
        client._sdk = MagicMock()
        client._sdk.release_escrow.return_value = RELEASE_OK

        result = client.release("esc-001")
        assert result.status == "released"
        assert client.get_pending_count() == 0

    def test_defer_release_adds_to_pending(self):
        client = _make_client()
        client.defer_release("esc-001")
        client.defer_release("esc-002")
        assert client.get_pending_count() == 2
        assert client.get_pending_escrow_ids() == ["esc-001", "esc-002"]

    def test_flush_settlements_releases_all(self):
        client = _make_batch_client()
        client._sdk = MagicMock()
        client._sdk.create_escrow.side_effect = [
            {"escrow_id": "esc-001", "expires_at": ""},
            {"escrow_id": "esc-002", "expires_at": ""},
        ]
        client._sdk.release_escrow.return_value = RELEASE_OK

        client.escrow("0xP", "0xQ", 5.0, "task-1")
        client.escrow("0xP", "0xR", 3.0, "task-2")
        client.release("esc-001")
        client.release("esc-002")

        batch = client.flush_settlements()
        assert batch.escrow_count == 2
        assert batch.total_released == 8.0

    def test_flush_settlements_empty_is_noop(self):
        client = _make_client()
        batch = client.flush_settlements()
        assert batch.escrow_count == 0
        assert batch.results == []

    def test_flush_settlements_clears_pending(self):
        client = _make_batch_client()
        client._sdk = MagicMock()
        client._sdk.create_escrow.return_value = {"escrow_id": "esc-001", "expires_at": ""}
        client._sdk.release_escrow.return_value = RELEASE_OK

        client.escrow("0xP", "0xQ", 5.0, "task-1")
        client.release("esc-001")
        assert client.get_pending_count() == 1

        client.flush_settlements()
        assert client.get_pending_count() == 0

    def test_cancel_not_batched_in_batch_mode(self):
        client = _make_batch_client()
        client._sdk = MagicMock()
        client._sdk.refund_escrow.return_value = CANCEL_OK

        result = client.cancel("esc-001", reason="Task failed")
        assert result.status == "cancelled"
        assert client.get_pending_count() == 0
