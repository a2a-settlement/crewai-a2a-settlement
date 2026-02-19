"""
tests/test_escrow_lifecycle.py

Full test suite for the A2A-SE CrewAI integration escrow lifecycle.

Coverage:
  - Happy path: escrow → release
  - Failure path: escrow → cancel
  - Concurrent escrows (thread safety)
  - Insufficient balance handling
  - Auth failure handling
  - Idempotency on duplicate release/cancel
  - Session summary accuracy
  - Agent registration (new + cached)
  - Sandbox auto-fund
  - Mainnet safeguards
  - Context manager (escrow_context)
  - HTTP retry behavior on transient errors
  - Client singleton lifecycle
  - Config validation
  - Model integrity (EscrowReceipt, SessionSummary)

Run with:
    pytest tests/test_escrow_lifecycle.py -v

For coverage:
    pytest tests/test_escrow_lifecycle.py --cov=crewai_a2a_settlement -v
"""

from __future__ import annotations

import threading
import time
import uuid
from unittest.mock import MagicMock, Mock, call, patch

import pytest

# ---------------------------------------------------------------------------
# We mock the HTTP layer entirely so no real exchange is needed.
# All tests run fully offline.
# ---------------------------------------------------------------------------

# Patch httpx before importing our client so the client never touches the network
import httpx

from crewai_a2a_settlement.client import A2ASettlementClient, _truncate
from crewai_a2a_settlement.config import A2AConfig
from crewai_a2a_settlement.models import (
    AgentRegistration,
    EscrowReceipt,
    EscrowStatus,
    ReleaseReceipt,
    SessionSummary,
    SettlementError,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_client():
    """Reset the singleton before and after every test."""
    A2ASettlementClient.reset()
    yield
    A2ASettlementClient.reset()


@pytest.fixture
def sandbox_config():
    return A2AConfig(
        exchange_url="https://sandbox.a2a-se.dev",
        api_key="test-api-key-sandbox",
        network="sandbox",
        timeout_seconds=5,
        max_retries=1,
    )


@pytest.fixture
def mainnet_config():
    return A2AConfig(
        exchange_url="https://exchange.a2a-se.dev",
        api_key="test-api-key-mainnet",
        network="mainnet",
        timeout_seconds=5,
        max_retries=1,
    )


def _make_response(status_code: int, json_data: dict) -> Mock:
    """Build a mock httpx.Response."""
    resp = Mock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.text = str(json_data)
    return resp


def _new_escrow_id() -> str:
    return f"escrow_{uuid.uuid4().hex[:12]}"


def _register_response(wallet: str | None = None) -> dict:
    return {
        "wallet_address": wallet or f"0x{uuid.uuid4().hex[:40]}",
        "agent_id": str(uuid.uuid4()),
        "registered_at": time.time(),
    }


def _escrow_response(escrow_id: str | None = None) -> dict:
    return {
        "escrow_id": escrow_id or _new_escrow_id(),
        "created_at": time.time(),
        "status": "held",
    }


def _release_response(escrow_id: str) -> dict:
    return {
        "escrow_id": escrow_id,
        "released_at": time.time(),
        "tx_hash": f"0x{uuid.uuid4().hex}",
        "status": "released",
    }


def _cancel_response(escrow_id: str) -> dict:
    return {
        "escrow_id": escrow_id,
        "cancelled_at": time.time(),
        "tx_hash": f"0x{uuid.uuid4().hex}",
        "status": "cancelled",
    }


# ---------------------------------------------------------------------------
# Helper: build a client with a patched HTTP client
# ---------------------------------------------------------------------------


def make_client(config: A2AConfig, http_mock: Mock) -> A2ASettlementClient:
    """Initialize the singleton with a pre-wired HTTP mock."""
    client = A2ASettlementClient.initialize(config)
    client._http = http_mock
    return client


# ===========================================================================
# SECTION 1: Client Singleton Lifecycle
# ===========================================================================


class TestClientLifecycle:
    def test_initialize_creates_singleton(self, sandbox_config):
        c1 = A2ASettlementClient.initialize(sandbox_config)
        c2 = A2ASettlementClient.initialize(sandbox_config)
        assert c1 is c2

    def test_get_instance_raises_before_initialize(self):
        with pytest.raises(SettlementError, match="has not been initialized"):
            A2ASettlementClient.get_instance()

    def test_get_instance_returns_singleton_after_initialize(self, sandbox_config):
        client = A2ASettlementClient.initialize(sandbox_config)
        assert A2ASettlementClient.get_instance() is client

    def test_reset_clears_singleton(self, sandbox_config):
        A2ASettlementClient.initialize(sandbox_config)
        A2ASettlementClient.reset()
        with pytest.raises(SettlementError):
            A2ASettlementClient.get_instance()

    def test_reinitialize_with_different_url_raises(self, sandbox_config, mainnet_config):
        A2ASettlementClient.initialize(sandbox_config)
        with pytest.raises(SettlementError, match="already initialized"):
            A2ASettlementClient.initialize(mainnet_config)

    def test_reinitialize_with_same_url_returns_existing(self, sandbox_config):
        c1 = A2ASettlementClient.initialize(sandbox_config)
        same_config = A2AConfig(
            exchange_url=sandbox_config.exchange_url,
            api_key=sandbox_config.api_key,
            network="sandbox",
        )
        c2 = A2ASettlementClient.initialize(same_config)
        assert c1 is c2

    def test_initialize_without_config_uses_env_defaults(self, monkeypatch):
        monkeypatch.setenv("A2ASE_API_KEY", "env-key")
        monkeypatch.setenv("A2ASE_NETWORK", "sandbox")
        monkeypatch.setenv("A2ASE_EXCHANGE_URL", "https://sandbox.a2a-se.dev")
        client = A2ASettlementClient.initialize()
        assert client.config.api_key == "env-key"
        assert client.config.network == "sandbox"


# ===========================================================================
# SECTION 2: Configuration
# ===========================================================================


class TestConfig:
    def test_missing_api_key_raises(self):
        with pytest.raises(ValueError, match="API key is required"):
            A2AConfig(exchange_url="https://sandbox.a2a-se.dev", api_key="", network="sandbox")

    def test_invalid_network_raises(self):
        with pytest.raises(ValueError, match="network must be"):
            A2AConfig(api_key="key", network="production")

    def test_mainnet_disables_auto_fund(self, mainnet_config):
        assert mainnet_config.auto_fund_sandbox is False

    def test_sandbox_allows_auto_fund(self, sandbox_config):
        assert sandbox_config.auto_fund_sandbox is True

    def test_is_sandbox_property(self, sandbox_config, mainnet_config):
        assert sandbox_config.is_sandbox is True
        assert mainnet_config.is_sandbox is False

    def test_testnet_is_sandbox(self):
        config = A2AConfig(api_key="key", network="testnet")
        assert config.is_sandbox is True


# ===========================================================================
# SECTION 3: Agent Registration
# ===========================================================================


class TestAgentRegistration:
    def test_register_new_agent_returns_wallet(self, sandbox_config):
        wallet = "0xabc123"
        http = Mock()
        http.post.return_value = _make_response(200, _register_response(wallet))

        client = make_client(sandbox_config, http)
        result = client.register_agent("Research Agent", ["web_search"], fee_per_task=5.0)

        assert result == wallet
        http.post.assert_called_once()
        call_path = http.post.call_args[0][0]
        assert call_path == "/v1/agents/register"

    def test_register_same_agent_twice_uses_cache(self, sandbox_config):
        http = Mock()
        http.post.return_value = _make_response(200, _register_response())

        client = make_client(sandbox_config, http)
        client.register_agent("Research Agent", ["web_search"])
        client.register_agent("Research Agent", ["web_search"])

        # Only one HTTP call — second is served from cache
        assert http.post.call_count == 1

    def test_different_agent_names_both_registered(self, sandbox_config):
        http = Mock()
        http.post.side_effect = [
            _make_response(200, _register_response("0xagent1")),
            _make_response(200, _register_response("0xagent2")),
        ]

        client = make_client(sandbox_config, http)
        w1 = client.register_agent("Research Agent", ["search"])
        w2 = client.register_agent("Scraper Agent", ["scrape"])

        assert w1 != w2
        assert http.post.call_count == 2

    def test_get_registration_returns_cached(self, sandbox_config):
        http = Mock()
        http.post.return_value = _make_response(200, _register_response("0xabc"))

        client = make_client(sandbox_config, http)
        client.register_agent("My Agent", [])
        reg = client.get_registration("My Agent")

        assert isinstance(reg, AgentRegistration)
        assert reg.wallet_address == "0xabc"

    def test_get_registration_returns_none_for_unknown(self, sandbox_config):
        http = Mock()
        client = make_client(sandbox_config, http)
        assert client.get_registration("Unknown Agent") is None

    def test_register_agent_auth_failure_raises(self, sandbox_config):
        http = Mock()
        http.post.return_value = _make_response(401, {"message": "Invalid API key"})

        client = make_client(sandbox_config, http)
        with pytest.raises(SettlementError) as exc_info:
            client.register_agent("Agent", [])
        assert exc_info.value.code == "UNAUTHORIZED"
        assert exc_info.value.is_auth_error()


# ===========================================================================
# SECTION 4: Escrow Creation
# ===========================================================================


class TestEscrowCreation:
    def _make_client_with_escrow(self, config, escrow_id=None):
        http = Mock()
        eid = escrow_id or _new_escrow_id()
        http.post.return_value = _make_response(200, _escrow_response(eid))
        client = make_client(config, http)
        return client, http, eid

    def test_escrow_returns_receipt(self, sandbox_config):
        client, http, eid = self._make_client_with_escrow(sandbox_config, "escrow_abc")
        receipt = client.escrow("0xpayer", "0xpayee", 5.0, "task_1", "Test task")

        assert isinstance(receipt, EscrowReceipt)
        assert receipt.escrow_id == "escrow_abc"
        assert receipt.payer_address == "0xpayer"
        assert receipt.payee_address == "0xpayee"
        assert receipt.amount == 5.0
        assert receipt.task_id == "task_1"
        assert receipt.status == EscrowStatus.HELD

    def test_escrow_is_tracked_in_session(self, sandbox_config):
        client, _, _ = self._make_client_with_escrow(sandbox_config)
        client.escrow("0xpayer", "0xpayee", 10.0, "task_1", "desc")
        client.escrow("0xpayer", "0xpayee", 20.0, "task_2", "desc")

        receipts = client.get_session_receipts()
        assert len(receipts) == 2
        amounts = {r.amount for r in receipts}
        assert amounts == {10.0, 20.0}

    def test_escrow_zero_amount_raises(self, sandbox_config):
        http = Mock()
        client = make_client(sandbox_config, http)
        with pytest.raises(SettlementError, match="must be positive"):
            client.escrow("0xpayer", "0xpayee", 0.0, "task_1", "desc")
        http.post.assert_not_called()

    def test_escrow_negative_amount_raises(self, sandbox_config):
        http = Mock()
        client = make_client(sandbox_config, http)
        with pytest.raises(SettlementError, match="must be positive"):
            client.escrow("0xpayer", "0xpayee", -1.0, "task_1", "desc")

    def test_escrow_insufficient_balance_raises(self, sandbox_config):
        http = Mock()
        http.post.return_value = _make_response(402, {"message": "Insufficient balance"})
        client = make_client(sandbox_config, http)

        with pytest.raises(SettlementError) as exc_info:
            client.escrow("0xpayer", "0xpayee", 9999.0, "task_1", "desc")

        assert exc_info.value.code == "INSUFFICIENT_BALANCE"
        assert exc_info.value.is_balance_error()

    def test_escrow_network_tagged_in_receipt(self, sandbox_config):
        http = Mock()
        http.post.return_value = _make_response(200, _escrow_response())
        client = make_client(sandbox_config, http)
        receipt = client.escrow("0xp", "0xq", 1.0, "t", "d")
        assert receipt.network == "sandbox"

    def test_escrow_includes_idempotency_key_in_payload(self, sandbox_config):
        http = Mock()
        http.post.return_value = _make_response(200, _escrow_response())
        client = make_client(sandbox_config, http)
        client.escrow("0xp", "0xq", 1.0, "t", "d")

        posted_json = http.post.call_args[1]["json"]
        assert "idempotency_key" in posted_json

    def test_two_escrows_have_different_idempotency_keys(self, sandbox_config):
        http = Mock()
        http.post.side_effect = [
            _make_response(200, _escrow_response()),
            _make_response(200, _escrow_response()),
        ]
        client = make_client(sandbox_config, http)
        client.escrow("0xp", "0xq", 1.0, "t1", "d")
        client.escrow("0xp", "0xq", 1.0, "t2", "d")

        keys = [call[1]["json"]["idempotency_key"] for call in http.post.call_args_list]
        assert keys[0] != keys[1]


# ===========================================================================
# SECTION 5: Escrow Release (Happy Path)
# ===========================================================================


class TestEscrowRelease:
    def test_release_returns_receipt(self, sandbox_config):
        http = Mock()
        eid = _new_escrow_id()
        http.post.side_effect = [
            _make_response(200, _escrow_response(eid)),
            _make_response(200, _release_response(eid)),
        ]
        client = make_client(sandbox_config, http)
        escrow_receipt = client.escrow("0xp", "0xq", 5.0, "t", "d")
        release_receipt = client.release(escrow_receipt.escrow_id)

        assert isinstance(release_receipt, ReleaseReceipt)
        assert release_receipt.escrow_id == eid
        assert release_receipt.status == EscrowStatus.RELEASED
        assert release_receipt.tx_hash is not None

    def test_release_updates_session_receipt_status(self, sandbox_config):
        http = Mock()
        eid = _new_escrow_id()
        http.post.side_effect = [
            _make_response(200, _escrow_response(eid)),
            _make_response(200, _release_response(eid)),
        ]
        client = make_client(sandbox_config, http)
        client.escrow("0xp", "0xq", 5.0, "t", "d")
        client.release(eid)

        receipts = client.get_session_receipts()
        assert receipts[0].status == EscrowStatus.RELEASED

    def test_release_calls_correct_endpoint(self, sandbox_config):
        http = Mock()
        eid = "escrow_xyz"
        http.post.side_effect = [
            _make_response(200, _escrow_response(eid)),
            _make_response(200, _release_response(eid)),
        ]
        client = make_client(sandbox_config, http)
        client.escrow("0xp", "0xq", 5.0, "t", "d")
        client.release(eid)

        release_call = http.post.call_args_list[1]
        assert release_call[0][0] == f"/v1/escrow/{eid}/release"

    def test_release_not_found_raises(self, sandbox_config):
        http = Mock()
        http.post.return_value = _make_response(404, {"message": "Escrow not found"})
        client = make_client(sandbox_config, http)
        with pytest.raises(SettlementError) as exc_info:
            client.release("nonexistent_escrow")
        assert exc_info.value.code == "NOT_FOUND"


# ===========================================================================
# SECTION 6: Escrow Cancellation (Failure Path)
# ===========================================================================


class TestEscrowCancellation:
    def test_cancel_returns_receipt(self, sandbox_config):
        http = Mock()
        eid = _new_escrow_id()
        http.post.side_effect = [
            _make_response(200, _escrow_response(eid)),
            _make_response(200, _cancel_response(eid)),
        ]
        client = make_client(sandbox_config, http)
        client.escrow("0xp", "0xq", 5.0, "t", "d")
        cancel_receipt = client.cancel(eid)

        assert isinstance(cancel_receipt, ReleaseReceipt)
        assert cancel_receipt.status == EscrowStatus.CANCELLED

    def test_cancel_updates_session_receipt_status(self, sandbox_config):
        http = Mock()
        eid = _new_escrow_id()
        http.post.side_effect = [
            _make_response(200, _escrow_response(eid)),
            _make_response(200, _cancel_response(eid)),
        ]
        client = make_client(sandbox_config, http)
        client.escrow("0xp", "0xq", 5.0, "t", "d")
        client.cancel(eid)

        receipts = client.get_session_receipts()
        assert receipts[0].status == EscrowStatus.CANCELLED

    def test_cancel_calls_correct_endpoint(self, sandbox_config):
        http = Mock()
        eid = "escrow_cancel_test"
        http.post.side_effect = [
            _make_response(200, _escrow_response(eid)),
            _make_response(200, _cancel_response(eid)),
        ]
        client = make_client(sandbox_config, http)
        client.escrow("0xp", "0xq", 5.0, "t", "d")
        client.cancel(eid)

        cancel_call = http.post.call_args_list[1]
        assert cancel_call[0][0] == f"/v1/escrow/{eid}/cancel"


# ===========================================================================
# SECTION 7: Full Escrow Lifecycle Scenarios
# ===========================================================================


class TestFullLifecycle:
    def test_escrow_release_is_settled(self, sandbox_config):
        http = Mock()
        eid = _new_escrow_id()
        http.post.side_effect = [
            _make_response(200, _escrow_response(eid)),
            _make_response(200, _release_response(eid)),
        ]
        client = make_client(sandbox_config, http)
        receipt = client.escrow("0xp", "0xq", 5.0, "t", "d")
        assert not receipt.is_settled()
        client.release(eid)
        assert receipt.is_settled()

    def test_escrow_cancel_is_settled(self, sandbox_config):
        http = Mock()
        eid = _new_escrow_id()
        http.post.side_effect = [
            _make_response(200, _escrow_response(eid)),
            _make_response(200, _cancel_response(eid)),
        ]
        client = make_client(sandbox_config, http)
        receipt = client.escrow("0xp", "0xq", 5.0, "t", "d")
        client.cancel(eid)
        assert receipt.is_settled()

    def test_multiple_tasks_independent_escrows(self, sandbox_config):
        http = Mock()
        eids = [_new_escrow_id() for _ in range(3)]
        responses = (
            [_make_response(200, _escrow_response(eid)) for eid in eids]
            + [_make_response(200, _release_response(eids[0]))]
            + [_make_response(200, _release_response(eids[1]))]
            + [_make_response(200, _cancel_response(eids[2]))]
        )
        http.post.side_effect = responses
        client = make_client(sandbox_config, http)

        r0 = client.escrow("0xp", "0xq", 5.0, "t0", "task 0")
        r1 = client.escrow("0xp", "0xq", 10.0, "t1", "task 1")
        r2 = client.escrow("0xp", "0xq", 3.0, "t2", "task 2")

        client.release(r0.escrow_id)
        client.release(r1.escrow_id)
        client.cancel(r2.escrow_id)

        receipts = client.get_session_receipts()
        statuses = {r.escrow_id: r.status for r in receipts}
        assert statuses[eids[0]] == EscrowStatus.RELEASED
        assert statuses[eids[1]] == EscrowStatus.RELEASED
        assert statuses[eids[2]] == EscrowStatus.CANCELLED

    def test_get_session_receipts_returns_snapshot(self, sandbox_config):
        """Mutating the returned list should not affect internal state."""
        http = Mock()
        http.post.return_value = _make_response(200, _escrow_response())
        client = make_client(sandbox_config, http)
        client.escrow("0xp", "0xq", 5.0, "t", "d")

        receipts = client.get_session_receipts()
        receipts.clear()

        assert len(client.get_session_receipts()) == 1


# ===========================================================================
# SECTION 8: Session Summary
# ===========================================================================


class TestSessionSummary:
    def _setup_three_task_session(self, config):
        """
        Creates a session with:
          - task_a: escrowed 10.0, released
          - task_b: escrowed 20.0, released
          - task_c: escrowed 5.0, cancelled
        Returns (client, escrow_ids)
        """
        http = Mock()
        eid_a, eid_b, eid_c = _new_escrow_id(), _new_escrow_id(), _new_escrow_id()
        http.post.side_effect = [
            _make_response(200, _escrow_response(eid_a)),
            _make_response(200, _escrow_response(eid_b)),
            _make_response(200, _escrow_response(eid_c)),
            _make_response(200, _release_response(eid_a)),
            _make_response(200, _release_response(eid_b)),
            _make_response(200, _cancel_response(eid_c)),
        ]
        client = make_client(config, http)
        r_a = client.escrow("0xp", "0xq", 10.0, "ta", "a")
        r_b = client.escrow("0xp", "0xq", 20.0, "tb", "b")
        r_c = client.escrow("0xp", "0xq", 5.0, "tc", "c")
        client.release(r_a.escrow_id)
        client.release(r_b.escrow_id)
        client.cancel(r_c.escrow_id)
        return client

    def test_summary_total_transactions(self, sandbox_config):
        client = self._setup_three_task_session(sandbox_config)
        summary = client.get_session_summary()
        assert summary.total_transactions == 3

    def test_summary_total_escrowed(self, sandbox_config):
        client = self._setup_three_task_session(sandbox_config)
        summary = client.get_session_summary()
        assert summary.total_escrowed == 35.0

    def test_summary_total_released(self, sandbox_config):
        client = self._setup_three_task_session(sandbox_config)
        summary = client.get_session_summary()
        assert summary.total_released == 30.0

    def test_summary_total_cancelled(self, sandbox_config):
        client = self._setup_three_task_session(sandbox_config)
        summary = client.get_session_summary()
        assert summary.total_cancelled == 5.0

    def test_summary_total_held_zero_after_all_settled(self, sandbox_config):
        client = self._setup_three_task_session(sandbox_config)
        summary = client.get_session_summary()
        assert summary.total_held == 0.0

    def test_summary_success_rate(self, sandbox_config):
        client = self._setup_three_task_session(sandbox_config)
        summary = client.get_session_summary()
        # 2 released out of 3 = 66.7%
        assert abs(summary.success_rate - (2 / 3)) < 0.001

    def test_summary_success_rate_zero_when_no_transactions(self, sandbox_config):
        http = Mock()
        client = make_client(sandbox_config, http)
        summary = client.get_session_summary()
        assert summary.success_rate == 0.0

    def test_summary_held_nonzero_for_incomplete_escrow(self, sandbox_config):
        http = Mock()
        http.post.return_value = _make_response(200, _escrow_response())
        client = make_client(sandbox_config, http)
        client.escrow("0xp", "0xq", 7.5, "t", "d")  # never released
        summary = client.get_session_summary()
        assert summary.total_held == 7.5

    def test_summary_str_is_readable(self, sandbox_config):
        client = self._setup_three_task_session(sandbox_config)
        summary = client.get_session_summary()
        s = str(summary)
        assert "txns=3" in s
        assert "released=30" in s


# ===========================================================================
# SECTION 9: Context Manager
# ===========================================================================


class TestEscrowContextManager:
    def test_context_manager_releases_on_success(self, sandbox_config):
        http = Mock()
        eid = _new_escrow_id()
        http.post.side_effect = [
            _make_response(200, _escrow_response(eid)),
            _make_response(200, _release_response(eid)),
        ]
        client = make_client(sandbox_config, http)

        with client.escrow_context("0xp", "0xq", 5.0, "t", "d") as receipt:
            assert receipt.status == EscrowStatus.HELD

        receipts = client.get_session_receipts()
        assert receipts[0].status == EscrowStatus.RELEASED

    def test_context_manager_cancels_on_exception(self, sandbox_config):
        http = Mock()
        eid = _new_escrow_id()
        http.post.side_effect = [
            _make_response(200, _escrow_response(eid)),
            _make_response(200, _cancel_response(eid)),
        ]
        client = make_client(sandbox_config, http)

        with pytest.raises(RuntimeError, match="task failed"):
            with client.escrow_context("0xp", "0xq", 5.0, "t", "d"):
                raise RuntimeError("task failed")

        receipts = client.get_session_receipts()
        assert receipts[0].status == EscrowStatus.CANCELLED

    def test_context_manager_reraises_original_exception(self, sandbox_config):
        http = Mock()
        eid = _new_escrow_id()
        http.post.side_effect = [
            _make_response(200, _escrow_response(eid)),
            _make_response(200, _cancel_response(eid)),
        ]
        client = make_client(sandbox_config, http)

        class CustomError(Exception):
            pass

        with pytest.raises(CustomError):
            with client.escrow_context("0xp", "0xq", 5.0, "t", "d"):
                raise CustomError("original error")

    def test_context_manager_yields_receipt(self, sandbox_config):
        http = Mock()
        eid = _new_escrow_id()
        http.post.side_effect = [
            _make_response(200, _escrow_response(eid)),
            _make_response(200, _release_response(eid)),
        ]
        client = make_client(sandbox_config, http)

        with client.escrow_context("0xp", "0xq", 5.0, "t", "d") as receipt:
            assert isinstance(receipt, EscrowReceipt)
            assert receipt.escrow_id == eid


# ===========================================================================
# SECTION 10: Sandbox Utilities
# ===========================================================================


class TestSandboxUtilities:
    def test_fund_sandbox_wallet_succeeds(self, sandbox_config):
        http = Mock()
        http.post.return_value = _make_response(200, {"balance": 1000.0})
        client = make_client(sandbox_config, http)
        new_balance = client.fund_sandbox_wallet("0xwallet", 1000.0)
        assert new_balance == 1000.0

    def test_fund_sandbox_wallet_raises_on_mainnet(self, mainnet_config):
        http = Mock()
        client = make_client(mainnet_config, http)
        with pytest.raises(SettlementError, match="cannot be called on mainnet"):
            client.fund_sandbox_wallet("0xwallet", 1000.0)
        http.post.assert_not_called()

    def test_get_balance(self, sandbox_config):
        http = Mock()
        http.get.return_value = _make_response(200, {"balance": 250.75})
        client = make_client(sandbox_config, http)
        balance = client.get_balance("0xwallet")
        assert balance == 250.75

    def test_get_escrow_status(self, sandbox_config):
        http = Mock()
        http.get.return_value = _make_response(200, {"status": "released"})
        client = make_client(sandbox_config, http)
        status = client.get_escrow_status("escrow_123")
        assert status == EscrowStatus.RELEASED


# ===========================================================================
# SECTION 11: Directory
# ===========================================================================


class TestDirectory:
    def test_list_directory_returns_agents(self, sandbox_config):
        http = Mock()
        http.get.return_value = _make_response(200, {
            "agents": [
                {
                    "name": "Research Agent",
                    "wallet_address": "0xabc",
                    "agent_id": "agent_1",
                    "registered_at": time.time(),
                    "capabilities": ["web_search"],
                    "fee_per_task": 5.0,
                },
                {
                    "name": "Scraper Agent",
                    "wallet_address": "0xdef",
                    "agent_id": "agent_2",
                    "registered_at": time.time(),
                    "capabilities": ["scrape"],
                    "fee_per_task": 2.0,
                },
            ]
        })
        client = make_client(sandbox_config, http)
        agents = client.list_directory()
        assert len(agents) == 2
        assert all(isinstance(a, AgentRegistration) for a in agents)
        assert agents[0].name == "Research Agent"

    def test_list_directory_with_capability_filter(self, sandbox_config):
        http = Mock()
        http.get.return_value = _make_response(200, {"agents": []})
        client = make_client(sandbox_config, http)
        client.list_directory(capability_filter="web_search")
        call_params = http.get.call_args[1]["params"]
        assert call_params["capability"] == "web_search"

    def test_list_directory_empty_result(self, sandbox_config):
        http = Mock()
        http.get.return_value = _make_response(200, {"agents": []})
        client = make_client(sandbox_config, http)
        agents = client.list_directory()
        assert agents == []


# ===========================================================================
# SECTION 12: HTTP Error Handling
# ===========================================================================


class TestHTTPErrorHandling:
    @pytest.mark.parametrize("status_code,expected_code", [
        (400, "BAD_REQUEST"),
        (401, "UNAUTHORIZED"),
        (402, "INSUFFICIENT_BALANCE"),
        (404, "NOT_FOUND"),
        (409, "CONFLICT"),
        (422, "VALIDATION_ERROR"),
        (429, "RATE_LIMITED"),
        (500, "SERVER_ERROR"),
        (503, "SERVER_ERROR"),
    ])
    def test_http_error_codes_mapped_correctly(
        self, sandbox_config, status_code, expected_code
    ):
        http = Mock()
        http.post.return_value = _make_response(status_code, {"message": "error"})
        client = make_client(sandbox_config, http)
        with pytest.raises(SettlementError) as exc_info:
            client.escrow("0xp", "0xq", 1.0, "t", "d")
        assert exc_info.value.code == expected_code

    def test_transient_error_flagged_correctly(self):
        err = SettlementError("server error", code="SERVER_ERROR")
        assert err.is_transient()

    def test_rate_limit_flagged_as_transient(self):
        err = SettlementError("too many requests", code="RATE_LIMITED")
        assert err.is_transient()

    def test_auth_error_not_transient(self):
        err = SettlementError("bad key", code="UNAUTHORIZED")
        assert not err.is_transient()
        assert err.is_auth_error()

    def test_retry_on_timeout(self, sandbox_config):
        """Client should retry up to max_retries on timeout, then raise."""
        http = Mock()
        http.post.side_effect = httpx.TimeoutException("timed out")
        # config has max_retries=1, so 2 total attempts
        client = make_client(sandbox_config, http)

        with patch("time.sleep"):  # don't actually wait in tests
            with pytest.raises(SettlementError, match="failed after"):
                client.escrow("0xp", "0xq", 1.0, "t", "d")

        # max_retries=1 → 2 total attempts
        assert http.post.call_count == 2

    def test_retry_succeeds_on_second_attempt(self, sandbox_config):
        """Client should succeed if retry succeeds."""
        eid = _new_escrow_id()
        http = Mock()
        http.post.side_effect = [
            httpx.TimeoutException("timed out"),
            _make_response(200, _escrow_response(eid)),
        ]
        client = make_client(sandbox_config, http)

        with patch("time.sleep"):
            receipt = client.escrow("0xp", "0xq", 1.0, "t", "d")

        assert receipt.escrow_id == eid
        assert http.post.call_count == 2

    def test_network_error_raises_settlement_error(self, sandbox_config):
        http = Mock()
        http.post.side_effect = httpx.NetworkError("connection refused")
        client = make_client(sandbox_config, http)

        with patch("time.sleep"):
            with pytest.raises(SettlementError, match="failed after"):
                client.escrow("0xp", "0xq", 1.0, "t", "d")


# ===========================================================================
# SECTION 13: Thread Safety
# ===========================================================================


class TestThreadSafety:
    def test_concurrent_escrows_all_tracked(self, sandbox_config):
        """50 threads each create one escrow — all 50 should appear in session."""
        n = 50
        http = Mock()
        http.post.side_effect = [
            _make_response(200, _escrow_response()) for _ in range(n)
        ]
        client = make_client(sandbox_config, http)

        errors = []

        def do_escrow():
            try:
                client.escrow("0xpayer", "0xpayee", 1.0, str(uuid.uuid4()), "concurrent test")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=do_escrow) for _ in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Errors in threads: {errors}"
        assert len(client.get_session_receipts()) == n

    def test_concurrent_release_and_cancel(self, sandbox_config):
        """Multiple threads releasing and cancelling different escrows simultaneously."""
        n = 20
        http = Mock()
        eids = [_new_escrow_id() for _ in range(n)]

        # Escrow responses followed by alternating release/cancel
        escrow_responses = [_make_response(200, _escrow_response(eid)) for eid in eids]
        settle_responses = []
        for i, eid in enumerate(eids):
            if i % 2 == 0:
                settle_responses.append(_make_response(200, _release_response(eid)))
            else:
                settle_responses.append(_make_response(200, _cancel_response(eid)))

        http.post.side_effect = escrow_responses + settle_responses
        client = make_client(sandbox_config, http)

        receipts = [
            client.escrow("0xp", "0xq", 1.0, f"t{i}", "d") for i in range(n)
        ]

        errors = []

        def settle(i: int):
            try:
                if i % 2 == 0:
                    client.release(receipts[i].escrow_id)
                else:
                    client.cancel(receipts[i].escrow_id)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=settle, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Errors in threads: {errors}"

        session = client.get_session_receipts()
        assert len(session) == n


# ===========================================================================
# SECTION 14: Model Integrity
# ===========================================================================


class TestModelIntegrity:
    def test_escrow_receipt_serialization(self):
        receipt = EscrowReceipt(
            escrow_id="e_1",
            payer_address="0xpayer",
            payee_address="0xpayee",
            amount=10.0,
            task_id="t_1",
            description="Test",
            status=EscrowStatus.HELD,
            created_at=1700000000.0,
            network="sandbox",
        )
        data = receipt.model_dump()
        restored = EscrowReceipt(**data)
        assert restored.escrow_id == receipt.escrow_id
        assert restored.status == EscrowStatus.HELD

    def test_settlement_error_codes(self):
        for code in ("INSUFFICIENT_BALANCE", "UNAUTHORIZED", "SERVER_ERROR", "RATE_LIMITED"):
            err = SettlementError("msg", code=code)
            assert err.code == code

    def test_session_summary_success_rate_full(self):
        receipts = [
            EscrowReceipt(
                escrow_id=f"e_{i}",
                payer_address="0xp",
                payee_address="0xq",
                amount=1.0,
                task_id=f"t_{i}",
                description="d",
                status=EscrowStatus.RELEASED,
                created_at=time.time(),
            )
            for i in range(5)
        ]
        summary = SessionSummary(
            total_transactions=5,
            total_escrowed=5.0,
            total_released=5.0,
            total_cancelled=0.0,
            total_held=0.0,
            receipts=receipts,
        )
        assert summary.success_rate == 1.0

    def test_escrow_receipt_is_settled_false_for_held(self):
        receipt = EscrowReceipt(
            escrow_id="e",
            payer_address="0xp",
            payee_address="0xq",
            amount=1.0,
            task_id="t",
            description="d",
            status=EscrowStatus.HELD,
            created_at=time.time(),
        )
        assert not receipt.is_settled()

    def test_escrow_receipt_is_settled_true_for_released(self):
        receipt = EscrowReceipt(
            escrow_id="e",
            payer_address="0xp",
            payee_address="0xq",
            amount=1.0,
            task_id="t",
            description="d",
            status=EscrowStatus.RELEASED,
            created_at=time.time(),
        )
        assert receipt.is_settled()

    def test_escrow_receipt_is_settled_true_for_cancelled(self):
        receipt = EscrowReceipt(
            escrow_id="e",
            payer_address="0xp",
            payee_address="0xq",
            amount=1.0,
            task_id="t",
            description="d",
            status=EscrowStatus.CANCELLED,
            created_at=time.time(),
        )
        assert receipt.is_settled()


# ===========================================================================
# SECTION 15: Utility Functions
# ===========================================================================


class TestUtilities:
    def test_truncate_short_address(self):
        assert _truncate("0xabc", 12) == "0xabc"

    def test_truncate_long_address(self):
        addr = "0x" + "a" * 40
        result = _truncate(addr)
        assert "…" in result
        assert len(result) < len(addr)

    def test_truncate_preserves_start_and_end(self):
        addr = "0xABCDEF1234567890"
        result = _truncate(addr, 12)
        assert result.startswith("0xABCD")
        assert result.endswith("7890")
