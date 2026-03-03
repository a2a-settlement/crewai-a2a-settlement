"""
shim_client.py -- Security Shim client for CrewAI integration.

Routes CrewAI tool calls through the A2A Settlement Security Shim
for escrow-gated, credential-isolated external API access.

Usage::

    from crewai_a2a_settlement.shim_client import ShimClient

    shim = ShimClient.initialize()
    result = shim.proxy_tool_call(
        escrow_id="escrow-uuid",
        tool_id="github-create-issue",
        body='{"title": "From CrewAI agent"}',
    )
"""

from __future__ import annotations

import logging
from typing import Optional

import httpx

logger = logging.getLogger("crewai_a2a_settlement.shim")


def _env(primary: str, legacy: str, default: str) -> str:
    import os
    return os.getenv(primary) or os.getenv(legacy, default)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ShimError(Exception):
    """Base exception for shim client errors."""


class ShimNotConfiguredError(ShimError):
    """Shim URL is not set."""


class ShimEscrowDepletedError(ShimError):
    """Escrow balance is exhausted (HTTP 402 from shim)."""

    def __init__(self, escrow_id: str, message: str):
        self.escrow_id = escrow_id
        super().__init__(message)


class ShimProxyError(ShimError):
    """Proxy request failed for a non-402 reason."""


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


class ProxyResult:
    """Result of a proxied tool call."""

    __slots__ = ("status_code", "body", "cost_charged", "escrow_remaining", "headers")

    def __init__(self, data: dict):
        self.status_code: int = data.get("status_code", 0)
        self.body: str = data.get("body", "")
        self.cost_charged: float = data.get("cost_charged", 0.0)
        self.escrow_remaining: Optional[float] = data.get("escrow_remaining")
        self.headers: dict = data.get("headers", {})

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 400


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class ShimClient:
    """Client for the A2A Settlement Security Shim.

    Routes tool calls through the shim for escrow-gated, credential-isolated
    external API access. Follows the same singleton pattern as A2ASettlementClient.
    """

    _instance: Optional["ShimClient"] = None

    def __init__(self, shim_url: str, api_key: str = "", timeout: float = 30.0):
        if not shim_url:
            raise ShimNotConfiguredError(
                "Shim URL is not set. Export A2A_SHIM_URL or pass it explicitly."
            )
        self._url = shim_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        logger.info("ShimClient initialized: url=%s", self._url)

    @classmethod
    def initialize(
        cls,
        shim_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
    ) -> "ShimClient":
        """Initialize the singleton ShimClient.

        Reads from environment if not explicitly provided:
            A2A_SHIM_URL -- base URL of the Security Shim
            A2A_SHIM_API_KEY -- falls back to A2A_API_KEY
        """
        resolved_url = shim_url or _env("A2A_SHIM_URL", "A2ASE_SHIM_URL", "")
        resolved_key = api_key or _env("A2A_SHIM_API_KEY", "A2A_API_KEY", "")
        cls._instance = cls(resolved_url, resolved_key, timeout)
        return cls._instance

    @classmethod
    def get_instance(cls) -> "ShimClient":
        if cls._instance is None:
            raise ShimNotConfiguredError(
                "ShimClient not initialized. Call ShimClient.initialize() first, "
                "or set A2A_SHIM_URL in the environment."
            )
        return cls._instance

    @classmethod
    def _clear_instance(cls) -> None:
        cls._instance = None

    # ------------------------------------------------------------------
    # Core proxy
    # ------------------------------------------------------------------

    def proxy_tool_call(
        self,
        escrow_id: str,
        tool_id: Optional[str] = None,
        destination_url: Optional[str] = None,
        method: str = "POST",
        headers: Optional[dict[str, str]] = None,
        body: Optional[str] = None,
        secret_id: Optional[str] = None,
    ) -> ProxyResult:
        """Route a tool call through the Security Shim.

        Args:
            escrow_id: Active escrow funding this call.
            tool_id: Registered tool ID (full air gap mode).
            destination_url: Direct destination URL (developer mode).
            method: HTTP method.
            headers: Extra headers for the outbound request.
            body: Request body.
            secret_id: Secret ID to resolve (direct mode only).

        Returns:
            ProxyResult with the upstream response and cost info.

        Raises:
            ShimEscrowDepletedError: If escrow balance is exhausted (HTTP 402).
            ShimProxyError: If the proxy request fails for another reason.
        """
        payload: dict = {"escrow_id": escrow_id, "method": method}
        if tool_id:
            payload["tool_id"] = tool_id
        if destination_url:
            payload["destination_url"] = destination_url
        if headers:
            payload["headers"] = headers
        if body:
            payload["body"] = body
        if secret_id:
            payload["secret_id"] = secret_id

        data = self._request("POST", "/shim/proxy", payload)
        result = ProxyResult(data)

        if result.status_code == 402:
            raise ShimEscrowDepletedError(
                escrow_id=escrow_id,
                message=f"Escrow {escrow_id} depleted: {result.body}",
            )

        return result

    # ------------------------------------------------------------------
    # Escrow management on shim
    # ------------------------------------------------------------------

    def register_escrow(self, escrow_id: str, amount: int) -> dict:
        """Register an escrow with the shim's local gate."""
        return self._request("POST", "/shim/escrows", {
            "escrow_id": escrow_id,
            "amount": amount,
        })

    def get_escrow_status(self, escrow_id: str) -> dict:
        """Check escrow balance remaining in the shim."""
        return self._request("GET", f"/shim/escrows/{escrow_id}")

    # ------------------------------------------------------------------
    # Tool management
    # ------------------------------------------------------------------

    def register_tool(
        self,
        tool_id: str,
        destination_url: str,
        method: str = "POST",
        secret_id: Optional[str] = None,
        inject_as: str = "bearer",
        inject_key: str = "Authorization",
        cost_override: Optional[float] = None,
        description: str = "",
    ) -> dict:
        """Register a tool mapping on the shim."""
        payload: dict = {
            "tool_id": tool_id,
            "destination_url": destination_url,
            "method": method,
            "inject_as": inject_as,
            "inject_key": inject_key,
            "description": description,
        }
        if secret_id:
            payload["secret_id"] = secret_id
        if cost_override is not None:
            payload["cost_override"] = cost_override
        return self._request("POST", "/shim/tools", payload)

    def list_tools(self) -> dict:
        """List registered tools on the shim."""
        return self._request("GET", "/shim/tools")

    # ------------------------------------------------------------------
    # Audit
    # ------------------------------------------------------------------

    def get_audit_log(self, limit: int = 50) -> dict:
        """Get recent audit entries from the shim."""
        return self._request("GET", f"/shim/audit?limit={limit}")

    # ------------------------------------------------------------------
    # Internal HTTP
    # ------------------------------------------------------------------

    def _request(self, method: str, path: str, body: Optional[dict] = None) -> dict:
        url = f"{self._url}{path}"
        req_headers: dict[str, str] = {}
        if self._api_key:
            req_headers["Authorization"] = f"Bearer {self._api_key}"

        try:
            resp = httpx.request(
                method, url, json=body, headers=req_headers, timeout=self._timeout,
            )
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            try:
                detail = e.response.json()
            except Exception:
                detail = {"error": str(e)}
            raise ShimProxyError(f"Shim request {method} {path} failed: {detail}") from e
        except httpx.RequestError as e:
            raise ShimError(
                f"Shim connection failed ({self._url}): {e}"
            ) from e
