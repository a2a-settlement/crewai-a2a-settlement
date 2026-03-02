"""
config.py — Environment-driven configuration for crewai-a2a-settlement.

All settings have defaults so a developer only needs to export
A2A_API_KEY for a working sandbox setup.

Env var convention follows the ecosystem standard:
    A2A_EXCHANGE_URL  — exchange base URL
    A2A_API_KEY       — API key for the exchange
Legacy A2ASE_ prefixed vars are still accepted for backwards compatibility.
"""

import os

from pydantic import BaseModel, field_validator


def _env(primary: str, legacy: str, default: str) -> str:
    return os.getenv(primary) or os.getenv(legacy, default)


class A2AConfig(BaseModel):
    """
    Configuration for the A2A Settlement Exchange client.

    Reads from environment variables:
        A2A_EXCHANGE_URL (legacy: A2ASE_EXCHANGE_URL)
        A2A_API_KEY      (legacy: A2ASE_API_KEY)
        A2A_NETWORK      (legacy: A2ASE_NETWORK)
        A2A_TIMEOUT      (legacy: A2ASE_TIMEOUT)
    """

    exchange_url: str = _env("A2A_EXCHANGE_URL", "A2ASE_EXCHANGE_URL", "https://sandbox.a2a-settlement.org")
    api_key: str = _env("A2A_API_KEY", "A2ASE_API_KEY", "")
    network: str = _env("A2A_NETWORK", "A2ASE_NETWORK", "sandbox")
    timeout_seconds: int = int(_env("A2A_TIMEOUT", "A2ASE_TIMEOUT", "30"))
    auto_register: bool = _env("A2A_AUTO_REGISTER", "A2ASE_AUTO_REGISTER", "true").lower() == "true"
    batch_settlements: bool = _env("A2A_BATCH_SETTLEMENTS", "A2ASE_BATCH_SETTLEMENTS", "false").lower() == "true"

    @field_validator("network")
    @classmethod
    def validate_network(cls, v: str) -> str:
        allowed = {"sandbox", "mainnet", "devnet"}
        if v not in allowed:
            raise ValueError(f"network must be one of {allowed}, got '{v}'")
        return v

    @field_validator("timeout_seconds")
    @classmethod
    def validate_timeout(cls, v: int) -> int:
        if v < 1 or v > 300:
            raise ValueError("timeout_seconds must be between 1 and 300")
        return v
