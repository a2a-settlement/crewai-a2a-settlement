"""
config.py — Environment-driven configuration for crewai-a2a-settlement.

All settings have defaults so a developer only needs to export
A2ASE_API_KEY and A2ASE_NETWORK for a working sandbox setup.
"""

import os

from pydantic import BaseModel, field_validator


class A2AConfig(BaseModel):
    """
    Configuration for the A2A Settlement Exchange client.

    Reads from environment variables by default:
        A2ASE_EXCHANGE_URL  — exchange base URL (default: sandbox)
        A2ASE_API_KEY       — required, get at sandbox.a2a-se.dev
        A2ASE_NETWORK       — "sandbox" or "mainnet" (default: sandbox)
        A2ASE_TIMEOUT       — HTTP timeout in seconds (default: 30)
        A2ASE_AUTO_REGISTER — auto-register agents at kickoff (default: true)
    """

    exchange_url: str = os.getenv(
        "A2ASE_EXCHANGE_URL", "https://sandbox.a2a-se.dev"
    )
    api_key: str = os.getenv("A2ASE_API_KEY", "")
    network: str = os.getenv("A2ASE_NETWORK", "sandbox")
    timeout_seconds: int = int(os.getenv("A2ASE_TIMEOUT", "30"))
    auto_register: bool = os.getenv("A2ASE_AUTO_REGISTER", "true").lower() == "true"

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
