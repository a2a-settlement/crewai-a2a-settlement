"""
Shared pytest fixtures for crewai-a2a-settlement tests.
"""

from __future__ import annotations

import pytest

from crewai_a2a_settlement.client import A2ASettlementClient


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Ensure every test starts and ends with a clean singleton."""
    A2ASettlementClient._clear_instance()
    yield
    A2ASettlementClient._clear_instance()
