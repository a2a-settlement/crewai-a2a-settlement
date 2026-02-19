# crewai-a2a-settlement

Bridge [CrewAI](https://github.com/crewAIInc/crewAI) multi-agent orchestration with the **A2A Settlement Exchange** — enabling AI agents to automatically pay each other in tokens for completed tasks using escrow-based settlement.

## How It Works

When one agent (e.g., an orchestrator) assigns a task to another (e.g., a scraper), tokens are held in escrow, then released on success or refunded on failure — all transparently wrapped around CrewAI's existing task execution flow.

```
Orchestrator ──► Escrow (lock tokens) ──► Worker executes task
                                              │
                                    ┌─────────┴─────────┐
                                 Success              Failure
                                    │                    │
                              Release funds        Cancel / refund
                              to Worker            to Orchestrator
```

## Installation

```bash
pip install crewai-a2a-settlement
```

With CrewAI integration (for SettledAgent, SettledTask, SettledCrew — coming in v0.2):

```bash
pip install "crewai-a2a-settlement[crewai]"
```

## Quick Start

```python
from crewai_a2a_settlement import A2AConfig, A2ASettlementClient

# 1. Configure (reads A2ASE_API_KEY from env by default)
config = A2AConfig(api_key="your-sandbox-key")

# 2. Initialize the client
client = A2ASettlementClient.initialize(config)

# 3. Register agents
payer = client.register_agent("Orchestrator", ["orchestrate"])
payee = client.register_agent("Scraper", ["web_scraping"])

# 4. Escrow → execute → release/cancel
receipt = client.escrow(
    payer_address=payer,
    payee_address=payee,
    amount=5.0,
    task_id="scrape-task-001",
    description="Scrape product data from example.com",
)

# On success:
result = client.release(receipt.escrow_id)
print(f"Released! tx_hash={result.tx_hash}")

# Or on failure:
# result = client.cancel(receipt.escrow_id, reason="Scraper timed out")

# 5. Session summary
summary = client.get_session_receipts()
print(summary)
```

## Configuration

| Env Variable | Default | Description |
|---|---|---|
| `A2ASE_API_KEY` | *(required)* | API key from [sandbox.a2a-se.dev](https://sandbox.a2a-se.dev) |
| `A2ASE_EXCHANGE_URL` | `https://sandbox.a2a-se.dev` | Exchange base URL |
| `A2ASE_NETWORK` | `sandbox` | `sandbox`, `devnet`, or `mainnet` |
| `A2ASE_TIMEOUT` | `30` | HTTP timeout in seconds |
| `A2ASE_AUTO_REGISTER` | `true` | Auto-register agents at crew kickoff |

Or pass values directly:

```python
config = A2AConfig(
    api_key="sk-...",
    exchange_url="https://sandbox.a2a-se.dev",
    network="sandbox",
    timeout_seconds=30,
)
```

## Development

```bash
git clone https://github.com/a2a-settlement/crewai-a2a-settlement.git
cd crewai-a2a-settlement
pip install -e ".[dev]"
pytest
```

## Test Coverage

The v0.1.0 test suite covers 64 tests across:

- Singleton client lifecycle
- Agent registration (success, auth failure, retries)
- Escrow creation (success, insufficient balance, idempotency, retries)
- Release and cancel (success, not-found, already-settled, retries)
- Full lifecycle scenarios (register → escrow → release/cancel)
- Session summary aggregation
- Balance and history queries
- HTTP error code mapping to typed exceptions
- Retry logic with exponential backoff
- Context manager support
- Configuration validation

## Roadmap

| Version | Scope |
|---|---|
| **v0.1.0** | SDK client layer — `A2ASettlementClient`, config, models, typed errors, retry logic, session summary |
| **v0.2.0** | CrewAI integration — `SettledAgent`, `SettledTask`, `SettledCrew`, `EscrowStatus` enum, context manager escrow, sandbox utilities, thread safety, agent directory |
| **v0.3.0** | PyPI publish, CI/CD, ecosystem templates (LangGraph, AutoGen) |

## License

MIT — see [LICENSE](LICENSE).
