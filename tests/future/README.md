# Future Tests

Tests in this directory target **v0.2.0** features that are not yet implemented.

## `test_escrow_lifecycle.py`

Requires the following additions to the codebase:

- `EscrowStatus` enum (`HELD`, `RELEASED`, `CANCELLED`)
- `ReleaseReceipt` model
- `SettlementError` with error codes (`UNAUTHORIZED`, `INSUFFICIENT_BALANCE`, `NOT_FOUND`, etc.)
- `EscrowReceipt` as Pydantic model with `is_settled()`, `network`, `description`, `model_dump()`
- `SessionSummary` with `total_transactions`, `total_held`, `success_rate`
- `A2AConfig.max_retries`, `auto_fund_sandbox`, `is_sandbox`
- `A2ASettlementClient.reset()`, `escrow_context()`, `fund_sandbox_wallet()`, `list_directory()`, `get_session_summary()`, `get_registration()`
- `_truncate()` utility function
- Thread-safe escrow tracking
- Registration caching

These tests are excluded from the default test run (`pytest` only collects from `tests/`
top-level by default with `testpaths = ["tests"]`). Move individual test files up to
`tests/` as the corresponding features are implemented.
