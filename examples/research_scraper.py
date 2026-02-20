"""
examples/research_scraper.py — Demo: orchestrator pays scraper per task.

This example shows the escrow lifecycle without CrewAI integration.
The CrewAI wrappers (SettledAgent, SettledTask, SettledCrew) will
automate these steps in v0.2.0.

Prerequisites:
    export A2ASE_API_KEY="your-sandbox-key"

Run:
    python examples/research_scraper.py              # immediate mode
    python examples/research_scraper.py --batch      # batch mode
"""

import sys

from crewai_a2a_settlement import A2AConfig, A2ASettlementClient


def main_immediate():
    """One escrow per task, released immediately on success."""
    config = A2AConfig()
    client = A2ASettlementClient.initialize(config)

    print("Registering agents...")
    orchestrator_wallet = client.register_agent(
        "Research Orchestrator",
        ["orchestrate", "delegate"],
    )
    scraper_wallet = client.register_agent(
        "Web Scraper",
        ["web_scraping", "data_extraction"],
    )
    print(f"  Orchestrator: {orchestrator_wallet}")
    print(f"  Scraper:      {scraper_wallet}")

    print("\nCreating escrow for scraping task...")
    receipt = client.escrow(
        payer_address=orchestrator_wallet,
        payee_address=scraper_wallet,
        amount=5.0,
        task_id="scrape-products-001",
        description="Scrape top 100 products from example.com",
    )
    print(f"  Escrow ID: {receipt.escrow_id}")
    print(f"  Amount:    {receipt.amount} tokens")

    task_succeeded = True

    if task_succeeded:
        print("\nTask succeeded — releasing escrow...")
        result = client.release(receipt.escrow_id)
        print(f"  Status:  {result.status}")
        print(f"  TX Hash: {result.tx_hash}")
    else:
        print("\nTask failed — cancelling escrow...")
        result = client.cancel(receipt.escrow_id, reason="Scraper timed out")
        print(f"  Status: {result.status}")

    print("\n" + str(client.get_session_receipts()))


def main_batch():
    """Multiple tasks, all released in a single batch at the end."""
    config = A2AConfig(batch_settlements=True)
    client = A2ASettlementClient.initialize(config)

    print("Registering agents...")
    orchestrator = client.register_agent("Orchestrator", ["orchestrate"])
    scraper = client.register_agent("Web Scraper", ["web_scraping"])
    analyst = client.register_agent("Analyst", ["data_analysis"])
    print(f"  Orchestrator: {orchestrator}")
    print(f"  Scraper:      {scraper}")
    print(f"  Analyst:      {analyst}")

    tasks = [
        ("scrape-001", scraper, 5.0, "Scrape product listings"),
        ("scrape-002", scraper, 3.0, "Scrape competitor prices"),
        ("analyze-001", analyst, 4.0, "Analyze pricing trends"),
    ]

    for task_id, payee, amount, desc in tasks:
        print(f"\nEscrow + execute: {desc}")
        receipt = client.escrow(
            payer_address=orchestrator,
            payee_address=payee,
            amount=amount,
            task_id=task_id,
            description=desc,
        )
        result = client.release(receipt.escrow_id)
        print(f"  Escrow {receipt.escrow_id}: status={result.status}")

    print(f"\nPending releases: {client.get_pending_count()}")
    print("Flushing batch settlement...")
    batch = client.flush_settlements()
    print(f"  Released:  {batch.escrow_count} escrows")
    print(f"  Total:     {batch.total_released} tokens")
    print(f"  Batch TX:  {batch.batch_tx_hash}")

    if batch.failed_escrow_ids:
        print(f"  Failed:    {batch.failed_escrow_ids}")

    print("\n" + str(client.get_session_receipts()))


if __name__ == "__main__":
    if "--batch" in sys.argv:
        main_batch()
    else:
        main_immediate()
