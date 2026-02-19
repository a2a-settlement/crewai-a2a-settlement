"""
examples/research_scraper.py — Demo: orchestrator pays scraper per task.

This example shows the escrow lifecycle without CrewAI integration.
The CrewAI wrappers (SettledAgent, SettledTask, SettledCrew) will
automate these steps in v0.2.0.

Prerequisites:
    export A2ASE_API_KEY="your-sandbox-key"

Run:
    python examples/research_scraper.py
"""

from crewai_a2a_settlement import A2AConfig, A2ASettlementClient


def main():
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

    # Simulate task execution
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


if __name__ == "__main__":
    main()
