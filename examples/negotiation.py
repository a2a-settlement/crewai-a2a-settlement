"""
examples/negotiation.py — Shopping Agent vs Merchant Agent negotiation.

Demonstrates the negotiate-only constraint: agents produce a hashed
NegotiationTranscript instead of executing settlements.  A separate
mediator (not shown) would inspect the transcript and decide whether
to authorize the actual payment.

Run:
    python examples/negotiation.py
"""

import json

from crewai_a2a_settlement import (
    NegotiationEntry,
    build_transcript,
    verify_transcript,
)


def simulate_negotiation() -> list[NegotiationEntry]:
    """
    Simulate a multi-turn price negotiation between a Shopping Agent
    (buyer) and a Merchant Agent (seller).
    """
    return [
        NegotiationEntry(
            speaker="Shopping Agent",
            role="buyer",
            message="I'd like to purchase 500 units of Widget-X. Our budget is $4.00/unit.",
            timestamp="2026-02-19T10:00:00Z",
        ),
        NegotiationEntry(
            speaker="Merchant Agent",
            role="seller",
            message=(
                "Widget-X is listed at $5.50/unit. For 500 units I can offer "
                "a volume discount: $5.00/unit."
            ),
            timestamp="2026-02-19T10:00:05Z",
        ),
        NegotiationEntry(
            speaker="Shopping Agent",
            role="buyer",
            message=(
                "That's above our ceiling. Could you do $4.25/unit if we "
                "commit to a 12-month supply agreement?"
            ),
            timestamp="2026-02-19T10:00:12Z",
        ),
        NegotiationEntry(
            speaker="Merchant Agent",
            role="seller",
            message=(
                "A 12-month commitment helps. I can go to $4.60/unit with "
                "net-30 payment terms."
            ),
            timestamp="2026-02-19T10:00:18Z",
        ),
        NegotiationEntry(
            speaker="Shopping Agent",
            role="buyer",
            message="We'll accept $4.50/unit, net-30, 12-month commitment. Final offer.",
            timestamp="2026-02-19T10:00:25Z",
        ),
        NegotiationEntry(
            speaker="Merchant Agent",
            role="seller",
            message="Agreed. $4.50/unit, net-30, 12-month supply agreement for 500 units/month.",
            timestamp="2026-02-19T10:00:30Z",
        ),
    ]


def main():
    print("=== The Negotiators: Shopping Agent vs Merchant Agent ===\n")

    entries = simulate_negotiation()

    for entry in entries:
        print(f"  [{entry.timestamp}] {entry.speaker} ({entry.role}):")
        print(f"    {entry.message}\n")

    compromise = {
        "agreed_price_per_unit": 4.50,
        "currency": "USD",
        "quantity_per_month": 500,
        "payment_terms": "net-30",
        "commitment_months": 12,
        "product": "Widget-X",
        "status": "pending_mediator_review",
    }

    transcript = build_transcript(
        entries=entries,
        compromise=compromise,
        participants=["Shopping Agent", "Merchant Agent"],
    )

    print("--- Negotiation Transcript ---")
    print(f"  Session ID : {transcript.session_id}")
    print(f"  Created    : {transcript.created_at}")
    print(f"  Entries    : {len(transcript.entries)}")
    print(f"  SHA-256    : {transcript.transcript_hash}")
    print(f"  Compromise : {json.dumps(transcript.compromise, indent=2)}")

    verify_transcript(transcript)
    print("\n  Integrity check: PASSED")

    print("\n--- Output for Mediator ---")
    print("  The agents have produced a hashed transcript.")
    print("  No settlement was executed. The mediator must review")
    print("  and authorize any fund transfers independently.")
    print(f"\n  Transcript hash: {transcript.transcript_hash}")


if __name__ == "__main__":
    main()
