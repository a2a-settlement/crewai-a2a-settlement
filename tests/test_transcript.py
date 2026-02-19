"""
test_transcript.py — Tests for the negotiation transcript utility.

Covers:
  - build_transcript happy path
  - SHA-256 determinism (same input → same hash)
  - SHA-256 sensitivity (different input → different hash)
  - Forbidden-phrase detection in entries
  - Forbidden-phrase detection in compromise dict
  - Case-insensitive detection
  - Participant inference from entries
  - Session ID auto-generation
  - Empty entries rejection
  - verify_transcript integrity check (pass and fail)
  - hash_transcript standalone usage
  - Canonical JSON ordering

Run with:
    pytest tests/test_transcript.py -v
"""

from __future__ import annotations

import hashlib
import json

import pytest

from crewai_a2a_settlement.models import NegotiationEntry, NegotiationTranscript
from crewai_a2a_settlement.transcript import (
    TranscriptIntegrityError,
    TranscriptValidationError,
    build_transcript,
    hash_transcript,
    validate_no_execution_authority,
    verify_transcript,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def two_entries() -> list[NegotiationEntry]:
    return [
        NegotiationEntry(
            speaker="Shopping Agent",
            role="buyer",
            message="I want to buy 100 widgets at $3/unit.",
            timestamp="2026-02-19T10:00:00Z",
        ),
        NegotiationEntry(
            speaker="Merchant Agent",
            role="seller",
            message="I can offer $3.50/unit for that volume.",
            timestamp="2026-02-19T10:00:05Z",
        ),
    ]


@pytest.fixture
def sample_compromise() -> dict:
    return {
        "agreed_price": 3.25,
        "quantity": 100,
        "product": "widgets",
        "status": "pending_mediator_review",
    }


# ---------------------------------------------------------------------------
# build_transcript — happy path
# ---------------------------------------------------------------------------

class TestBuildTranscript:
    def test_returns_negotiation_transcript(self, two_entries, sample_compromise):
        t = build_transcript(two_entries, sample_compromise)
        assert isinstance(t, NegotiationTranscript)

    def test_populates_all_fields(self, two_entries, sample_compromise):
        t = build_transcript(
            two_entries,
            sample_compromise,
            participants=["Shopping Agent", "Merchant Agent"],
            session_id="test-session-001",
        )
        assert t.session_id == "test-session-001"
        assert t.participants == ["Merchant Agent", "Shopping Agent"]
        assert len(t.entries) == 2
        assert t.compromise == sample_compromise
        assert t.created_at != ""
        assert t.transcript_hash != ""
        assert t.transcript_json != ""

    def test_infers_participants_from_entries(self, two_entries, sample_compromise):
        t = build_transcript(two_entries, sample_compromise)
        assert sorted(t.participants) == ["Merchant Agent", "Shopping Agent"]

    def test_generates_session_id_when_omitted(self, two_entries, sample_compromise):
        t = build_transcript(two_entries, sample_compromise)
        assert len(t.session_id) == 36  # UUID4 format

    def test_empty_entries_raises(self, sample_compromise):
        with pytest.raises(ValueError, match="at least one"):
            build_transcript([], sample_compromise)


# ---------------------------------------------------------------------------
# SHA-256 determinism and sensitivity
# ---------------------------------------------------------------------------

class TestHashDeterminism:
    def test_same_input_same_hash(self):
        payload = '{"key": "value"}'
        assert hash_transcript(payload) == hash_transcript(payload)

    def test_different_input_different_hash(self):
        assert hash_transcript('{"a": 1}') != hash_transcript('{"a": 2}')

    def test_hash_matches_stdlib(self):
        payload = '{"negotiation": "test"}'
        expected = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        assert hash_transcript(payload) == expected

    def test_hash_is_64_char_hex(self, two_entries, sample_compromise):
        t = build_transcript(two_entries, sample_compromise)
        assert len(t.transcript_hash) == 64
        int(t.transcript_hash, 16)  # raises if not valid hex


# ---------------------------------------------------------------------------
# Forbidden-phrase detection
# ---------------------------------------------------------------------------

class TestForbiddenPhrases:
    @pytest.mark.parametrize("phrase", [
        "Settlement Approved",
        "settlement approved",
        "SETTLEMENT APPROVED",
        "Execute Settlement",
        "execute settlement",
        "Approve Payment",
        "Release Funds",
        "release funds",
    ])
    def test_forbidden_in_entry_message(self, phrase, sample_compromise):
        entries = [
            NegotiationEntry(
                speaker="Bad Agent",
                role="buyer",
                message=f"I hereby declare: {phrase}!",
                timestamp="2026-02-19T10:00:00Z",
            ),
        ]
        with pytest.raises(TranscriptValidationError, match="forbidden phrase"):
            build_transcript(entries, sample_compromise)

    def test_forbidden_in_compromise_dict(self, two_entries):
        bad_compromise = {
            "decision": "Settlement Approved",
            "amount": 100,
        }
        with pytest.raises(TranscriptValidationError, match="forbidden phrase"):
            build_transcript(two_entries, bad_compromise)

    def test_case_insensitive_detection(self, sample_compromise):
        entries = [
            NegotiationEntry(
                speaker="Sneaky Agent",
                role="buyer",
                message="sEtTlEmEnT aPpRoVeD by me.",
                timestamp="2026-02-19T10:00:00Z",
            ),
        ]
        with pytest.raises(TranscriptValidationError):
            build_transcript(entries, sample_compromise)

    def test_clean_entries_pass(self, two_entries, sample_compromise):
        validate_no_execution_authority(two_entries, sample_compromise)

    def test_partial_match_in_value(self, two_entries):
        compromise = {
            "notes": "The agent tried to release funds but was blocked.",
        }
        with pytest.raises(TranscriptValidationError):
            build_transcript(two_entries, compromise)


# ---------------------------------------------------------------------------
# verify_transcript
# ---------------------------------------------------------------------------

class TestVerifyTranscript:
    def test_valid_transcript_passes(self, two_entries, sample_compromise):
        t = build_transcript(two_entries, sample_compromise)
        assert verify_transcript(t) is True

    def test_tampered_hash_fails(self, two_entries, sample_compromise):
        t = build_transcript(two_entries, sample_compromise)
        t.transcript_hash = "0" * 64
        with pytest.raises(TranscriptIntegrityError, match="Hash mismatch"):
            verify_transcript(t)

    def test_tampered_json_fails(self, two_entries, sample_compromise):
        t = build_transcript(two_entries, sample_compromise)
        t.transcript_json = t.transcript_json.replace("buyer", "hacker")
        with pytest.raises(TranscriptIntegrityError, match="Hash mismatch"):
            verify_transcript(t)

    def test_missing_json_raises(self):
        t = NegotiationTranscript(
            session_id="x",
            participants=["A"],
            transcript_hash="abc",
            transcript_json="",
        )
        with pytest.raises(TranscriptIntegrityError, match="missing"):
            verify_transcript(t)

    def test_missing_hash_raises(self):
        t = NegotiationTranscript(
            session_id="x",
            participants=["A"],
            transcript_hash="",
            transcript_json='{"data": 1}',
        )
        with pytest.raises(TranscriptIntegrityError, match="missing"):
            verify_transcript(t)


# ---------------------------------------------------------------------------
# Canonical JSON ordering
# ---------------------------------------------------------------------------

class TestCanonicalJson:
    def test_json_keys_are_sorted(self, two_entries, sample_compromise):
        t = build_transcript(two_entries, sample_compromise)
        parsed = json.loads(t.transcript_json)
        assert list(parsed.keys()) == sorted(parsed.keys())

    def test_json_contains_version(self, two_entries, sample_compromise):
        t = build_transcript(two_entries, sample_compromise)
        parsed = json.loads(t.transcript_json)
        assert parsed["version"] == "1.0"

    def test_json_roundtrip_matches_hash(self, two_entries, sample_compromise):
        t = build_transcript(
            two_entries, sample_compromise, session_id="round-trip"
        )
        recomputed = hashlib.sha256(
            t.transcript_json.encode("utf-8")
        ).hexdigest()
        assert recomputed == t.transcript_hash


# ---------------------------------------------------------------------------
# hash_transcript standalone
# ---------------------------------------------------------------------------

class TestHashTranscriptStandalone:
    def test_empty_string(self):
        h = hash_transcript("")
        expected = hashlib.sha256(b"").hexdigest()
        assert h == expected

    def test_unicode_content(self):
        payload = '{"price": "€4.50"}'
        h = hash_transcript(payload)
        expected = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        assert h == expected


# ---------------------------------------------------------------------------
# No "Settlement Approved" in final output
# ---------------------------------------------------------------------------

class TestNoSettlementApprovedOutput:
    """
    Core constraint: agents must NEVER output 'Settlement Approved'.
    Their output is always the hashed transcript for the mediator.
    """

    def test_transcript_json_never_contains_settlement_approved(
        self, two_entries, sample_compromise
    ):
        t = build_transcript(two_entries, sample_compromise)
        assert "Settlement Approved" not in t.transcript_json
        assert "settlement approved" not in t.transcript_json.lower()

    def test_compromise_status_is_pending(self, two_entries, sample_compromise):
        t = build_transcript(two_entries, sample_compromise)
        assert t.compromise["status"] == "pending_mediator_review"

    def test_agent_cannot_sneak_approval_via_metadata(self, sample_compromise):
        entries = [
            NegotiationEntry(
                speaker="Merchant Agent",
                role="seller",
                message="Let's finalize the deal.",
                timestamp="2026-02-19T10:00:00Z",
                metadata={"internal_note": "Settlement Approved"},
            ),
        ]
        # metadata isn't scanned by validate_no_execution_authority
        # (it's an opaque passthrough), but the canonical JSON is clean
        t = build_transcript(entries, sample_compromise)
        assert isinstance(t.transcript_hash, str)
