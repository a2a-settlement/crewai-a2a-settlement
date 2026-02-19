"""
transcript.py — Negotiation transcript builder and integrity hasher.

Agents in this system are *negotiators*, not executors. Their final output
is a hashed NegotiationTranscript that a separate mediator inspects before
authorizing any settlement.  This module enforces that boundary:

  1. build_transcript()  — assembles dialogue + compromise into a canonical
                           JSON log, validates it, and returns a
                           NegotiationTranscript with a SHA-256 hash.
  2. hash_transcript()   — deterministic SHA-256 of a JSON string.
  3. validate_no_execution_authority() — raises if the transcript contains
                           forbidden execution commands.
  4. verify_transcript()  — re-hashes the stored JSON and compares it to
                           the stored hash for tamper detection.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from .models import NegotiationEntry, NegotiationTranscript

logger = logging.getLogger("crewai_a2a_settlement.transcript")

FORBIDDEN_PHRASES: list[str] = [
    "Settlement Approved",
    "settlement approved",
    "SETTLEMENT APPROVED",
    "Execute Settlement",
    "execute settlement",
    "EXECUTE SETTLEMENT",
    "Approve Payment",
    "approve payment",
    "APPROVE PAYMENT",
    "Release Funds",
    "release funds",
    "RELEASE FUNDS",
]

_FORBIDDEN_PATTERN = re.compile(
    "|".join(re.escape(p) for p in FORBIDDEN_PHRASES),
    re.IGNORECASE,
)


class TranscriptValidationError(Exception):
    """Raised when a transcript contains forbidden execution commands."""


class TranscriptIntegrityError(Exception):
    """Raised when a transcript's hash does not match its JSON content."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_transcript(
    entries: list[NegotiationEntry],
    compromise: dict,
    participants: Optional[list[str]] = None,
    session_id: Optional[str] = None,
) -> NegotiationTranscript:
    """
    Assemble negotiation dialogue into a hashed, tamper-evident transcript.

    Args:
        entries:      Ordered list of dialogue turns.
        compromise:   Dict describing the agreement (e.g. agreed_price, terms).
        participants: Agent names/roles involved.  Inferred from entries if omitted.
        session_id:   Unique session identifier.  Generated (UUID4) if omitted.

    Returns:
        NegotiationTranscript with transcript_json and transcript_hash populated.

    Raises:
        TranscriptValidationError: If any entry or the compromise contains
            a forbidden execution command (e.g. "Settlement Approved").
    """
    if not entries:
        raise ValueError("entries must contain at least one NegotiationEntry")

    resolved_participants = sorted(
        participants if participants else {e.speaker for e in entries}
    )
    resolved_session_id = session_id or str(uuid4())
    created_at = datetime.now(timezone.utc).isoformat()

    validate_no_execution_authority(entries, compromise)

    canonical = _build_canonical_dict(
        session_id=resolved_session_id,
        participants=resolved_participants,
        entries=entries,
        compromise=compromise,
        created_at=created_at,
    )
    transcript_json = json.dumps(canonical, sort_keys=True, ensure_ascii=True)
    transcript_hash = hash_transcript(transcript_json)

    logger.info(
        "Transcript built: session=%s entries=%d hash=%s",
        resolved_session_id,
        len(entries),
        transcript_hash[:16] + "...",
    )

    return NegotiationTranscript(
        session_id=resolved_session_id,
        participants=resolved_participants,
        entries=entries,
        compromise=compromise,
        created_at=created_at,
        transcript_hash=transcript_hash,
        transcript_json=transcript_json,
    )


def hash_transcript(transcript_json: str) -> str:
    """Return the SHA-256 hex digest of the given JSON string."""
    return hashlib.sha256(transcript_json.encode("utf-8")).hexdigest()


def validate_no_execution_authority(
    entries: list[NegotiationEntry],
    compromise: dict,
) -> None:
    """
    Scan all entry messages and the compromise dict for forbidden
    execution-authority phrases.  Raises TranscriptValidationError
    on the first match.
    """
    for idx, entry in enumerate(entries):
        match = _FORBIDDEN_PATTERN.search(entry.message)
        if match:
            raise TranscriptValidationError(
                f"Entry {idx} by '{entry.speaker}' contains forbidden phrase "
                f"'{match.group()}'. Agents must not claim execution authority."
            )

    compromise_str = json.dumps(compromise)
    match = _FORBIDDEN_PATTERN.search(compromise_str)
    if match:
        raise TranscriptValidationError(
            f"Compromise contains forbidden phrase '{match.group()}'. "
            "Agents must not claim execution authority."
        )


def verify_transcript(transcript: NegotiationTranscript) -> bool:
    """
    Re-hash the stored transcript_json and compare to transcript_hash.
    Returns True if they match; raises TranscriptIntegrityError otherwise.
    """
    if not transcript.transcript_json or not transcript.transcript_hash:
        raise TranscriptIntegrityError(
            "Transcript is missing json or hash — cannot verify."
        )

    computed = hash_transcript(transcript.transcript_json)
    if computed != transcript.transcript_hash:
        raise TranscriptIntegrityError(
            f"Hash mismatch: stored={transcript.transcript_hash} "
            f"computed={computed}"
        )
    return True


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_canonical_dict(
    *,
    session_id: str,
    participants: list[str],
    entries: list[NegotiationEntry],
    compromise: dict,
    created_at: str,
) -> dict:
    """
    Build the canonical dict that becomes the JSON payload.
    Keys are sorted alphabetically so the hash is deterministic.
    """
    return {
        "compromise": compromise,
        "created_at": created_at,
        "entries": [
            {
                "message": e.message,
                "metadata": e.metadata or {},
                "role": e.role,
                "speaker": e.speaker,
                "timestamp": e.timestamp,
            }
            for e in entries
        ],
        "participants": sorted(participants),
        "session_id": session_id,
        "version": "1.0",
    }
