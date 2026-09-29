"""
services/fast_path.py — Lightweight Fast-Path Classifier

Handles crystal-clear queries without consuming the full agent loop.
Only two categories qualify for fast-path:
  1. sql_only   — bare 12-digit registration number OR faculty/designation lookups
  2. map_only   — pure navigation query with no student/document component

Everything else returns None → falls through to the full agentic loop.
"""

import re

# ── Patterns ───────────────────────────────────────────────────────────────────

# Anna University 12-digit reg number (always → SQL)
_REG_NO_RE = re.compile(r"\b5\d{11}\b")

# Faculty / professor / designation queries (→ SQL faculty table)
_FACULTY_RE = re.compile(
    r"\b("
    # By prefix (Prof., Dr., Mr., Mrs.)
    r"(who is|find|search|get|show|fetch|contact of|phone of|email of|cabin of|room of|details of|info of)\s+"
    r"(prof\.?|dr\.?|mr\.?|mrs\.?|ms\.?)\b"
    # By designation keyword
    r"|who is (the )?(hod|head of department|principal|director|dean|warden|librarian|"
    r"counsellor|counselor|placement officer|class advisor|coordinator)\b"
    r"|(contact|phone|email|cabin|room|details|find|show|get)\b.{0,30}"
    r"(hod|principal|director|dean|professor|faculty|staff)\b"
    # List faculty by dept
    r"|(list|show|all)\b.{0,20}(faculty|staff|professors?|teachers?)\b.{0,30}"
    r"(cse|it|ece|eee|mech|civil|ai|department|dept)\b"
    r")\b",
    re.IGNORECASE,
)

# Pure navigation phrases (→ map tools only)
_NAV_RE = re.compile(
    r"\b(navigate|how (do i|to) get (to|from)|directions? (to|from)|"
    r"find .{0,20}(room|lab|office|block|floor)|"
    r"path (from|to)|route (to|from)|"
    r"where is .{0,30}(room|lab|block|department|floor|building|toilet|canteen|library))\b",
    re.IGNORECASE,
)


def fast_path(question: str) -> str | None:
    """
    Returns:
        "sql_only"  — question definitely needs only SQL (reg number or faculty lookup)
        "map_only"  — question is purely about campus navigation
        None        — ambiguous / complex → use full agent loop
    """
    q = question.strip()

    # Bare reg number → direct SQL lookup
    if _REG_NO_RE.search(q):
        return "sql_only"

    # Faculty / designation lookups → direct SQL
    if _FACULTY_RE.search(q):
        return "sql_only"

    # Pure navigation (no student/document signals)
    if _NAV_RE.search(q):
        return "map_only"

    return None  # full agent loop
