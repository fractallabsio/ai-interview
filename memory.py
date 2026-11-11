"""In-memory session store with a token budget (no external DB or cache layer).

Holds two things, both already PII-masked:
- facts: the active order id plus compact snapshots of orders, refunds and tickets seen this
  session, each with its source id, so repeated questions reuse what the customer already gave;
- turns: recent user/assistant text, oldest evicted first once the budget is exceeded.

Facts are never evicted (they are small and are what "don't ask again" depends on).
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Literal

from schemas import CITATION_RE

ORDER_ID_IN_TEXT = re.compile(r"\b[A-Za-z]\d{4}\b")
MAX_ORDER_SNAPSHOTS = 5


def estimate_tokens(text: str) -> int:
    """~4 characters per token. Cheap and good enough for budgeting; no API call per turn."""
    return max(1, math.ceil(len(text) / 4))


@dataclass
class Turn:
    role: Literal["user", "assistant"]
    text: str


class SessionMemory:
    def __init__(self, token_budget: int) -> None:
        self.token_budget = token_budget
        self.turns: list[Turn] = []
        self.active_order_id: str | None = None
        self.orders: dict[str, dict] = {}  # order_id -> masked snapshot
        self.refunds: dict[str, dict] = {}  # order_id -> refund result
        self.tickets: dict[str, dict] = {}  # ticket_id -> ticket
        self.evicted_turns = 0

    # ----- facts -----

    def note_order_ids(self, text: str) -> None:
        ids = [m.upper() for m in ORDER_ID_IN_TEXT.findall(text)]
        if ids:
            self.active_order_id = ids[-1]

    def remember_order(self, snapshot: dict) -> None:
        oid = snapshot["order_id"]
        self.orders.pop(oid, None)  # re-insert so it becomes most recent
        self.orders[oid] = snapshot
        while len(self.orders) > MAX_ORDER_SNAPSHOTS:
            self.orders.pop(next(iter(self.orders)))
        self.active_order_id = oid

    def remember_refund(self, order_id: str, refund: dict) -> None:
        self.refunds[order_id] = refund

    def remember_ticket(self, ticket: dict) -> None:
        self.tickets[ticket["ticket_id"]] = ticket

    def ticket_for_order(self, order_id: str) -> dict | None:
        return next((t for t in self.tickets.values() if t.get("order_id") == order_id), None)

    def known_sources(self) -> set[str]:
        """Source ids the model may cite from memory: stored facts plus ids cited in remembered turns."""
        ids = {f"ORDER-{oid}" for oid in self.orders}
        ids |= {f"REFUND-{oid}" for oid in self.refunds}
        ids |= {f"TICKET-{tid}" for tid in self.tickets}
        for turn in self.turns:
            ids |= set(CITATION_RE.findall(turn.text))
        return ids

    def facts_block(self) -> str:
        lines = ["<session_memory>"]
        lines.append(f"active_order_id: {self.active_order_id or 'none'}")
        for oid, snap in self.orders.items():
            lines.append(f"[ORDER-{oid}] {json.dumps(snap, separators=(',', ':'))}")
        for oid, ref in self.refunds.items():
            lines.append(f"[REFUND-{oid}] {json.dumps(ref, separators=(',', ':'))}")
        for tid, t in self.tickets.items():
            lines.append(f"[TICKET-{tid}] {json.dumps(t, separators=(',', ':'))}")
        lines.append("</session_memory>")
        return "\n".join(lines)

    # ----- turns -----

    def add_exchange(self, user_text: str, assistant_text: str) -> None:
        self.turns.append(Turn("user", user_text))
        self.turns.append(Turn("assistant", assistant_text))
        self._enforce_budget()

    def tokens(self) -> int:
        return estimate_tokens(self.facts_block()) + sum(estimate_tokens(t.text) for t in self.turns)

    def _enforce_budget(self) -> None:
        # Evict whole user+assistant pairs so the history still starts with a user turn.
        while self.tokens() > self.token_budget and len(self.turns) > 2:
            del self.turns[:2]
            self.evicted_turns += 2

    def history_messages(self) -> list[dict]:
        return [{"role": t.role, "content": t.text} for t in self.turns]
