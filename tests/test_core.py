"""Offline tests: no API calls. Run: python -m pytest -q"""

import importlib
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import store  # noqa: E402
from agent import SupportAgent  # noqa: E402
from memory import SessionMemory  # noqa: E402
from pii import PIIVault  # noqa: E402
from policy import check_refund_eligibility  # noqa: E402
from retrieval import BM25, load_corpus, route_query  # noqa: E402
from schemas import AgentReply, GetOrderInput, Order  # noqa: E402


@pytest.fixture(autouse=True)
def fresh_store():
    importlib.reload(store)  # store is in-memory and mutated by refunds/tickets
    yield


def order(oid: str) -> Order:
    return Order.model_validate(store.get_order(oid))


# ---------- policy ----------


@pytest.mark.parametrize(
    "oid, allowed, code",
    [
        ("A1001", True, "eligible"),  # delivered 14 days ago, $79.99
        ("A1002", False, "not_delivered"),  # shipped
        ("A1003", False, "outside_window"),  # delivered 52 days ago
        ("A1004", False, "over_limit"),  # $899
        ("A1005", False, "already_refunded"),
    ],
)
def test_refund_policy(oid, allowed, code):
    d = check_refund_eligibility(order(oid))
    assert (d.allowed, d.code) == (allowed, code)


# ---------- retrieval / routing ----------

INDEX = BM25(load_corpus())


@pytest.mark.parametrize(
    "query, kind, top",
    [
        ("How long does shipping take?", "faq_direct", "FAQ-1"),
        ("Do products have a warranty?", "faq_direct", "FAQ-5"),
        ("Do I need to send the item back?", "faq_direct", "FAQ-4"),
        ("what payment methods do you accept", "llm", None),  # only matches answer text
        ("What's your return policy?", "llm", None),  # ambiguous between refunds and returns
        ("Where is my order A1002?", "llm", None),
        ("I want to talk to a human", "llm", None),
        ("hi", "llm", None),
    ],
)
def test_router(query, kind, top):
    r = route_query(query, INDEX)
    assert r.kind == kind
    if top:
        assert r.hits[0][0].source_id == top


def test_router_followup_uses_memory():
    mem = SessionMemory(token_budget=2000)
    assert "follow-up" not in route_query("when will it arrive?", INDEX, mem).reason  # no active order yet
    mem.note_order_ids("my order is a1002")
    r = route_query("when will it arrive?", INDEX, mem)
    assert r.kind == "llm" and "A1002" in r.reason


# ---------- PII ----------


def test_pii_masking_roundtrip():
    v = PIIVault()
    raw = "I'm Maria Lopez, email maria@example.com, phone 415-555-0134, card 4111 1111 1111 1111. Order A1001."
    masked = v.mask(raw)
    for secret in ["Maria Lopez", "maria@example.com", "415-555-0134", "4111 1111 1111 1111"]:
        assert secret not in masked
    assert "A1001" in masked  # order ids are not PII
    assert v.unmask(masked) == raw


def test_pii_known_values_masked_later():
    v = PIIVault()
    tok = v.mask_value("NAME", "James Chen")
    assert v.mask("Is James Chen's order here?") == f"Is {tok}'s order here?"


# ---------- memory ----------


def test_memory_budget_evicts_old_turns_keeps_facts():
    mem = SessionMemory(token_budget=200)
    mem.note_order_ids("refund A1001")
    for i in range(20):
        mem.add_exchange(f"question {i} " + "x" * 100, f"answer {i} " + "y" * 100)
    assert mem.tokens() <= 200
    assert mem.evicted_turns > 0
    assert mem.turns[0].role == "user"
    assert mem.active_order_id == "A1001"  # facts survive eviction


# ---------- Pydantic reply validation ----------


def test_reply_rejects_unknown_citation():
    with pytest.raises(ValidationError, match="unknown source"):
        AgentReply.model_validate_json(
            '{"kind":"answer","answer":"Ships in 2 days [FAQ-9].","citations":["FAQ-9"]}',
            context={"known_sources": {"FAQ-1"}},
        )


def test_reply_requires_citation_for_facts():
    with pytest.raises(ValidationError, match="must cite"):
        AgentReply.model_validate_json('{"kind":"answer","answer":"Ships in 2 days.","citations":[]}')


def test_reply_collects_inline_citations():
    r = AgentReply.model_validate_json(
        '{"kind":"answer","answer":"3 to 5 days [FAQ-1].","citations":[]}', context={"known_sources": {"FAQ-1"}}
    )
    assert r.citations == ["FAQ-1"]


def test_tool_input_validation():
    assert GetOrderInput.model_validate({"order_id": " a1001 "}).order_id == "A1001"
    with pytest.raises(ValidationError):
        GetOrderInput.model_validate({"order_id": "1001; DROP TABLE"})


# ---------- tools (no LLM) ----------


def test_refund_tool_issues_valid_refund_once():
    agent = SupportAgent(client=object())
    payload, sources, err = agent._run_tool("request_refund", {"order_id": "A1001", "reason": "stopped working"})
    assert not err and payload["refunded"] and "REFUND-A1001" in sources
    assert store.get_order("A1001")["status"] == "refunded"
    payload, _, _ = agent._run_tool("request_refund", {"order_id": "A1001", "reason": "again"})
    assert not payload["refunded"] and payload["decision"] == "already_refunded"


def test_refund_tool_over_limit_opens_one_ticket():
    agent = SupportAgent(client=object())
    p1, _, _ = agent._run_tool("request_refund", {"order_id": "A1004", "reason": "dead pixels"})
    p2, _, _ = agent._run_tool("request_refund", {"order_id": "A1004", "reason": "dead pixels"})
    assert not p1["refunded"] and p1["ticket_id"] == p2["ticket_id"]
    assert len(store.list_tickets()) == 1
    assert store.get_order("A1004")["status"] == "delivered"


def test_get_order_masks_customer_name():
    agent = SupportAgent(client=object())
    payload, sources, _ = agent._run_tool("get_order", {"order_id": "A1002"})
    assert sources == {"ORDER-A1002"}
    import json

    assert "Maria" not in agent.vault.mask(json.dumps(payload))


def test_unknown_order_is_tool_error():
    agent = SupportAgent(client=object())
    payload, _, err = agent._run_tool("get_order", {"order_id": "A9999"})
    assert err and payload["found"] is False


def test_faq_direct_path_makes_no_llm_call():
    agent = SupportAgent(client=object())  # any API call would crash on this client
    reply = agent.respond("How long does shipping take?")
    assert "[FAQ-1]" in reply and agent.last_llm_calls == 0
