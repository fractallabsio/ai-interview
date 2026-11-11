"""Brightcart support agent.

Per customer message:
  1. mask PII (pii.PIIVault) - the LLM only ever sees placeholders
  2. route (retrieval.route_query) - confident FAQ hits are answered from BM25 directly, no LLM call
  3. otherwise call Claude with: session memory facts + top BM25 hits + tools
  4. tools hit store.py; refund policy is enforced in code (policy.py), never by the model
  5. the final reply is JSON validated by Pydantic (schemas.AgentReply): every cited source id
     must be one the model was actually shown this turn or holds in memory; one corrective retry
  6. unmask placeholders and return
"""

from __future__ import annotations

import json
import logging

import anthropic
from pydantic import BaseModel, ValidationError

import store
from memory import SessionMemory, estimate_tokens
from pii import PIIVault
from policy import POLICY_SOURCE_ID, check_refund_eligibility
from retrieval import BM25, Route, load_corpus, route_query
from schemas import (
    AgentReply,
    EscalateInput,
    GetOrderInput,
    Order,
    RequestRefundInput,
    SearchFaqInput,
)

log = logging.getLogger(__name__)

MODEL = "claude-opus-5-5"
EFFORT = "medium"

# Token budget for the conversation context we send per turn (memory + retrieved docs + message).
# Session memory gets half of it; the other half is left for retrieval and tool results.
LLM_CONTEXT_TOKEN_BUDGET = 4000
MEMORY_TOKEN_BUDGET = LLM_CONTEXT_TOKEN_BUDGET // 2

MAX_TOOL_ROUNDS = 6
MAX_VALIDATION_RETRIES = 1

SYSTEM_PROMPT = f"""You are the customer support assistant for Brightcart, an online electronics store. Today is {store.TODAY.isoformat()}.

Grounding (most important):
- State store facts only when they come from a source shown in this conversation: <retrieved_context>, <session_memory>, or a tool result. Every source has an id such as FAQ-2, POLICY-REFUND, ORDER-A1001, REFUND-A1001 or TICKET-T-1A2B3C.
- Put the id in square brackets right after the sentence it supports, e.g. "Standard shipping takes 3 to 5 business days [FAQ-1].", and list every id you used in `citations`.
- If no source covers the question, say you don't have that information and offer to connect the customer with a human agent. Do not guess or use general knowledge about how stores usually work.

Orders:
- Use get_order to look up an order. If the customer refers to their order without giving an id, use active_order_id from <session_memory>. Never ask for an order id the customer already gave.
- For a repeated question about an order already in <session_memory>, you may answer from that snapshot and cite it.

Refunds:
- Always call request_refund for a refund request. It applies the store policy and either issues the refund, explains why not, or opens a ticket for a human. Never promise or deny a refund yourself.
- Relay the tool's reason when a refund is not allowed. If a ticket was opened, give the customer the ticket id.

Humans:
- If the customer asks for a person, call escalate_to_human and give them the ticket id.

Personal details:
- Placeholders such as [NAME_1], [EMAIL_1] or [PHONE_1] stand for the customer's masked personal details. Use them as-is when needed; never guess the real values.

Reply format:
- `kind`: "answer" when you state store facts (must cite), "clarification" when you need information from the customer, "smalltalk" for greetings or thanks, "unable" when no source covers the question.
- Keep answers short and friendly: 1 to 4 sentences of plain text, no markdown headings."""


def _tool(name: str, description: str, model: type[BaseModel]) -> dict:
    return {"name": name, "description": description, "input_schema": model.model_json_schema(), "strict": True}


TOOLS: list[dict] = [
    _tool("get_order", "Look up an order by id: item, total, status, delivery dates, tracking.", GetOrderInput),
    _tool(
        "search_faq",
        "Search the store FAQ and policies (shipping, refunds, returns, warranty, contact). "
        "Use when <retrieved_context> doesn't already answer the question.",
        SearchFaqInput,
    ),
    _tool(
        "request_refund",
        "Request a refund for an order. The store policy is checked automatically: the refund is issued "
        "if allowed; otherwise the reason is returned, and a ticket for a human agent is opened when the "
        "order is over the automatic refund limit.",
        RequestRefundInput,
    ),
    _tool(
        "escalate_to_human",
        "Open a support ticket for a human agent. Use when the customer asks for a person.",
        EscalateInput,
    ),
]
REPLY_SCHEMA = AgentReply.model_json_schema()

FALLBACK_REPLY = AgentReply(
    kind="unable",
    answer="Sorry, I couldn't put together a reliable answer to that. I can open a ticket for a human agent if you'd like.",
    citations=[],
)


class SupportAgent:
    def __init__(self, client: anthropic.Anthropic | None = None) -> None:
        self._client = client
        self.index = BM25(load_corpus())
        self.vault = PIIVault()
        self.memory = SessionMemory(token_budget=MEMORY_TOKEN_BUDGET)
        # Last turn's details, for the CLI and tests.
        self.last_route: Route | None = None
        self.last_reply: AgentReply | None = None
        self.last_llm_calls = 0

    @property
    def client(self) -> anthropic.Anthropic:
        if self._client is None:
            self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
        return self._client

    # ---------- public ----------

    def respond(self, user_message: str) -> str:
        masked = self.vault.mask(user_message)
        self.memory.note_order_ids(masked)
        self.last_llm_calls = 0

        route = route_query(masked, self.index, self.memory)
        self.last_route = route
        if route.kind == "faq_direct":
            doc = route.hits[0][0]
            reply = AgentReply(kind="answer", answer=f"{doc.answer} [{doc.source_id}]", citations=[doc.source_id])
        else:
            reply = self._llm_turn(masked, route)

        self.last_reply = reply
        self.memory.add_exchange(masked, reply.answer)
        return self.vault.unmask(reply.answer)

    # ---------- LLM path ----------

    def _context_block(self, route: Route, sources: set[str]) -> str:
        """Top BM25 hits, trimmed so memory + retrieval stays inside the context budget."""
        remaining = LLM_CONTEXT_TOKEN_BUDGET - self.memory.tokens()
        lines = ["<retrieved_context>"]
        for doc, _score in route.hits:
            line = f"[{doc.source_id}] Q: {doc.question} A: {doc.answer}"
            if estimate_tokens(line) > remaining:
                break
            remaining -= estimate_tokens(line)
            lines.append(line)
            sources.add(doc.source_id)
        if len(lines) == 1:
            lines.append("(no matching FAQ entries)")
        lines.append("</retrieved_context>")
        return "\n".join(lines)

    def _llm_turn(self, masked: str, route: Route) -> AgentReply:
        known = self.memory.known_sources()
        content = [
            {"type": "text", "text": self.memory.facts_block()},
            {"type": "text", "text": self._context_block(route, known)},
            {"type": "text", "text": f"<customer_message>\n{masked}\n</customer_message>"},
        ]
        # Past turns are replayed as plain text; within this turn, messages are append-only.
        messages: list[dict] = self.memory.history_messages() + [{"role": "user", "content": content}]

        tool_rounds = 0
        retries = 0
        while True:
            response = self._call(messages)
            if response.stop_reason == "refusal":
                return FALLBACK_REPLY
            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "tool_use":
                tool_rounds += 1
                if tool_rounds > MAX_TOOL_ROUNDS:
                    log.warning("tool round limit hit")
                    return FALLBACK_REPLY
                results = []
                for block in response.content:
                    if block.type == "tool_use":
                        payload, sources, is_error = self._run_tool(block.name, block.input)
                        known |= sources
                        results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": self.vault.mask(json.dumps(payload)),
                                "is_error": is_error,
                            }
                        )
                messages.append({"role": "user", "content": results})
                continue

            text = next((b.text for b in reversed(response.content) if b.type == "text"), "")
            try:
                return AgentReply.model_validate_json(text, context={"known_sources": known})
            except ValidationError as err:
                log.warning("reply failed validation: %s", err)
                if retries >= MAX_VALIDATION_RETRIES or response.stop_reason == "max_tokens":
                    return FALLBACK_REPLY
                retries += 1
                problems = "; ".join(e["msg"] for e in err.errors())
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Your reply failed validation: {problems}. Reply again in the required JSON format. "
                            f"You may only cite these source ids: {sorted(known)}."
                        ),
                    }
                )

    def _call(self, messages: list[dict]):
        self.last_llm_calls += 1
        return self.client.beta.messages.create(
            model=MODEL,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
            output_config={"effort": EFFORT, "format": {"type": "json_schema", "schema": REPLY_SCHEMA}},
            # If a safety classifier declines, retry on a fallback model within the same call.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )

    # ---------- tools ----------

    def _run_tool(self, name: str, args: dict) -> tuple[dict, set[str], bool]:
        """Returns (payload, source ids it introduces, is_error). Payload is masked by the caller."""
        handlers = {
            "get_order": self._tool_get_order,
            "search_faq": self._tool_search_faq,
            "request_refund": self._tool_request_refund,
            "escalate_to_human": self._tool_escalate,
        }
        handler = handlers.get(name)
        if handler is None:
            return {"error": f"unknown tool {name}"}, set(), True
        try:
            payload, sources = handler(args)
            return payload, sources, False
        except ValidationError as err:
            return {"error": "invalid input", "details": [e["msg"] for e in err.errors()]}, set(), True
        except store.NotFoundError as err:
            return {"error": str(err), "found": False}, set(), True

    def _load_order(self, order_id: str) -> Order:
        order = Order.model_validate(store.get_order(order_id))
        self.vault.mask_value("NAME", order.customer_name)  # names from the store never reach the LLM
        snapshot = order.model_dump(mode="json", exclude={"customer_id"}, exclude_none=True)
        self.memory.remember_order(snapshot)
        return order

    def _tool_get_order(self, args: dict) -> tuple[dict, set[str]]:
        inp = GetOrderInput.model_validate(args)
        order = self._load_order(inp.order_id)
        sid = f"ORDER-{order.order_id}"
        return {"source_id": sid, **self.memory.orders[order.order_id]}, {sid}

    def _tool_search_faq(self, args: dict) -> tuple[dict, set[str]]:
        inp = SearchFaqInput.model_validate(args)
        hits = self.index.search(inp.query, k=3)
        results = [{"source_id": d.source_id, "question": d.question, "answer": d.answer} for d, _ in hits]
        return {"results": results}, {d.source_id for d, _ in hits}

    def _tool_request_refund(self, args: dict) -> tuple[dict, set[str]]:
        inp = RequestRefundInput.model_validate(args)
        order = self._load_order(inp.order_id)
        decision = check_refund_eligibility(order)
        order_sid = f"ORDER-{order.order_id}"
        sources = {order_sid, POLICY_SOURCE_ID}
        result: dict = {
            "order_source_id": order_sid,
            "policy_source_id": POLICY_SOURCE_ID,
            "refunded": decision.allowed,
            "decision": decision.code,
            "reason": decision.reason,
        }

        if decision.allowed:
            refund = store.issue_refund(order.order_id, self.vault.unmask(inp.reason))
            self._load_order(order.order_id)  # refresh the snapshot: status is now "refunded"
            refund_record = {"order_id": order.order_id, "refunded_amount": refund["refunded_amount"]}
            self.memory.remember_refund(order.order_id, refund_record)
            result |= {"refund_source_id": f"REFUND-{order.order_id}", "refunded_amount": refund["refunded_amount"]}
            sources.add(f"REFUND-{order.order_id}")
        elif decision.code == "over_limit":
            ticket = self._open_ticket(
                f"Refund over ${store.MAX_AUTO_REFUND:.0f} needs approval for order {order.order_id} "
                f"(${order.total:.2f}). Customer reason: {inp.reason}",
                order.order_id,
            )
            result |= {"ticket_source_id": f"TICKET-{ticket['ticket_id']}", "ticket_id": ticket["ticket_id"]}
            sources.add(f"TICKET-{ticket['ticket_id']}")
        return result, sources

    def _tool_escalate(self, args: dict) -> tuple[dict, set[str]]:
        inp = EscalateInput.model_validate(args)
        ticket = self._open_ticket(inp.summary, inp.order_id)
        sid = f"TICKET-{ticket['ticket_id']}"
        return {"source_id": sid, **ticket}, {sid}

    def _open_ticket(self, summary: str, order_id: str | None) -> dict:
        """One ticket per order per session: asking twice returns the existing ticket."""
        if order_id and (existing := self.memory.ticket_for_order(order_id)):
            return existing
        # The human agent reads the real details, so unmask before writing to the store.
        ticket = store.create_ticket(summary=self.vault.unmask(summary), order_id=order_id)
        masked_ticket = {**ticket, "summary": self.vault.mask(ticket["summary"])}
        self.memory.remember_ticket(masked_ticket)
        return masked_ticket
