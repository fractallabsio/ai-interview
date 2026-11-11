"""Pydantic models: tool inputs, store records, policy decisions, and the LLM's reply.

Tool-input and reply schemas are sent to the API (strict tools / structured output),
so they avoid Field constraints the API schema subset may not support; the real
checks live in validators, which run on our side.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

ORDER_ID_RE = re.compile(r"^[A-Z]\d{4}$")
# Inline citation markers the model writes into its answer, e.g. [FAQ-2] or [TICKET-T-1A2B3C].
CITATION_RE = re.compile(r"\[((?:FAQ|ORDER|POLICY|REFUND|TICKET)-[A-Za-z0-9-]+)\]")


def normalise_order_id(value: str) -> str:
    value = value.strip().upper()
    if not ORDER_ID_RE.match(value):
        raise ValueError(f"order_id must look like A1001, got {value!r}")
    return value


# ---------- tool inputs (what the model sends us) ----------


class _ToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GetOrderInput(_ToolInput):
    order_id: str = Field(description="Order id, e.g. A1001")

    @field_validator("order_id")
    @classmethod
    def _check_id(cls, v: str) -> str:
        return normalise_order_id(v)


class SearchFaqInput(_ToolInput):
    query: str = Field(description="Keywords describing what the customer wants to know")

    @field_validator("query")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be empty")
        return v.strip()


class RequestRefundInput(_ToolInput):
    order_id: str = Field(description="Order id to refund, e.g. A1001")
    reason: str = Field(description="Short reason the customer gave for the refund")

    @field_validator("order_id")
    @classmethod
    def _check_id(cls, v: str) -> str:
        return normalise_order_id(v)


class EscalateInput(_ToolInput):
    summary: str = Field(description="One-sentence summary of the issue for the human agent")
    order_id: str | None = Field(description="Related order id, or null if none")

    @field_validator("order_id")
    @classmethod
    def _check_id(cls, v: str | None) -> str | None:
        return None if v in (None, "") else normalise_order_id(v)


# ---------- store records ----------


class Order(BaseModel):
    order_id: str
    customer_id: str
    customer_name: str
    item: str
    total: float
    status: str
    delivered_on: date | None = None
    estimated_delivery: date | None = None
    tracking: str | None = None


class RefundDecision(BaseModel):
    allowed: bool
    code: Literal["eligible", "not_delivered", "already_refunded", "outside_window", "over_limit"]
    reason: str
    days_since_delivery: int | None = None


# ---------- the LLM's final reply ----------


class AgentReply(BaseModel):
    """The structured reply the model must return. Validated against the sources it was shown."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["answer", "clarification", "smalltalk", "unable"] = Field(
        description=(
            "answer = states store facts (must cite); clarification = asks the customer for info; "
            "smalltalk = greeting/thanks; unable = no source covers it"
        )
    )
    answer: str = Field(description="Reply to the customer, with [SOURCE-ID] markers after supported sentences")
    citations: list[str] = Field(description="Every source id used in the answer")

    @model_validator(mode="after")
    def _grounded(self, info: ValidationInfo) -> AgentReply:
        if not self.answer.strip():
            raise ValueError("answer must not be empty")
        cited = set(self.citations) | set(CITATION_RE.findall(self.answer))
        if self.kind == "answer" and not cited:
            raise ValueError("kind 'answer' states store facts, so it must cite at least one source id")
        known = (info.context or {}).get("known_sources")
        if known is not None:
            unknown = cited - set(known)
            if unknown:
                raise ValueError(f"cited unknown source ids {sorted(unknown)}; only these exist: {sorted(known)}")
        self.citations = sorted(cited)
        return self
