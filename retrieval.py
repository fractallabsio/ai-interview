"""Lexical retrieval (BM25, no vector DB) and the hybrid router.

The router answers confident, unambiguous FAQ questions straight from the BM25 hit
(no LLM call). Anything personal, action-shaped, ambiguous or low-scoring goes to the LLM,
which still gets the top BM25 hits as grounded context.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import store
from policy import POLICY_SOURCE_ID, POLICY_TEXT

if TYPE_CHECKING:
    from memory import SessionMemory

# Router thresholds, tuned on the store's FAQ (see tests/test_core.py).
MIN_DIRECT_SCORE = 2.5  # top BM25 score needed to skip the LLM
MIN_DIRECT_MARGIN = 1.5  # top score must beat the runner-up by this factor

STOPWORDS = set(
    """a an the and or but if of to in on at for from by with about into is are was were be been being
    do does did doing have has had i me my we our you your it its this that these those what whats which who
    whom how when where why can could would should will shall may might must there here s t don isn
    please hi hello hey thanks thank any some get got""".split()
)


def _stem(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS and len(t) > 1]


@dataclass(frozen=True)
class Doc:
    source_id: str
    topic: str
    question: str
    answer: str

    @property
    def search_text(self) -> str:
        return f"{self.topic} {self.question} {self.answer}"


def load_corpus() -> list[Doc]:
    # store.py exposes no "list all FAQ entries" call, so read its loaded data directly (read-only).
    docs = [
        Doc(f"FAQ-{i}", e["topic"], e["question"], e["answer"]) for i, e in enumerate(store._DATA["faq"], start=1)
    ]
    docs.append(Doc(POLICY_SOURCE_ID, "refunds", "What are the exact refund rules?", POLICY_TEXT))
    return docs


class BM25:
    def __init__(self, docs: list[Doc], k1: float = 1.5, b: float = 0.75) -> None:
        self.docs = docs
        self.k1, self.b = k1, b
        tokens = [tokenize(d.search_text) for d in docs]
        self.tf = [Counter(t) for t in tokens]
        self.lengths = [len(t) for t in tokens]
        self.avgdl = sum(self.lengths) / len(self.lengths)
        df: Counter[str] = Counter()
        for t in tokens:
            df.update(set(t))
        n = len(docs)
        self.idf = {term: math.log(1 + (n - c + 0.5) / (c + 0.5)) for term, c in df.items()}

    def search(self, query: str, k: int = 3) -> list[tuple[Doc, float]]:
        terms = set(tokenize(query))
        scored = []
        for doc, tf, length in zip(self.docs, self.tf, self.lengths):
            score = 0.0
            for term in terms:
                f = tf.get(term)
                if f:
                    norm = f + self.k1 * (1 - self.b + self.b * length / self.avgdl)
                    score += self.idf[term] * f * (self.k1 + 1) / norm
            if score > 0:
                scored.append((doc, score))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:k]


# ---------- hybrid router ----------

ORDER_ID_IN_TEXT = re.compile(r"\b[A-Za-z]\d{4}\b")
HUMAN_RE = re.compile(
    r"\b(human|real person|a person|someone|agent|representative|rep|manager|supervisor|escalat\w*|speak to|talk to)\b",
    re.I,
)
PERSONAL_RE = re.compile(r"\b(my|mine|i ordered|i bought|i received|i got)\b", re.I)
REQUEST_RE = re.compile(
    r"\b(i\s*(?:want|need|would like|'d like)|can you|could you|please)\b.*\b(refund|return|cancel|ticket|escalat\w*)",
    re.I,
)
FOLLOWUP_RE = re.compile(r"\b(it|that|this|them|again|status|refund|where|when|track\w*|still)\b", re.I)


@dataclass
class Route:
    kind: Literal["faq_direct", "llm"]
    reason: str
    hits: list[tuple[Doc, float]] = field(default_factory=list)


def route_query(masked_text: str, index: BM25, memory: SessionMemory | None = None) -> Route:
    hits = index.search(masked_text, k=3)
    if ORDER_ID_IN_TEXT.search(masked_text):
        return Route("llm", "mentions an order id", hits)
    if HUMAN_RE.search(masked_text):
        return Route("llm", "asks for a human", hits)
    if REQUEST_RE.search(masked_text):
        return Route("llm", "asks for an action", hits)
    if PERSONAL_RE.search(masked_text):
        return Route("llm", "about the customer's own situation", hits)
    if memory is not None and memory.active_order_id and FOLLOWUP_RE.search(masked_text):
        return Route("llm", f"follow-up on active order {memory.active_order_id}", hits)
    if not hits:
        return Route("llm", "no lexical match", hits)
    top = hits[0][1]
    runner_up = hits[1][1] if len(hits) > 1 else 0.0
    if top < MIN_DIRECT_SCORE:
        return Route("llm", f"low lexical confidence ({top:.2f})", hits)
    if runner_up and top < runner_up * MIN_DIRECT_MARGIN:
        return Route("llm", f"ambiguous match ({top:.2f} vs {runner_up:.2f})", hits)
    # The query must overlap the entry's question/topic, not just words in its answer
    # ("payment methods" appears in the refund answer, but that entry doesn't answer it).
    best = hits[0][0]
    if not set(tokenize(masked_text)) & set(tokenize(f"{best.topic} {best.question}")):
        return Route("llm", "match is only in the answer text", hits)
    return Route("faq_direct", f"confident BM25 match ({top:.2f})", hits)
