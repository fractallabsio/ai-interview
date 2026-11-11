"""Reversible PII masking. The LLM only ever sees placeholders like [EMAIL_1];
the vault swaps the real values back in before the reply reaches the customer
or a value is written to the store (e.g. a ticket summary for a human agent)."""

from __future__ import annotations

import re
from collections import Counter

# Order matters: longer / more specific patterns first so a card number isn't half-eaten as a phone.
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("EMAIL", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    ("CARD", re.compile(r"\b(?:\d[ -]?){12,18}\d\b")),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("PHONE", re.compile(r"(?<![\w-])(?:\+?\d{1,3}[ .-]?)?(?:\(\d{3}\)|\d{3})[ .-]?\d{3}[ .-]?\d{4}\b")),
    (
        "ADDRESS",
        re.compile(
            r"\b\d{1,5}\s+(?:[A-Z][a-z]+\s+){1,3}"
            r"(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Way|Place|Pl)\b\.?"
        ),
    ),
]
# "my name is Maria Lopez", "I'm Maria", "this is James Chen" -> mask the name part only.
_NAME_INTRO = re.compile(r"(?i:\bmy name is|\bi am|\bi'm|\bthis is|\bname's)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)")
_TOKEN_RE = re.compile(r"\[(?:EMAIL|CARD|SSN|PHONE|ADDRESS|NAME)_\d+\]")


class PIIVault:
    def __init__(self) -> None:
        self._token_for: dict[str, str] = {}
        self._value_for: dict[str, str] = {}
        self._counts: Counter[str] = Counter()

    def mask_value(self, kind: str, value: str) -> str:
        """Register a known PII value (e.g. a customer name from the store) and return its placeholder."""
        if value not in self._token_for:
            self._counts[kind] += 1
            token = f"[{kind}_{self._counts[kind]}]"
            self._token_for[value] = token
            self._value_for[token] = value
        return self._token_for[value]

    def mask(self, text: str) -> str:
        # Values we already know (from earlier turns or store records), longest first.
        for value in sorted(self._token_for, key=len, reverse=True):
            text = text.replace(value, self._token_for[value])
        for kind, pattern in _PATTERNS:
            text = pattern.sub(lambda m, k=kind: self.mask_value(k, m.group(0)), text)
        return _NAME_INTRO.sub(lambda m: m.group(0).replace(m.group(1), self.mask_value("NAME", m.group(1))), text)

    def unmask(self, text: str) -> str:
        return _TOKEN_RE.sub(lambda m: self._value_for.get(m.group(0), m.group(0)), text)
