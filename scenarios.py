"""Live end-to-end run of the README test messages plus memory/PII checks (calls the API).

Run: python scenarios.py
"""

from dotenv import load_dotenv

load_dotenv()

import store  # noqa: E402
from agent import SupportAgent  # noqa: E402

SESSIONS = {
    "README messages": [
        "What's your return policy?",
        "Where is my order A1002?",
        "I want a refund for A1001, the earbuds stopped working.",
        "Refund A1003 please.",
        "I need a refund for A1004.",
    ],
    "memory + PII": [
        "Hi, I'm Maria Lopez, my email is maria.l@example.com. My order is A1002.",
        "When will it arrive?",
        "And what's the tracking number again?",
        "How long does shipping take?",
        "I'd like to talk to a human about it please.",
    ],
    "grounding": [
        "Do you price-match other stores?",
    ],
}


def main() -> None:
    for title, messages in SESSIONS.items():
        print(f"\n===== {title} =====")
        agent = SupportAgent()
        for msg in messages:
            reply = agent.respond(msg)
            r = agent.last_route
            print(f"\nyou>   {msg}")
            print(f"agent> {reply}")
            print(
                f"       route={r.kind} ({r.reason}) llm_calls={agent.last_llm_calls} "
                f"kind={agent.last_reply.kind} citations={agent.last_reply.citations} "
                f"memory_tokens={agent.memory.tokens()}/{agent.memory.token_budget}"
            )
        print(f"\nwhat the LLM saw (masked history):\n{agent.memory.history_messages()[0]['content']}")
    print(f"\ntickets: {store.list_tickets()}")


if __name__ == "__main__":
    main()
