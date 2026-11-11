"""Terminal chat with the support agent. Run: python main.py"""

from dotenv import load_dotenv

load_dotenv()

from agent import SupportAgent  # noqa: E402

import store  # noqa: E402


def main() -> None:
    agent = SupportAgent()
    print("Brightcart support. Type 'quit' to exit, 'tickets' to list open tickets.\n")
    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user:
            continue
        if user == "quit":
            break
        if user == "tickets":
            print(store.list_tickets())
            continue
        reply = agent.respond(user)
        route = agent.last_route
        print(f"agent> {reply}")
        print(f"       [route: {route.kind} - {route.reason} | llm calls: {agent.last_llm_calls}]\n")


if __name__ == "__main__":
    main()
