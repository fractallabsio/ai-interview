# Live challenge: mini customer support agent

You have about 28 minutes to build, then we'll test it together for a few minutes. Think out loud. We care more about how you reason than about finishing everything.

## The setup

Brightcart is an online electronics store. Customers chat with a support assistant. This project already has a chat loop, a working Claude call, and a fake backend (`store.py`) with orders, an FAQ, refunds, and support tickets.

Right now the assistant can chat but can't look anything up. Make it a real support agent.

## Get it running (about 3 minutes)

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install --index-url https://pypi.org/simple -r requirements.txt
cp .env.example .env             # paste the API key the interviewer gave you
python main.py
```

Type `hi` and you should get a reply.


## Files

| File               | What it is                                        | Change it?                  |
| ------------------ | ------------------------------------------------- | --------------------------- |
| `agent.py`         | The agent, with a working Claude call and TODOs   | Yes, this is where you work |
| `main.py`          | Terminal chat loop                                | Only if you want to         |
| `store.py`         | Fake store backend: orders, FAQ, refunds, tickets | No (read it)                |
| `data/store.json`  | The data `store.py` loads                         | No                          |
| `requirements.txt` | `anthropic` and `python-dotenv`                | Add to it if you need to    |
| `.env.example`     | Copy it to `.env` and paste in the API key        | No                          |

## What the agent should do

1. Answer general questions (shipping, refunds, warranty) from the store's FAQ, not from the model's own guesses.
2. Look up an order by id and tell the customer its status.
3. Issue a refund when the store's policy allows it:
   - the order status is `delivered`
   - it was delivered no more than 30 days ago (`store.TODAY` is "today")
   - the total is $500 or less
4. When a refund isn't allowed, say why. If the order is over $500, or the customer asks for a human, open a ticket for a human agent and give the customer the ticket id.
5. Keep context across turns. If the customer gave an order id two messages ago, don't ask for it again.

## Ground rules

- `store.issue_refund()` does no checking on its own. Making sure only valid refunds happen is up to you.
- You can use any library, the official docs, and web search.
- The starter uses the [google-genai](https://googleapis.github.io/python-genai/) SDK with `client.models.generate_content`. You can switch to `client.chats`, the Interactions API, or a framework you like (Pydantic AI, LangChain, Google ADK). If you switch to TypeScript, the data is plain JSON, but setup time comes out of your 28 minutes.
- Hard-coding is fine where it saves time. Tell us what you'd change with more time.

## Try these when you're done

```text
What's your return policy?
Where is my order A1002?
I want a refund for A1001, the earbuds stopped working.
Refund A1003 please.
I need a refund for A1004.
```

## If you finish early

Pick one: expose the agent over a FastAPI `POST /chat` endpoint, build a small React + Tailwind chat UI for it, stream the replies, or write a script that runs the test messages and checks the results.
