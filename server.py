"""HTTP API + minimal chat page for the support agent.

Run: uvicorn server:app --port 3000
"""

import uuid
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

import store  # noqa: E402
from agent import SupportAgent  # noqa: E402

app = FastAPI(title="Brightcart support")

# One agent (and so one session memory + PII vault) per chat session, held in process memory.
_sessions: dict[str, SupportAgent] = {}
MAX_SESSIONS = 500


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    session_id: str | None = None


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    kind: str
    citations: list[str]
    route: str
    route_reason: str
    llm_calls: int
    memory_tokens: int
    memory_budget: int


@app.get("/")
def index() -> FileResponse:
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    sid = req.session_id if req.session_id in _sessions else str(uuid.uuid4())
    if sid not in _sessions:
        if len(_sessions) >= MAX_SESSIONS:
            _sessions.pop(next(iter(_sessions)))
        _sessions[sid] = SupportAgent()
    agent = _sessions[sid]
    try:
        reply = agent.respond(req.message)
    except Exception as err:  # surface API/config failures to the UI instead of a bare 500
        raise HTTPException(status_code=502, detail=f"Agent error: {type(err).__name__}: {err}") from err
    return ChatResponse(
        session_id=sid,
        reply=reply,
        kind=agent.last_reply.kind,
        citations=agent.last_reply.citations,
        route=agent.last_route.kind,
        route_reason=agent.last_route.reason,
        llm_calls=agent.last_llm_calls,
        memory_tokens=agent.memory.tokens(),
        memory_budget=agent.memory.token_budget,
    )


@app.get("/tickets")
def tickets() -> list[dict]:
    return store.list_tickets()
