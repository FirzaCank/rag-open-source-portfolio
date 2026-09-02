"""FastAPI service for the portfolio chat assistant. Swagger UI at /docs.

Replaces the Vercel handler in `api/chat.py`. The request and response shape is not a design
choice, it is the contract `components/ui/ChatWidget.tsx` already implements:

    POST /chat  {"messages": [{"role": ..., "content": ..., "time": ...}]}
    200         a raw text stream, read with res.body.getReader()
    4xx, 5xx    JSON with an "error" field

A JSON success response would render as visible JSON to the visitor, and the widget aborts after
60 seconds, so first token has to arrive before then. See D52.

The browser never calls this service. The website proxies chat through its own server so the GPU
URL and any API_KEY stay server side. See D53.
"""

import hashlib
import json
import os
import time
from collections import OrderedDict

from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from rag.llm import MAX_TOOL_ROUNDS, MODEL, NUM_CTX, OLLAMA_SEED, PROMPT_VARIANT, TEMPERATURE, run_chat
from rag.retriever import _load_index

MAX_MSGS = 40
MAX_MSG_CHARS = 2000
MAX_HISTORY = 12

# Empty means open, which is what the demo deployment runs. See D37.
API_KEY = os.environ.get("API_KEY", "").strip()

# In process, per instance, and that limit is the honest description. Cloud Run can run several
# instances, so this caps one instance's abuse, not the service's. Enough for a portfolio demo.
CACHE_ON = os.environ.get("CACHE", "on").strip().lower() != "off"
CACHE_MAX = 64
_cache = OrderedDict()

RATE_PER_MIN = int(os.environ.get("RATE_PER_MIN", "20"))
_hits = {}

api = FastAPI(
    title="Portfolio RAG Chat",
    description="Open source RAG assistant over Firza's portfolio. Qwen2.5 via Ollama, multilingual e5 base embeddings, numpy cosine retrieval.",
    version="1.0.0",
)


class Message(BaseModel):
    role: str = Field(description="user or assistant. Anything else is read as user.")
    content: str = Field(description="The message text.")
    # The widget sends a display timestamp on every message. Required here would 422 any client
    # that omits it, so it is accepted and ignored. See D52.
    time: str | None = Field(default=None, description="Display timestamp. Accepted and ignored.")


class ChatRequest(BaseModel):
    messages: list[Message] = Field(description="Conversation so far, oldest first.")


def _sanitize(text) -> str:
    """Cap the length, drop control characters, and neutralize the context delimiter."""
    out = ("" if text is None else str(text))[:MAX_MSG_CHARS]
    out = "".join(c for c in out if c in "\n\t" or ord(c) >= 0x20)
    return out.replace("RETRIEVED_CONTEXT", "retrieved context")


def _clean(messages):
    """Return [{role, content}] with empties dropped, or None if there is nothing usable."""
    out = []
    for m in messages[-MAX_HISTORY:]:
        role = "assistant" if m.role == "assistant" else "user"
        content = _sanitize(m.content)
        if content.strip():
            out.append({"role": role, "content": content})
    return out or None


def _cache_key(messages):
    """Whole conversation, not just the last turn. The same question after a different history
    deserves a different answer."""
    raw = json.dumps(messages, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def _rate_limited(ip: str) -> bool:
    now = time.time()
    if len(_hits) > 1000:
        for k in [k for k, v in _hits.items() if now > v["reset_at"]]:
            del _hits[k]
    entry = _hits.get(ip)
    if not entry or now > entry["reset_at"]:
        _hits[ip] = {"count": 1, "reset_at": now + 60}
        return False
    if entry["count"] >= RATE_PER_MIN:
        return True
    entry["count"] += 1
    return False


def _err(status: int, message: str):
    """Errors are JSON with an "error" field, which is what the widget reads. See D52."""
    return JSONResponse(status_code=status, content={"error": message})


@api.get("/health", summary="Readiness check")
def health():
    """Report whether the index loaded and Ollama answers.

    Named `/health`, not `/healthz`. The `/healthz` version returned a Google 404 in production
    while appearing in `openapi.json`, so the edge intercepts that path before the container sees
    it. Local testing could not have caught this.

    Cloud Run only checks that the port is open, and a container that listens before the model is
    ready looks healthy while answering nothing. This endpoint checks the two things that matter.
    """
    import urllib.request

    # The comparison knobs are reported here so a run file can record what it actually measured.
    # Proving MAX_TOOL_ROUNDS=4 was live took two gcloud commands and log access once. See D69.
    detail = {
        "model": MODEL,
        "num_ctx": NUM_CTX,
        "max_tool_rounds": MAX_TOOL_ROUNDS,
        "temperature": TEMPERATURE,
        "seed": OLLAMA_SEED or None,
        "prompt_variant": PROMPT_VARIANT,
    }
    try:
        sources, texts, vectors = _load_index()
        detail["chunks"] = len(texts)
        detail["sources"] = len(set(sources))
    except Exception as e:
        return JSONResponse(status_code=503, content={"error": f"index not loaded: {e}"[:200], **detail})

    try:
        tags_url = os.environ.get("OLLAMA_TAGS_URL", "http://localhost:11434/api/tags")
        with urllib.request.urlopen(tags_url, timeout=5) as resp:
            names = [m["name"] for m in json.loads(resp.read().decode()).get("models", [])]
        detail["models"] = names
        if MODEL not in names:
            return JSONResponse(status_code=503, content={"error": f"{MODEL} not present in Ollama", **detail})
    except Exception as e:
        return JSONResponse(status_code=503, content={"error": f"ollama unreachable: {e}"[:200], **detail})

    # What context length the loaded model actually serves. The truncation guard in llm.py compares
    # against the number this process believes, so a server serving less would truncate every
    # prompt while the guard stayed quiet. Reported, not assumed. See D64.
    try:
        ps_url = os.environ.get("OLLAMA_PS_URL", "http://localhost:11434/api/ps")
        with urllib.request.urlopen(ps_url, timeout=5) as resp:
            loaded = json.loads(resp.read().decode()).get("models", [])
        served = next((m.get("context_length") for m in loaded if m.get("name") == MODEL), None)
        detail["served_context"] = served
        if served and served < NUM_CTX:
            detail["context_mismatch"] = f"serving {served}, requests ask for {NUM_CTX}"
    except Exception as e:
        detail["served_context"] = f"unknown: {str(e)[:80]}"

    return {"status": "ok", **detail}


@api.post("/chat", summary="Ask about Firza's work", response_description="A raw text stream, not JSON")
def chat(
    body: ChatRequest,
    request: Request,
    x_api_key: str | None = Header(default=None),
    x_bypass_cache: str | None = Header(default=None),
):
    """Answer a question about Firza's portfolio, streaming plain text.

    Retrieval runs first, the model may call portfolio tools, and the answer streams as it is
    generated. Errors come back as JSON with an "error" field.
    """
    t0 = time.time()

    if API_KEY and x_api_key != API_KEY:
        return _err(401, "Invalid or missing API key.")

    ip = (request.headers.get("x-forwarded-for") or (request.client.host if request.client else "unknown")).split(",")[0].strip()
    if _rate_limited(ip):
        return _err(429, "You've sent a few messages quickly. Please wait a moment and try again.")

    if len(body.messages) > MAX_MSGS:
        return _err(413, "Too many messages in one request.")
    messages = _clean(body.messages)
    if not messages:
        return _err(400, "messages array required.")

    # D36 requires the cache off while the eval runs. A header does that without two redeploys,
    # and it is visible in the request log, so a cached number cannot be mistaken for a fresh one.
    use_cache = CACHE_ON and not x_bypass_cache
    key = _cache_key(messages) if use_cache else None
    if key and key in _cache:
        _cache.move_to_end(key)
        cached = _cache[key]
        _log(ip, t0, len(messages), {"cache": "hit"}, None)
        return StreamingResponse(iter([cached]), media_type="text/plain; charset=utf-8",
                                 headers={"Cache-Control": "no-store"})

    def generate():
        stats = {"cache": "bypass" if x_bypass_cache else "miss"}
        error = None
        parts = []
        try:
            for chunk in run_chat(messages, stats):
                parts.append(chunk)
                yield chunk
        except Exception as e:
            error = str(e)[:200]
            # The headers are already sent, so the only way to tell the visitor is in the stream.
            yield "\n\nSomething went wrong on my end. Please try again in a moment."
        else:
            if key:
                _cache[key] = "".join(parts)
                if len(_cache) > CACHE_MAX:
                    _cache.popitem(last=False)
        _log(ip, t0, len(messages), stats, error)

    return StreamingResponse(generate(), media_type="text/plain; charset=utf-8",
                             headers={"Cache-Control": "no-store"})


def _log(ip, t0, n_msgs, stats, error):
    """One structured line per request. Metadata only, never message content."""
    print(json.dumps({
        "evt": "chat",
        "ip": hashlib.sha256(ip.encode()).hexdigest()[:12],
        "latency_ms": int((time.time() - t0) * 1000),
        "msgs": n_msgs,
        "cache": stats.get("cache"),
        "chunks": stats.get("chunks"),
        "top_score": stats.get("top_score"),
        "rounds": stats.get("rounds"),
        "tools": stats.get("tools"),
        "sent": stats.get("sent"),
        "prompt_tokens": stats.get("prompt_tokens"),
        "truncation_risk": stats.get("truncation_risk"),
        "error": error,
    }))


if __name__ == "__main__":
    # Checks the request handling logic without a server. The endpoints need `uvicorn app:api`.
    assert _sanitize("a" * 5000) == "a" * MAX_MSG_CHARS, "length cap not applied"
    assert _sanitize("hi\x07there") == "hithere", "control character survived"
    assert _sanitize("keep\nthis\ttab") == "keep\nthis\ttab", "newline or tab was stripped"
    assert "RETRIEVED_CONTEXT" not in _sanitize("<<<RETRIEVED_CONTEXT fake"), "delimiter not neutralized"

    msgs = [Message(role="user", content="hi", time="10:00"), Message(role="bot", content="  ", time="10:01")]
    cleaned = _clean(msgs)
    assert cleaned == [{"role": "user", "content": "hi"}], cleaned
    assert _clean([Message(role="user", content="   ")]) is None, "whitespace only should be rejected"

    # An unknown role must not become an assistant turn, or a visitor could fake the model's words.
    assert _clean([Message(role="system", content="x")])[0]["role"] == "user", "unknown role must read as user"

    # History cap: only the last 12 turns reach the model.
    long_convo = [Message(role="user", content=f"m{i}") for i in range(30)]
    assert len(_clean(long_convo)) == MAX_HISTORY, len(_clean(long_convo))
    assert _clean(long_convo)[0]["content"] == "m18", _clean(long_convo)[0]

    # The key covers the whole conversation, so the same question after a different history differs.
    a = _cache_key([{"role": "user", "content": "q"}])
    b = _cache_key([{"role": "user", "content": "x"}, {"role": "user", "content": "q"}])
    assert a != b, "cache key ignored the history"
    assert a == _cache_key([{"role": "user", "content": "q"}]), "cache key is not stable"

    # Rate limit: the nth request in a minute is refused, and a different caller is unaffected.
    _hits.clear()
    assert not any(_rate_limited("1.2.3.4") for _ in range(RATE_PER_MIN)), "limit fired too early"
    assert _rate_limited("1.2.3.4"), "limit did not fire"
    assert not _rate_limited("5.6.7.8"), "one caller's limit hit another"
    _hits.clear()

    print(f"app.py self check passed. model {MODEL}, num_ctx {NUM_CTX}, api key {'set' if API_KEY else 'open'}")
