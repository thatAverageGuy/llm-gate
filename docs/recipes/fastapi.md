# FastAPI Integration

A production-ready FastAPI service using `acompletion` with proper error handling.

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llmgate import acompletion, LLMGate
from llmgate.middleware import RetryMiddleware, LoggingMiddleware
from llmgate.exceptions import AuthError, RateLimitError, ProviderAPIError


# One LLMGate instance for the app lifetime
gate = LLMGate(middleware=[
    RetryMiddleware(max_retries=2),
    LoggingMiddleware(),
])

app = FastAPI(title="LLM API", version="1.0.0")


# ── Request / Response models ────────────────────────────────────────────────

class ChatRequest(BaseModel):
    model: str = "groq/llama-3.3-70b-versatile"
    messages: list[dict]
    max_tokens: int | None = None
    temperature: float | None = None

class ChatResponse(BaseModel):
    text: str
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


# ── Routes ───────────────────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        resp = await gate.acompletion(
            req.model,
            req.messages,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
        )
    except AuthError as e:
        raise HTTPException(status_code=401, detail=f"Invalid API key for {e.provider}")
    except RateLimitError as e:
        raise HTTPException(status_code=429, detail=f"Rate limited by {e.provider}")
    except ProviderAPIError as e:
        raise HTTPException(status_code=502, detail=str(e))

    return ChatResponse(
        text=resp.text,
        provider=resp.provider,
        model=resp.model,
        prompt_tokens=resp.usage.prompt_tokens,
        completion_tokens=resp.usage.completion_tokens,
        total_tokens=resp.usage.total_tokens,
    )


@app.get("/health")
async def health():
    return {"status": "ok"}
```

Run with:

```bash
uvicorn main:app --reload
```

---

## Streaming endpoint

```python
from fastapi.responses import StreamingResponse
from llmgate import acompletion

@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    async def generate():
        async for chunk in await acompletion(req.model, req.messages, stream=True):
            yield f"data: {chunk.delta}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
```
