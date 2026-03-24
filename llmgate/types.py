"""
llmgate.types
~~~~~~~~~~~~~
Shared, provider-neutral Pydantic v2 models used across the library.

These types form the stable public contract of llmgate — every provider maps
its own SDK response onto these models so callers always work with a consistent
interface regardless of which provider is under the hood.
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


class Message(BaseModel):
    """A single chat message (user / assistant / system / tool)."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


class CompletionRequest(BaseModel):
    """
    Provider-agnostic completion request.

    Extra provider-specific kwargs can be passed via ``extra_kwargs`` and are
    forwarded verbatim to the underlying SDK call.
    """

    model: str
    messages: list[Message]
    max_tokens: Optional[int] = Field(default=None)
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    stream: bool = Field(default=False)
    extra_kwargs: dict[str, Any] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------


class TokenUsage(BaseModel):
    """Token usage statistics returned by the provider."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class Choice(BaseModel):
    """A single completion choice."""

    index: int
    message: Message
    finish_reason: Optional[str] = None


class CompletionResponse(BaseModel):
    """
    Provider-agnostic completion response.

    ``provider``  — the name of the provider that served the request.
    ``raw``       — the raw SDK response object (if you need provider-specific
                    fields not captured here). Not serialised by default.
    """

    id: str
    model: str
    provider: str
    choices: list[Choice]
    usage: TokenUsage = Field(default_factory=TokenUsage)
    raw: Optional[Any] = Field(default=None, exclude=True)

    # Convenience shortcut
    @property
    def text(self) -> str:
        """Return the content of the first choice."""
        return self.choices[0].message.content if self.choices else ""


class StreamChunk(BaseModel):
    """
    A single streamed delta chunk, yielded by ``completion(..., stream=True)``
    and ``acompletion(..., stream=True)``.

    Iterate over these to build the full response incrementally::

        for chunk in completion("gpt-4o-mini", messages, stream=True):
            print(chunk.delta, end="", flush=True)
    """

    id: str
    model: str
    provider: str
    delta: str                              # text fragment for this chunk
    finish_reason: Optional[str] = None     # set on the final chunk
    index: int = 0
    usage: Optional[TokenUsage] = None      # populated on the last chunk by some providers
