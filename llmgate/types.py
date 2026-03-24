"""
llmgate.types
~~~~~~~~~~~~~
Shared, provider-neutral Pydantic v2 models used across the library.

These types form the stable public contract of llmgate — every provider maps
its own SDK response onto these models so callers always work with a consistent
interface regardless of which provider is under the hood.
"""
from __future__ import annotations

import json
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Tool / function-calling types
# ---------------------------------------------------------------------------


class FunctionDefinition(BaseModel):
    """Describes a callable tool that the LLM can invoke."""

    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    """JSON Schema object describing the function's parameters."""


class ToolDefinition(BaseModel):
    """Wrapper matching the OpenAI tool-object shape."""

    type: Literal["function"] = "function"
    function: FunctionDefinition


class ToolCall(BaseModel):
    """A tool/function call requested by the model."""

    id: str
    function: str        # function name
    arguments: dict[str, Any] = Field(default_factory=dict)

    def arguments_json(self) -> str:
        """Return arguments serialised as a JSON string."""
        return json.dumps(self.arguments)


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


class Message(BaseModel):
    """A single chat message (user / assistant / system / tool)."""

    role: Literal["system", "user", "assistant", "tool"]
    content: Optional[str] = None
    """Message text. May be None for pure tool-call assistant messages."""

    # Tool-calling fields
    tool_calls: Optional[list[ToolCall]] = None
    """Populated when role='assistant' and the model requested tool calls."""

    tool_call_id: Optional[str] = None
    """Populated when role='tool' — references the ToolCall.id being answered."""

    name: Optional[str] = None
    """Tool function name. Required by some providers on role='tool' messages."""

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"role": self.role}
        if self.content is not None:
            d["content"] = self.content
        if self.tool_calls is not None:
            d["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function,
                        "arguments": tc.arguments_json(),
                    },
                }
                for tc in self.tool_calls
            ]
        if self.tool_call_id is not None:
            d["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            d["name"] = self.name
        return d


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
    tools: Optional[list[ToolDefinition]] = Field(default=None)
    tool_choice: Optional[Any] = Field(default=None)
    """
    "auto" | "none" | {"type": "function", "function": {"name": "..."}}
    Each provider normalises this to its own format internally.
    """
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

    # Convenience shortcuts
    @property
    def text(self) -> str:
        """Return the content of the first choice (empty string if tool-only response)."""
        if not self.choices:
            return ""
        return self.choices[0].message.content or ""

    @property
    def tool_calls(self) -> list[ToolCall]:
        """Return tool calls from the first choice (empty list if none)."""
        if not self.choices:
            return []
        return self.choices[0].message.tool_calls or []


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
