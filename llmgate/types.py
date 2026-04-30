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
from typing import TYPE_CHECKING, Any, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator

# ---------------------------------------------------------------------------
# Streaming fallback strategy
# ---------------------------------------------------------------------------

StreamFallbackMode = Literal["restart", "prefill", "user_turn"]
"""
Controls how llmgate behaves when a streaming model fails mid-stream during
fallback routing (i.e. when ``model`` is a list).

``"restart"`` *(default)*
    Safe and universal. On any failure the next model starts fresh with the
    original messages. No partial text is carried forward.

``"prefill"``
    Buffer-and-resume via assistant prefill. Appends the partial text already
    yielded as an ``{"role": "assistant"}`` message, so the fallback model
    continues from that exact point. Only works with providers that support
    prefill (Gemini, Groq, Mistral, Cohere, Ollama). For providers that don't
    (OpenAI, Anthropic, Azure, Bedrock) llmgate automatically downgrades to
    ``"user_turn"`` and emits a ``UserWarning``.

``"user_turn"``
    Like ``"prefill"`` but wraps the partial text in a ``user`` continuation
    message. Works universally across all providers.
"""

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Vision / multimodal types
# ---------------------------------------------------------------------------


class ImageURL(BaseModel):
    """A URL-based image reference (https:// or data: URI)."""

    url: str
    """Image URL (public HTTPS or base64 data URI of the form ``data:image/jpeg;base64,...``)."""

    detail: Optional[Literal["auto", "low", "high"]] = None
    """Resolution hint. Honoured by OpenAI / Azure; silently ignored by other providers."""


class ImageBytes(BaseModel):
    """An inline base64-encoded image."""

    data: str
    """Raw base64-encoded image bytes (no data-URI prefix)."""

    mime_type: str
    """MIME type, e.g. ``"image/jpeg"``, ``"image/png"``, ``"image/webp"``, ``"image/gif"``."""


class TextPart(BaseModel):
    """A plain-text segment within a multipart message."""

    type: Literal["text"] = "text"
    text: str


class ImagePart(BaseModel):
    """An image segment within a multipart message."""

    type: Literal["image_url", "image_bytes"]
    image_url: Optional[ImageURL] = None
    """Set when ``type='image_url'``."""

    image_bytes: Optional[ImageBytes] = None
    """Set when ``type='image_bytes'``."""

    @model_validator(mode="after")
    def _validate_payload(self) -> "ImagePart":
        if self.type == "image_url" and self.image_url is None:
            raise ValueError("image_url is required when type='image_url'")
        if self.type == "image_bytes" and self.image_bytes is None:
            raise ValueError("image_bytes is required when type='image_bytes'")
        return self


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
    """A single chat message (user / assistant / system / tool).

    ``content`` accepts either a plain string (text-only, the common case) or
    a list of :class:`TextPart` / :class:`ImagePart` objects for multimodal
    (vision) messages.  Existing text-only callers are fully unaffected.
    """

    role: Literal["system", "user", "assistant", "tool"]
    content: Optional[Union[str, list[Union[TextPart, ImagePart]]]] = None
    """Message content — plain string *or* list of text/image parts."""

    # Tool-calling fields
    tool_calls: Optional[list[ToolCall]] = None
    """Populated when role='assistant' and the model requested tool calls."""

    tool_call_id: Optional[str] = None
    """Populated when role='tool' — references the ToolCall.id being answered."""

    name: Optional[str] = None
    """Tool function name. Required by some providers on role='tool' messages."""

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict (text-only; use provider vision helpers for images)."""
        d: dict[str, Any] = {"role": self.role}
        if self.content is not None:
            # For text-only content, serialise directly; image parts are
            # handled separately by the vision normaliser in each provider.
            if isinstance(self.content, str):
                d["content"] = self.content
            else:
                d["content"] = self.content  # pass list as-is (provider will re-serialise)
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
    response_format: Optional[Any] = Field(default=None)
    """
    Pass a Pydantic ``BaseModel`` *class* (not an instance) to request a
    structured output.  The provider will be instructed to return JSON
    conforming to the model's schema; the parsed instance is available at
    ``CompletionResponse.parsed``.

    Example::

        class Movie(BaseModel):
            title: str
            year: int

        resp = completion("gpt-4o-mini", messages, response_format=Movie)
        movie: Movie = resp.parsed
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
    parsed: Optional[Any] = Field(default=None, exclude=True)
    """Validated Pydantic model instance when ``response_format`` was supplied."""

    fallback_attempts: list[str] = Field(default_factory=list)
    """Models that were tried (and failed) before this response was produced.

    Empty list when the first model in the chain succeeded.  Populated when
    fallback routing kicked in — e.g. ``["gpt-4o-mini"]`` means the first model
    failed and the second succeeded.  Useful for observability / logging.
    """

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

    # Fallback observability fields — populated only during multi-model fallback streaming
    fallback_attempts: list[str] = Field(default_factory=list)
    """Models that were tried (and failed) before this chunk's model started streaming.

    Populated on the **first chunk** from each model in the chain; empty list on all
    subsequent chunks from the same model run.  When the primary model succeeded,
    this is always ``[]``.
    """

    resumed_from_partial: bool = False
    """``True`` on the **first chunk from a fallback model** that used ``"prefill"`` or
    ``"user_turn"`` mode to resume mid-stream (i.e. the primary model had already yielded
    some text before failing).  ``False`` in all other cases, including when the fallback
    started fresh (``"restart"`` mode or pre-first-chunk failure).
    """


# ---------------------------------------------------------------------------
# Embeddings types
# ---------------------------------------------------------------------------


class EmbeddingRequest(BaseModel):
    """Input to :func:`~llmgate.embeddings.embed`."""

    model: str
    input: str | list[str]
    """Text or list of texts to embed. A list enables batch embedding in one call."""

    dimensions: Optional[int] = None
    """
    Requested output dimensionality.

    Supported by: OpenAI (text-embedding-3 only), Azure, Gemini, Mistral
    (``output_dimension``), Bedrock Titan V2 (256 / 512 / 1024).
    """

    task_type: Optional[str] = None
    """
    Embedding optimisation hint — **Gemini only**.

    Valid values: ``"RETRIEVAL_DOCUMENT"``, ``"RETRIEVAL_QUERY"``,
    ``"SEMANTIC_SIMILARITY"``, ``"CLASSIFICATION"``, ``"CLUSTERING"``,
    ``"QUESTION_ANSWERING"``, ``"FACT_VERIFICATION"``.

    Using the correct task type meaningfully improves retrieval quality in RAG
    pipelines.  Not supported on the legacy ``models/embedding-001`` model.
    """

    title: Optional[str] = None
    """
    Document title — **Gemini only**, only applicable when
    ``task_type="RETRIEVAL_DOCUMENT"``.

    Providing a title improves the quality of document embeddings.
    """

    input_type: Optional[str] = None
    """
    Embedding purpose hint — **Cohere only**.

    Valid values: ``"search_document"`` (default), ``"search_query"``,
    ``"classification"``, ``"clustering"``.

    **Important:** Always set this correctly — using ``"search_document"``
    for query vectors (or vice-versa) degrades retrieval quality.
    """

    truncate: Optional[str] = None
    """
    How to handle inputs that exceed the model's context window.

    * **Cohere**: ``"NONE"`` (error, default), ``"START"``, ``"END"``.
    * **Ollama**: ``True`` (truncate silently, default) or ``False`` (error).

    Pass a string for Cohere, a boolean-as-string (``"true"`` / ``"false"``)
    or let ``extra_kwargs`` override for Ollama.
    """

    encoding_format: Optional[str] = None
    """
    Output encoding — **OpenAI / Azure / Mistral**.

    ``"float"`` (default) or ``"base64"`` (smaller wire payload; you must
    decode the base64 bytes yourself).
    """

    user: Optional[str] = None
    """
    End-user identifier for abuse monitoring — **OpenAI / Azure only**.
    Forwarded verbatim as the ``user`` parameter.
    """

    extra_kwargs: dict[str, Any] = Field(default_factory=dict)


class EmbeddingResponse(BaseModel):
    """Returned by :func:`~llmgate.embeddings.embed`."""

    model: str
    provider: str
    embeddings: list[list[float]]
    """One embedding vector per input text. Always a list even for single-string input."""

    usage: TokenUsage = Field(default_factory=TokenUsage)
    raw: Any = Field(default=None, exclude=True)


# ---------------------------------------------------------------------------
# Batch types
# ---------------------------------------------------------------------------


class BatchError(BaseModel):
    """Error details for a single failed request within a batch operation."""

    index: int
    """Original index of the failed request in the input list."""

    request: CompletionRequest
    """The request that failed."""

    error: str
    """Human-readable error message."""

    error_type: str
    """Exception class name (e.g. 'RateLimitError', 'BatchTimeoutError')."""


class BatchResult(BaseModel):
    """Aggregated results from a :func:`~llmgate.batch.batch` or
    :func:`~llmgate.batch.abatch` operation.

    ``results`` preserves the original input order and contains ``None`` at
    positions where a request failed.  Cross-reference with ``errors`` (which
    carry the original ``index``) to identify which requests failed and why.
    """

    results: list[Optional[CompletionResponse]]
    """Responses in input order.  ``None`` for every request that failed."""

    errors: list[BatchError] = Field(default_factory=list)
    """Error details for each failed request (empty when all succeeded)."""

    successful: int
    """Number of requests that completed successfully."""

    failed: int
    """Number of requests that raised an exception."""

    total_tokens: int
    """Sum of ``usage.total_tokens`` across all successful responses."""

    @property
    def success_rate(self) -> float:
        """Fraction of requests that succeeded, in the range [0.0, 1.0]."""
        total = self.successful + self.failed
        return self.successful / total if total > 0 else 0.0
