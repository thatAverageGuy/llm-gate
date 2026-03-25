"""
llmgate.structured
~~~~~~~~~~~~~~~~~~
Shared utilities for structured output (Pydantic ``response_format``).

Providers pull these helpers to:
  1. Generate a JSON Schema from any Pydantic model.
  2. Extract a JSON object from free-form LLM text (strips markdown fences).
  3. Validate the extracted JSON against the model.
  4. Inject a schema-describing system message for providers without native
     JSON mode.
"""
from __future__ import annotations

import json
import re
from typing import Any, TypeVar, TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic import BaseModel

T = TypeVar("T")


# ---------------------------------------------------------------------------
# JSON Schema generation
# ---------------------------------------------------------------------------

def get_json_schema(model_cls: type[BaseModel]) -> dict[str, Any]:
    """Return the JSON Schema dict for a Pydantic model (v2 API)."""
    return model_cls.model_json_schema()


# ---------------------------------------------------------------------------
# JSON extraction from free-form text
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(
    r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```",
    re.DOTALL,
)
_BARE_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_json(text: str) -> str:
    """
    Extract the first JSON object from LLM text.

    Handles:
    - Bare JSON objects / arrays
    - Markdown code fences: `` ```json { ... } ``` ``
    """
    # Try fenced block first
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1).strip()

    # Fall back to first { ... } block
    m = _BARE_OBJ_RE.search(text)
    if m:
        return m.group(0).strip()

    # Return as-is and let json.loads raise a useful error
    return text.strip()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_parsed(text: str | None, model_cls: type[T]) -> T:  # type: ignore[type-var]
    """
    Extract JSON from ``text`` and validate it against ``model_cls``.

    Raises:
        ``pydantic.ValidationError`` if the JSON doesn't match the schema.
        ``json.JSONDecodeError`` if the text isn't valid JSON after extraction.
    """
    from pydantic import BaseModel  # noqa: PLC0415 — lazy import avoids circular dep

    raw = extract_json(text or "")
    data = json.loads(raw)
    if not isinstance(model_cls, type) or not issubclass(model_cls, BaseModel):
        raise TypeError(f"response_format must be a Pydantic BaseModel subclass, got {model_cls!r}")
    return model_cls.model_validate(data)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Prompt injection for providers without native JSON support
# ---------------------------------------------------------------------------

_SCHEMA_SYSTEM_TEMPLATE = """\
You must respond with a single valid JSON object that matches the following \
JSON Schema exactly. Do NOT include any explanatory text, markdown fences, \
or any content outside the JSON object.

JSON Schema:
{schema}
"""


def build_schema_system_message(model_cls: type[BaseModel]) -> str:
    """Return a system prompt instructing the model to reply with valid JSON."""
    schema = json.dumps(get_json_schema(model_cls), indent=2)
    return _SCHEMA_SYSTEM_TEMPLATE.format(schema=schema)


def inject_schema_prompt(
    messages: list[Any],
    model_cls: type[BaseModel],
) -> list[Any]:
    """
    Return a copy of ``messages`` with a structured-output system instruction
    prepended (if no system message exists) or merged with the existing one.

    Each element must be a ``Message`` instance.
    """
    from llmgate.types import Message  # noqa: PLC0415

    instruction = build_schema_system_message(model_cls)
    msgs = list(messages)

    # Look for an existing system message
    for i, m in enumerate(msgs):
        if m.role == "system":
            msgs[i] = Message(
                role="system",
                content=(m.content or "") + "\n\n" + instruction,
            )
            return msgs

    # Prepend a fresh system message
    msgs.insert(0, Message(role="system", content=instruction))
    return msgs
