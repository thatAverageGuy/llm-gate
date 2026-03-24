"""
llmgate.base
~~~~~~~~~~~~
Abstract base class for all llmgate providers.

Every provider must implement the two abstract methods:
    - ``complete(request)``  — synchronous completion
    - ``acomplete(request)`` — asynchronous completion

The ``supports(model)`` classmethod is used by the provider registry to route
a model string to the right provider without instantiating it.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from llmgate.types import CompletionRequest, CompletionResponse


class BaseProvider(ABC):
    """Abstract base for all llmgate providers."""

    # -----------------------------------------------------------------------
    # Class-level metadata — must be set on each concrete subclass
    # -----------------------------------------------------------------------

    #: Human-readable name for this provider (e.g. "openai")
    name: ClassVar[str]

    #: Tuple of lowercase model-name prefixes this provider handles.
    #: The registry calls ``supports()`` which checks these prefixes.
    supported_model_prefixes: ClassVar[tuple[str, ...]]

    # -----------------------------------------------------------------------
    # Routing helper
    # -----------------------------------------------------------------------

    @classmethod
    def supports(cls, model: str) -> bool:
        """Return True if this provider can handle ``model``."""
        model_lower = model.lower()
        return any(model_lower.startswith(p) for p in cls.supported_model_prefixes)

    # -----------------------------------------------------------------------
    # Abstract interface
    # -----------------------------------------------------------------------

    @abstractmethod
    def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Perform a synchronous chat completion."""

    @abstractmethod
    async def acomplete(self, request: CompletionRequest) -> CompletionResponse:
        """Perform an asynchronous chat completion."""

    # -----------------------------------------------------------------------
    # Shared helpers for subclasses
    # -----------------------------------------------------------------------

    def _strip_prefix(self, model: str) -> str:
        """
        Strip provider routing prefix from model name if present.

        E.g. "groq/llama-3.1-8b-instant" -> "llama-3.1-8b-instant"

        Prefixes that already end with '/' (like "groq/") are matched as-is;
        prefixes without a trailing slash have one appended for the check.
        """
        for prefix in self.supported_model_prefixes:
            # Normalise: ensure prefix ends with '/' for the startswith check
            normalised = prefix if prefix.endswith("/") else prefix + "/"
            if model.lower().startswith(normalised):
                return model[len(normalised):]
        return model
