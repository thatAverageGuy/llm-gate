# Custom Provider

Add your own provider in three steps.

## 1. Subclass `BaseProvider`

```python
# my_provider.py
import os
from typing import Any, AsyncIterator, ClassVar, Iterator

from llmgate.base import BaseProvider
from llmgate.exceptions import AuthError, ProviderAPIError
from llmgate.types import (
    Choice, CompletionRequest, CompletionResponse,
    Message, StreamChunk, TokenUsage,
)


class MyProvider(BaseProvider):
    name: ClassVar[str] = "myprovider"
    supported_model_prefixes: ClassVar[tuple[str, ...]] = ("myprovider/",)

    def __init__(self, api_key: str | None = None, **kwargs: Any) -> None:
        try:
            import my_sdk  # noqa: PLC0415
        except ImportError as e:
            raise ImportError("my_sdk required: pip install my-sdk") from e

        key = api_key or os.environ.get("MY_API_KEY")
        if not key:
            raise AuthError("MY_API_KEY not set", provider=self.name)

        self._client = my_sdk.Client(api_key=key)
        self._async_client = my_sdk.AsyncClient(api_key=key)

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        model = self._strip_prefix(request.model)
        try:
            raw = self._client.generate(
                model=model,
                messages=[m.to_dict() for m in request.messages],
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )
        except Exception as exc:
            raise ProviderAPIError(str(exc), provider=self.name) from exc

        return CompletionResponse(
            id=raw.id,
            model=request.model,
            provider=self.name,
            choices=[Choice(
                index=0,
                message=Message(role="assistant", content=raw.text),
                finish_reason=raw.finish_reason,
            )],
            usage=TokenUsage(
                prompt_tokens=raw.usage.input,
                completion_tokens=raw.usage.output,
                total_tokens=raw.usage.total,
            ),
            raw=raw,
        )

    async def acomplete(self, request: CompletionRequest) -> CompletionResponse:
        # ... async version
        ...

    def stream(self, request: CompletionRequest) -> Iterator[StreamChunk]:
        model = self._strip_prefix(request.model)
        for chunk in self._client.stream(model=model, messages=...):
            yield StreamChunk(delta=chunk.text)

    async def astream(self, request: CompletionRequest) -> AsyncIterator[StreamChunk]:
        model = self._strip_prefix(request.model)
        async for chunk in self._async_client.stream(model=model, messages=...):
            yield StreamChunk(delta=chunk.text)
```

## 2. Register it

```python
import llmgate.completion as _registry
from my_provider import MyProvider

# Prepend so your provider takes priority over built-ins for its prefix
_registry._OPTIONAL_PROVIDERS.insert(0, ("myprovider/", "my_provider", "MyProvider"))
```

## 3. Use it

```python
from llmgate import completion

resp = completion("myprovider/my-model-v1", messages)
print(resp.provider)   # "myprovider"
```
