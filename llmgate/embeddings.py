"""
llmgate.embeddings
~~~~~~~~~~~~~~~~~~
Text embedding API — ``embed()`` and ``aembed()``.

Supported providers and their model-prefix routing:

    text-embedding-*      → OpenAI  (prefix optional; auto-detected)
    gemini/<model>        → Gemini  (e.g. ``gemini/text-embedding-004``)
    azure/<deployment>    → Azure OpenAI
    cohere/<model>        → Cohere  (e.g. ``cohere/embed-english-v3.0``)
    mistral/<model>       → Mistral (e.g. ``mistral/mistral-embed``)
    ollama/<model>        → Ollama  (e.g. ``ollama/nomic-embed-text``)
    bedrock/<model>       → AWS Bedrock

Anthropic and Groq do not offer embedding APIs and will raise
:class:`~llmgate.exceptions.EmbeddingsNotSupported`.

Usage::

    from llmgate import embed

    # Single text
    resp = embed("text-embedding-3-small", "Hello world")
    vector = resp.embeddings[0]   # list[float]

    # Batch
    resp = embed("text-embedding-3-small", ["Hello", "world"])
    vectors = resp.embeddings     # list[list[float]]

    # Async
    resp = await aembed("gemini/text-embedding-004", "Hello")
"""
from __future__ import annotations

import os
from typing import Any

from llmgate.exceptions import AuthError, EmbeddingsNotSupported, ProviderAPIError
from llmgate.types import EmbeddingRequest, EmbeddingResponse, TokenUsage


# ---------------------------------------------------------------------------
# Provider implementations
# ---------------------------------------------------------------------------


def _embed_openai(
    request: EmbeddingRequest,
    api_key: str | None,
    *,
    azure_endpoint: str | None = None,
    api_version: str | None = None,
) -> EmbeddingResponse:
    try:
        import openai  # noqa: PLC0415
    except ImportError as e:
        raise ImportError("openai package required: pip install openai") from e

    if azure_endpoint:
        client = openai.AzureOpenAI(
            api_key=api_key or os.environ.get("AZURE_OPENAI_API_KEY"),
            azure_endpoint=azure_endpoint,
            api_version=api_version or os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        )
        provider_name = "azure"
    else:
        client = openai.OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        provider_name = "openai"

    inputs = request.input if isinstance(request.input, list) else [request.input]
    kwargs: dict[str, Any] = {"model": request.model, "input": inputs}
    if request.dimensions is not None:
        kwargs["dimensions"] = request.dimensions
    kwargs.update(request.extra_kwargs)

    try:
        raw = client.embeddings.create(**kwargs)
    except openai.AuthenticationError as exc:
        raise AuthError(str(exc), provider=provider_name) from exc
    except Exception as exc:
        raise ProviderAPIError(str(exc), provider=provider_name) from exc

    embeddings = [item.embedding for item in sorted(raw.data, key=lambda x: x.index)]
    usage = TokenUsage(
        prompt_tokens=raw.usage.prompt_tokens if raw.usage else 0,
        total_tokens=raw.usage.total_tokens if raw.usage else 0,
    )
    return EmbeddingResponse(
        model=request.model,
        provider=provider_name,
        embeddings=embeddings,
        usage=usage,
        raw=raw,
    )


async def _aembed_openai(
    request: EmbeddingRequest,
    api_key: str | None,
    *,
    azure_endpoint: str | None = None,
    api_version: str | None = None,
) -> EmbeddingResponse:
    try:
        import openai  # noqa: PLC0415
    except ImportError as e:
        raise ImportError("openai package required: pip install openai") from e

    if azure_endpoint:
        client = openai.AsyncAzureOpenAI(
            api_key=api_key or os.environ.get("AZURE_OPENAI_API_KEY"),
            azure_endpoint=azure_endpoint,
            api_version=api_version or os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        )
        provider_name = "azure"
    else:
        client = openai.AsyncOpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        provider_name = "openai"

    inputs = request.input if isinstance(request.input, list) else [request.input]
    kwargs: dict[str, Any] = {"model": request.model, "input": inputs}
    if request.dimensions is not None:
        kwargs["dimensions"] = request.dimensions
    kwargs.update(request.extra_kwargs)

    try:
        raw = await client.embeddings.create(**kwargs)
    except openai.AuthenticationError as exc:
        raise AuthError(str(exc), provider=provider_name) from exc
    except Exception as exc:
        raise ProviderAPIError(str(exc), provider=provider_name) from exc

    embeddings = [item.embedding for item in sorted(raw.data, key=lambda x: x.index)]
    usage = TokenUsage(
        prompt_tokens=raw.usage.prompt_tokens if raw.usage else 0,
        total_tokens=raw.usage.total_tokens if raw.usage else 0,
    )
    return EmbeddingResponse(
        model=request.model,
        provider=provider_name,
        embeddings=embeddings,
        usage=usage,
        raw=raw,
    )


def _embed_gemini(request: EmbeddingRequest, api_key: str | None) -> EmbeddingResponse:
    try:
        from google import genai  # noqa: PLC0415
    except ImportError as e:
        raise ImportError("google-genai package required: pip install google-genai") from e

    resolved_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    client = genai.Client(api_key=resolved_key)

    # Strip "gemini/" prefix if present
    model_name = request.model
    if model_name.startswith("gemini/"):
        model_name = model_name[len("gemini/"):]

    inputs = request.input if isinstance(request.input, list) else [request.input]

    try:
        all_embeddings: list[list[float]] = []
        total_tokens = 0
        for text in inputs:
            kwargs: dict[str, Any] = {"model": model_name, "contents": text}
            if request.dimensions is not None:
                kwargs["config"] = {"output_dimensionality": request.dimensions}
            raw = client.models.embed_content(**kwargs)
            vals = getattr(raw.embeddings[0] if hasattr(raw, "embeddings") else raw.embedding, "values", None)
            if vals is None:
                raw_emb = raw.embeddings[0] if hasattr(raw, "embeddings") else raw.embedding
                vals = list(raw_emb) if hasattr(raw_emb, "__iter__") else []
            all_embeddings.append(list(vals))
            total_tokens += len(text.split())  # approximate
    except Exception as exc:
        raise ProviderAPIError(str(exc), provider="gemini") from exc

    return EmbeddingResponse(
        model=request.model,
        provider="gemini",
        embeddings=all_embeddings,
        usage=TokenUsage(total_tokens=total_tokens),
        raw=None,
    )


async def _aembed_gemini(request: EmbeddingRequest, api_key: str | None) -> EmbeddingResponse:
    import asyncio  # noqa: PLC0415
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _embed_gemini, request, api_key)


def _embed_cohere(request: EmbeddingRequest, api_key: str | None) -> EmbeddingResponse:
    try:
        import cohere  # noqa: PLC0415
    except ImportError as e:
        raise ImportError("cohere package required: pip install cohere") from e

    resolved_key = api_key or os.environ.get("COHERE_API_KEY")
    client = cohere.ClientV2(api_key=resolved_key)

    model_name = request.model
    if model_name.startswith("cohere/"):
        model_name = model_name[len("cohere/"):]

    inputs = request.input if isinstance(request.input, list) else [request.input]
    kwargs: dict[str, Any] = {
        "model": model_name,
        "texts": inputs,
        "input_type": request.extra_kwargs.pop("input_type", "search_document"),
        "embedding_types": ["float"],
    }
    kwargs.update(request.extra_kwargs)

    try:
        raw = client.embed(**kwargs)
    except Exception as exc:
        raise ProviderAPIError(str(exc), provider="cohere") from exc

    # Cohere V2: raw.embeddings.float_ is list[list[float]]
    emb_data = getattr(raw.embeddings, "float_", None) or getattr(raw.embeddings, "float", None) or []
    return EmbeddingResponse(
        model=request.model,
        provider="cohere",
        embeddings=[list(e) for e in emb_data],
        usage=TokenUsage(
            prompt_tokens=getattr(raw.meta.billed_units, "input_tokens", 0) if getattr(raw, "meta", None) else 0,
            total_tokens=getattr(raw.meta.billed_units, "input_tokens", 0) if getattr(raw, "meta", None) else 0,
        ),
        raw=raw,
    )


async def _aembed_cohere(request: EmbeddingRequest, api_key: str | None) -> EmbeddingResponse:
    import asyncio  # noqa: PLC0415
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _embed_cohere, request, api_key)


def _embed_mistral(request: EmbeddingRequest, api_key: str | None) -> EmbeddingResponse:
    try:
        from mistralai import Mistral  # noqa: PLC0415
    except ImportError as e:
        raise ImportError("mistralai package required: pip install mistralai") from e

    resolved_key = api_key or os.environ.get("MISTRAL_API_KEY")
    client = Mistral(api_key=resolved_key)

    model_name = request.model
    if model_name.startswith("mistral/"):
        model_name = model_name[len("mistral/"):]

    inputs = request.input if isinstance(request.input, list) else [request.input]

    try:
        raw = client.embeddings.create(model=model_name, inputs=inputs)
    except Exception as exc:
        raise ProviderAPIError(str(exc), provider="mistral") from exc

    embeddings = [item.embedding for item in sorted(raw.data, key=lambda x: x.index)]
    usage = raw.usage
    return EmbeddingResponse(
        model=request.model,
        provider="mistral",
        embeddings=embeddings,
        usage=TokenUsage(
            prompt_tokens=usage.prompt_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
        ),
        raw=raw,
    )


async def _aembed_mistral(request: EmbeddingRequest, api_key: str | None) -> EmbeddingResponse:
    import asyncio  # noqa: PLC0415
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _embed_mistral, request, api_key)


def _embed_ollama(request: EmbeddingRequest, api_key: str | None) -> EmbeddingResponse:  # noqa: ARG001
    try:
        import ollama  # noqa: PLC0415
    except ImportError as e:
        raise ImportError("ollama package required: pip install llmgate[ollama]") from e

    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    client = ollama.Client(host=host)

    model_name = request.model
    if model_name.startswith("ollama/"):
        model_name = model_name[len("ollama/"):]

    inputs = request.input if isinstance(request.input, list) else [request.input]

    try:
        all_embeddings: list[list[float]] = []
        for text in inputs:
            raw = client.embed(model=model_name, input=text)
            vecs = raw.embeddings if hasattr(raw, "embeddings") else raw.get("embeddings", [])
            all_embeddings.extend([list(v) for v in vecs])
    except Exception as exc:
        raise ProviderAPIError(
            f"Ollama error: {exc}. Is Ollama running?", provider="ollama"
        ) from exc

    return EmbeddingResponse(
        model=request.model,
        provider="ollama",
        embeddings=all_embeddings,
        raw=None,
    )


async def _aembed_ollama(request: EmbeddingRequest, api_key: str | None) -> EmbeddingResponse:
    import asyncio  # noqa: PLC0415
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _embed_ollama, request, api_key)


def _embed_bedrock(request: EmbeddingRequest, api_key: str | None) -> EmbeddingResponse:  # noqa: ARG001
    import json as _json  # noqa: PLC0415
    try:
        import boto3  # noqa: PLC0415
    except ImportError as e:
        raise ImportError("boto3 required: pip install llmgate[bedrock]") from e

    model_id = request.model
    if model_id.startswith("bedrock/"):
        model_id = model_id[len("bedrock/"):]

    client = boto3.client("bedrock-runtime")
    inputs = request.input if isinstance(request.input, list) else [request.input]

    try:
        all_embeddings: list[list[float]] = []
        total_tokens = 0
        for text in inputs:
            body_dict: dict[str, Any] = {"inputText": text}
            if "cohere" in model_id.lower():
                body_dict = {"texts": [text], "input_type": "search_document"}
            raw = client.invoke_model(
                modelId=model_id,
                body=_json.dumps(body_dict),
                contentType="application/json",
                accept="application/json",
            )
            resp_body = _json.loads(raw["body"].read())
            if "embedding" in resp_body:
                all_embeddings.append(resp_body["embedding"])
            elif "embeddings" in resp_body:
                all_embeddings.extend(resp_body["embeddings"])
            total_tokens += resp_body.get("inputTextTokenCount", 0)
    except Exception as exc:
        raise ProviderAPIError(str(exc), provider="bedrock") from exc

    return EmbeddingResponse(
        model=request.model,
        provider="bedrock",
        embeddings=all_embeddings,
        usage=TokenUsage(prompt_tokens=total_tokens, total_tokens=total_tokens),
        raw=None,
    )


async def _aembed_bedrock(request: EmbeddingRequest, api_key: str | None) -> EmbeddingResponse:
    import asyncio  # noqa: PLC0415
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _embed_bedrock, request, api_key)


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def _route(model: str) -> str:
    """Return the provider key for a given model string."""
    if model.startswith("gemini/"):
        return "gemini"
    if model.startswith("azure/"):
        return "azure"
    if model.startswith("cohere/"):
        return "cohere"
    if model.startswith("mistral/"):
        return "mistral"
    if model.startswith("ollama/"):
        return "ollama"
    if model.startswith("bedrock/"):
        return "bedrock"
    if model.startswith("anthropic/") or model.startswith("claude-"):
        return "anthropic"
    if model.startswith("groq/"):
        return "groq"
    # Default: OpenAI (text-embedding-*, or any other bare model)
    return "openai"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def embed(
    model: str,
    input: str | list[str],  # noqa: A002
    *,
    api_key: str | None = None,
    dimensions: int | None = None,
    **kwargs: Any,
) -> EmbeddingResponse:
    """
    Generate embeddings for one or more texts.

    Args:
        model:      Model name with optional provider prefix (e.g.
                    ``"text-embedding-3-small"``, ``"gemini/text-embedding-004"``).
        input:      A single string or a list of strings to embed.
        api_key:    Override the API key env var for this call.
        dimensions: Requested embedding dimensionality (OpenAI / Gemini / Azure).
        **kwargs:   Extra parameters forwarded to the provider.

    Returns:
        :class:`~llmgate.types.EmbeddingResponse` with
        ``embeddings: list[list[float]]`` — one vector per input text.

    Raises:
        :class:`~llmgate.exceptions.EmbeddingsNotSupported`:
            Provider does not support embeddings (Anthropic, Groq).
        :class:`~llmgate.exceptions.ProviderAPIError`: Provider returned an error.
    """
    request = EmbeddingRequest(model=model, input=input, dimensions=dimensions, extra_kwargs=kwargs)
    provider = _route(model)

    if provider == "anthropic":
        raise EmbeddingsNotSupported("anthropic")
    if provider == "groq":
        raise EmbeddingsNotSupported("groq")
    if provider == "openai":
        return _embed_openai(request, api_key)
    if provider == "gemini":
        return _embed_gemini(request, api_key)
    if provider == "azure":
        return _embed_openai(
            request, api_key,
            azure_endpoint=kwargs.pop("azure_endpoint", os.environ.get("AZURE_OPENAI_ENDPOINT")),
            api_version=kwargs.pop("api_version", None),
        )
    if provider == "cohere":
        return _embed_cohere(request, api_key)
    if provider == "mistral":
        return _embed_mistral(request, api_key)
    if provider == "ollama":
        return _embed_ollama(request, api_key)
    if provider == "bedrock":
        return _embed_bedrock(request, api_key)
    raise ProviderAPIError(f"Unknown provider '{provider}' for model '{model}'", provider=provider)


async def aembed(
    model: str,
    input: str | list[str],  # noqa: A002
    *,
    api_key: str | None = None,
    dimensions: int | None = None,
    **kwargs: Any,
) -> EmbeddingResponse:
    """Async version of :func:`embed`."""
    request = EmbeddingRequest(model=model, input=input, dimensions=dimensions, extra_kwargs=kwargs)
    provider = _route(model)

    if provider == "anthropic":
        raise EmbeddingsNotSupported("anthropic")
    if provider == "groq":
        raise EmbeddingsNotSupported("groq")
    if provider == "openai":
        return await _aembed_openai(request, api_key)
    if provider == "gemini":
        return await _aembed_gemini(request, api_key)
    if provider == "azure":
        return await _aembed_openai(
            request, api_key,
            azure_endpoint=kwargs.pop("azure_endpoint", os.environ.get("AZURE_OPENAI_ENDPOINT")),
            api_version=kwargs.pop("api_version", None),
        )
    if provider == "cohere":
        return await _aembed_cohere(request, api_key)
    if provider == "mistral":
        return await _aembed_mistral(request, api_key)
    if provider == "ollama":
        return await _aembed_ollama(request, api_key)
    if provider == "bedrock":
        return await _aembed_bedrock(request, api_key)
    raise ProviderAPIError(f"Unknown provider '{provider}' for model '{model}'", provider=provider)
