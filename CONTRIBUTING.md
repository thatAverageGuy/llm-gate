# Contributing to llmgate

Thank you for your interest in contributing to llmgate! This document covers everything you need to know to contribute effectively.

---

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Project Philosophy](#project-philosophy)
3. [Getting Started](#getting-started)
4. [Project Structure](#project-structure)
5. [Making Changes](#making-changes)
6. [Adding a Provider](#adding-a-provider)
7. [Writing Tests](#writing-tests)
8. [Coding Standards](#coding-standards)
9. [Pull Request Process](#pull-request-process)
10. [Reporting Issues](#reporting-issues)

---

## Code of Conduct

Be kind, collaborative, and constructive. We welcome contributors of all backgrounds and experience levels.

---

## Project Philosophy

llmgate is deliberately **minimal**. Before adding a feature, ask:

> *Does this belong in a calling library, or in a higher-level orchestration layer?*

**In scope (v0.x):**
- LLM completion calls (sync + async)
- New provider integrations
- Streaming support
- Tool / function calling

**Out of scope (won't accept PRs without discussion):**
- Gateway/proxy server
- Admin UI or dashboards
- Database or persistence layers
- Framework integrations beyond thin helpers

If you're unsure, open an issue first.

---

## Getting Started

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (strongly recommended)
- Git

### Fork and clone

```bash
git clone https://github.com/<your-username>/llm-gate.git
cd llm-gate
```

### Set up the environment

```bash
uv sync   # installs all deps including dev deps
```

### Verify everything works

```bash
uv run pytest        # should show 29 passed
uv run python -c "import llmgate; print(llmgate.__version__)"
```

---

## Project Structure

```
llm-gate/
├── llmgate/
│   ├── __init__.py         # Public API surface + __version__
│   ├── types.py            # Pydantic v2 models (Message, CompletionRequest, etc.)
│   ├── exceptions.py       # Error hierarchy
│   ├── base.py             # BaseProvider ABC
│   ├── completion.py       # completion() / acompletion() + provider registry
│   └── providers/
│       ├── __init__.py
│       ├── openai.py
│       ├── gemini.py
│       ├── anthropic.py
│       └── groq.py
├── tests/
│   ├── test_types.py
│   ├── test_completion.py
│   └── test_providers.py
├── pyproject.toml
├── README.md
├── USER_GUIDE.md
└── CONTRIBUTING.md
```

Key design rules:
- **`types.py`** — provider-neutral. Never import provider SDKs here.
- **`base.py`** — the abstract contract. Stable; changes need careful thought.
- **`completion.py`** — the registry. Keep it slim; no provider-specific logic here.
- **`providers/`** — all SDK interaction lives here, isolated per file.

---

## Making Changes

### Branching convention

```
feature/<short-description>    # new feature
fix/<short-description>        # bug fix
docs/<short-description>       # documentation only
provider/<name>                # new provider
```

### Workflow

```bash
git checkout -b feature/my-thing
# ... make changes ...
uv run pytest                  # all tests must pass
git add .
git commit -m "feat: describe your change"
git push origin feature/my-thing
# open a PR on GitHub
```

---

## Adding a Provider

This is the most common contribution. Here's the full checklist:

### 1. Create `llmgate/providers/<name>.py`

Follow the pattern of an existing provider (e.g. `groq.py` for OpenAI-compatible SDKs, `anthropic.py` for APIs with different message formats):

```python
from __future__ import annotations
import os
from typing import Any, ClassVar
from llmgate.base import BaseProvider
from llmgate.exceptions import AuthError, ProviderAPIError, RateLimitError
from llmgate.types import Choice, CompletionRequest, CompletionResponse, Message, TokenUsage

class MyProvider(BaseProvider):
    name: ClassVar[str] = "myprovider"
    supported_model_prefixes: ClassVar[tuple[str, ...]] = ("myprovider/",)

    def __init__(self, api_key: str | None = None, **kwargs: Any) -> None:
        ...

    def _build_params(self, request: CompletionRequest) -> dict[str, Any]:
        ...

    def _map_response(self, raw: Any, model: str) -> CompletionResponse:
        ...

    def _handle_error(self, exc: Exception) -> None:
        ...

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        ...

    async def acomplete(self, request: CompletionRequest) -> CompletionResponse:
        ...
```

#### Rules for providers:
- Always support both `complete()` and `acomplete()`
- Always honour `max_tokens`, `temperature`, `top_p`, `extra_kwargs`
- Map to `CompletionResponse` exactly — no leakage of SDK types into the response
- Re-raise errors as `AuthError`, `RateLimitError`, or `ProviderAPIError`
- Store the raw SDK response in `CompletionResponse.raw`

### 2. Register in `llmgate/providers/__init__.py`

```python
from llmgate.providers.myprovider import MyProvider
__all__ = [..., "MyProvider"]
```

### 3. Register in `llmgate/completion.py`

```python
from llmgate.providers.myprovider import MyProvider

_PROVIDER_CLASSES: list[type[BaseProvider]] = [
    OpenAIProvider,
    GeminiProvider,
    AnthropicProvider,
    GroqProvider,
    MyProvider,   # add here
]
```

### 4. Add the SDK to `pyproject.toml`

```toml
dependencies = [
  ...
  "myprovider-sdk>=1.0",
]
```

### 5. Write tests in `tests/test_providers.py`

See [Writing Tests](#writing-tests) below.

### 6. Update README.md

Add a row to the Supported Providers table.

---

## Writing Tests

**All tests must be mocked — no real API calls.** This keeps CI fast and keyless.

### Test structure

```python
class TestMyProvider:
    def _make_raw(self):
        """Return a fake SDK response object."""
        from types import SimpleNamespace
        return SimpleNamespace(
            id="resp-123",
            choices=[SimpleNamespace(
                index=0,
                message=SimpleNamespace(role="assistant", content="Hello"),
                finish_reason="stop",
            )],
            usage=SimpleNamespace(prompt_tokens=5, completion_tokens=2, total_tokens=7),
        )

    def test_complete_maps_response(self):
        with patch("llmgate.providers.myprovider.MyProvider.__init__", return_value=None):
            from llmgate.providers.myprovider import MyProvider
            provider = MyProvider.__new__(MyProvider)
            provider._client = MagicMock()
            provider._client.chat.create.return_value = self._make_raw()

            req = CompletionRequest(model="myprovider/v1", messages=[Message(role="user", content="hi")])
            resp = provider.complete(req)
            assert resp.text == "Hello"
            assert resp.provider == "myprovider"

    @pytest.mark.asyncio
    async def test_acomplete_maps_response(self):
        ...
```

### Running tests

```bash
uv run pytest tests/ -v          # all tests
uv run pytest tests/test_providers.py -v -k "MyProvider"   # specific
```

### Coverage expectations

Every new provider must have tests for:
- [ ] `complete()` maps response correctly
- [ ] `acomplete()` maps response correctly
- [ ] Request parameters (`max_tokens`, `temperature`, `extra_kwargs`) forwarded
- [ ] Any special message handling (e.g. system prompt extraction)
- [ ] Any prefix stripping (e.g. `myprovider/model-name` → `model-name`)

---

## Coding Standards

### Style

- Follow **PEP 8**. We use 4-space indentation.
- Max line length: 100 characters.
- All public functions and classes must have docstrings.
- Use `from __future__ import annotations` in every file.

### Types

- All function signatures must be **fully type-annotated**.
- Use Pydantic v2 models for any new public-facing data structures.
- Avoid `Any` except in provider-specific SDK interaction code.

### Imports

- Standard library first, then third-party, then llmgate-internal.
- Provider SDK imports go **inside `__init__`**, not at module level — this avoids `ImportError` for users who haven't installed a specific provider SDK.

```python
# ✅ Correct
def __init__(self, api_key=None, **kwargs):
    try:
        import openai
    except ImportError as e:
        raise ImportError("openai package required: pip install openai") from e
    ...

# ❌ Wrong
import openai  # top-level import
```

### Commit messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add Mistral provider
fix: correct _strip_prefix for groq/ routing
docs: expand USER_GUIDE with FastAPI example
test: add async provider tests
chore: update google-generativeai to google-genai
```

---

## Pull Request Process

1. **Open an issue first** for non-trivial changes, so we can align on approach.
2. **Branch from `main`** — don't PR from your fork's `main`.
3. **Keep PRs focused** — one feature or fix per PR.
4. **Fill in the PR template** — describe what you changed and why.
5. **All tests must pass** — `uv run pytest` with zero failures.
6. **Update docs** — README, USER_GUIDE if behaviour changes.
7. **Bump version** in `pyproject.toml` if adding a provider or breaking change.

PRs are reviewed within a few days. Feedback will be constructive; don't take it personally.

---

## Reporting Issues

Use [GitHub Issues](https://github.com/thatAverageGuy/llm-gate/issues).

For **bug reports**, include:
- llmgate version (`python -c "import llmgate; print(llmgate.__version__)"`)
- Python version
- Provider and model name
- Minimal reproducing example (with API key redacted)
- Full traceback

For **feature requests**, describe the use case, not just the feature.

**Security issues** — do not file a public issue. Email the maintainer directly.
