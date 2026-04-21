# Structured Outputs

Get typed, validated Pydantic instances directly from any LLM — no manual JSON parsing.

---

## The `parse()` shorthand

```python
from pydantic import BaseModel
from llmgate import parse

class Movie(BaseModel):
    title: str
    year: int
    director: str
    rating: float

movie = parse(
    "groq/llama-3.3-70b-versatile",
    [{"role": "user", "content": "Name a classic sci-fi film with details."}],
    response_format=Movie,
)

print(movie.title)    # "2001: A Space Odyssey"
print(movie.year)     # 1968
print(type(movie))    # <class 'Movie'>
```

---

## Via `completion()` with `response_format`

```python
from llmgate import completion

resp = completion(
    "gpt-4o-mini",
    messages,
    response_format=Movie,
)

movie: Movie = resp.parsed    # validated Pydantic instance
print(resp.text)              # also available as raw JSON string
```

---

## Async

```python
from llmgate import aparse

movie = await aparse("gemini-2.5-flash-lite", messages, response_format=Movie)
```

---

## Nested models

```python
from pydantic import BaseModel
from typing import list

class Actor(BaseModel):
    name: str
    role: str

class Film(BaseModel):
    title: str
    year: int
    cast: list[Actor]
    synopsis: str

film = parse("claude-3-5-sonnet-20241022", messages, response_format=Film)
for actor in film.cast:
    print(f"{actor.name} as {actor.role}")
```

---

## Provider strategies

Each provider implements structured output differently. llmgate picks the best approach automatically:

| Provider | Strategy |
|---|---|
| OpenAI / Azure | Native `json_schema` with `strict: true` — schema-constrained decoding |
| Gemini | `response_schema` + `application/json` MIME — native structured generation |
| Groq / Mistral / Ollama | `json_object` mode + Pydantic `model_validate_json()` |
| Anthropic / Bedrock / Cohere | Schema injected into system prompt → JSON extracted and validated |

---

!!! warning "Incompatibility"
    `response_format` and `stream=True` cannot be used together.
    Structured outputs require the complete response for validation.

!!! tip "Complex schemas"
    For Anthropic and Bedrock, keep schemas straightforward — very deeply nested or recursive schemas can confuse the prompt-injection approach. For complex cases, prefer OpenAI or Gemini which use native schema-constrained decoding.
