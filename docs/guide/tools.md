# Tool Calling

llmgate provides a unified `ToolCall` interface for function calling across all providers that support it.

---

## Define tools

Tools are defined as `ToolDefinition` objects wrapping a `FunctionDefinition`:

```python
from llmgate.types import ToolDefinition, FunctionDefinition

tools = [
    ToolDefinition(function=FunctionDefinition(
        name="get_weather",
        description="Get the current weather for a city",
        parameters={
            "type": "object",
            "properties": {
                "city":  {"type": "string", "description": "City name"},
                "units": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    )),
]
```

You can also pass raw dicts in the OpenAI format — both work:

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": { ... },
    },
}]
```

---

## Making the call

```python
from llmgate import completion

resp = completion(
    "gpt-4o-mini",
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    tools=tools,
    tool_choice="auto",   # "auto" | "none" | {"type": "function", "function": {"name": "..."}}
)
```

---

## Handling tool calls

When the model wants to call a function, `resp.tool_calls` is populated:

```python
if resp.tool_calls:
    for tc in resp.tool_calls:
        print(tc.id)          # "call_abc123"
        print(tc.function)    # "get_weather"
        print(tc.arguments)   # {"city": "Tokyo", "units": "celsius"}

        # Execute your function
        result = get_weather(**tc.arguments)

        # Feed the result back
        messages.append({"role": "assistant", "tool_calls": [...]})
        messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": str(result),
        })
```

---

## Full multi-turn tool loop

```python
from llmgate import completion
from llmgate.types import Message, ToolCall

def get_weather(city: str, units: str = "celsius") -> str:
    return f"22°{'C' if units == 'celsius' else 'F'}, sunny"

messages = [{"role": "user", "content": "What's the weather in Paris and Berlin?"}]

while True:
    resp = completion("gpt-4o-mini", messages, tools=tools)

    if not resp.tool_calls:
        print(resp.text)
        break

    # Build assistant message with tool_calls
    messages.append({
        "role": "assistant",
        "content": resp.choices[0].message.content,
        "tool_calls": [
            {"id": tc.id, "type": "function",
             "function": {"name": tc.function, "arguments": tc.arguments}}
            for tc in resp.tool_calls
        ],
    })

    # Execute each tool and append results
    for tc in resp.tool_calls:
        result = get_weather(**tc.arguments)
        messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": result,
        })
```

---

## Provider support

| Provider | Tool calling |
|---|---|
| OpenAI | ✅ Native |
| Azure OpenAI | ✅ Native |
| Anthropic | ✅ Native |
| Gemini | ✅ Native |
| Groq | ✅ Native |
| Mistral | ✅ Native |
| Bedrock | ✅ Native (Converse API) |
| Ollama | ✅ Model-dependent |
| Cohere | ✅ Native |
