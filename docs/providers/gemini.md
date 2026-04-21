# Google Gemini

## Setup

```bash
export GEMINI_API_KEY="AIza..."
# or
export GOOGLE_API_KEY="AIza..."
```

No extra install — included with `pip install llmgate`.

---

## Model prefix

Models starting with `gemini-` are routed to Gemini automatically.

```python
completion("gemini-2.5-flash-lite", messages)
completion("gemini-2.0-flash", messages)
completion("gemini-2.5-pro", messages)
```

---

## Vision

Gemini does **not** accept image URLs natively. llmgate fetches URL images client-side using `httpx` and sends them as inline bytes — this is transparent to you:

```python
completion("gemini-2.0-flash", messages=[{
    "role": "user",
    "content": [
        {"type": "text", "text": "Describe this chart."},
        {"type": "image_url", "image_url": {"url": "https://..."}},  # fetched automatically
    ],
}])
```

---

## Structured outputs

Uses Gemini's native `response_schema` + `application/json` — very reliable:

```python
movie = parse("gemini-2.5-flash-lite", messages, response_format=Movie)
```

---

## Notes

- Uses the `google-genai` SDK (not the older `google-generativeai`).
- Conversation roles alternate `user` / `model`. llmgate converts `assistant` → `model` automatically.
- Tool results are sent as `function_response` parts in a user turn — handled automatically.
