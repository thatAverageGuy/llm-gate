# Chatbot

A simple stateful chatbot that maintains conversation history across turns.

```python
from llmgate import completion
from llmgate.types import Message


class Chatbot:
    def __init__(self, model: str = "gpt-4o-mini", system: str | None = None):
        self.model = model
        self.history: list[Message] = []
        if system:
            self.history.append(Message(role="system", content=system))

    def chat(self, user_input: str) -> str:
        self.history.append(Message(role="user", content=user_input))
        response = completion(self.model, self.history)
        reply = response.text
        self.history.append(Message(role="assistant", content=reply))
        return reply

    def reset(self):
        """Clear history but keep the system message."""
        self.history = [m for m in self.history if m.role == "system"]

    @property
    def turn_count(self) -> int:
        return sum(1 for m in self.history if m.role == "user")


# Usage
bot = Chatbot(
    model="groq/llama-3.3-70b-versatile",
    system="You are a helpful assistant. Be concise.",
)

print(bot.chat("My name is Alice."))
print(bot.chat("What's my name?"))   # "Your name is Alice."
print(f"Turns: {bot.turn_count}")

bot.reset()
print(bot.chat("Hello!"))  # fresh conversation, no memory of Alice
```

---

## Async chatbot

```python
import asyncio
from llmgate import acompletion
from llmgate.types import Message


class AsyncChatbot:
    def __init__(self, model: str = "gpt-4o-mini", system: str | None = None):
        self.model = model
        self.history: list[Message] = []
        if system:
            self.history.append(Message(role="system", content=system))

    async def chat(self, user_input: str) -> str:
        self.history.append(Message(role="user", content=user_input))
        response = await acompletion(self.model, self.history)
        reply = response.text
        self.history.append(Message(role="assistant", content=reply))
        return reply


async def main():
    bot = AsyncChatbot("gemini-2.5-flash-lite", system="You are a helpful assistant.")
    print(await bot.chat("Hi!"))
    print(await bot.chat("What did I just say?"))

asyncio.run(main())
```

---

## Vision chatbot

```python
import base64
from pathlib import Path
from llmgate import completion
from llmgate.types import Message, TextPart, ImagePart, ImageBytes


class VisionChatbot:
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.history: list[Message] = []

    def chat(self, text: str, image_path: str | None = None) -> str:
        if image_path:
            b64 = base64.b64encode(Path(image_path).read_bytes()).decode()
            suffix = Path(image_path).suffix.lstrip(".")
            mime = f"image/{suffix}"
            content = [
                TextPart(text=text),
                ImagePart(type="image_bytes", image_bytes=ImageBytes(data=b64, mime_type=mime)),
            ]
        else:
            content = text

        self.history.append(Message(role="user", content=content))
        resp = completion(self.model, self.history)
        self.history.append(Message(role="assistant", content=resp.text))
        return resp.text


bot = VisionChatbot("gpt-4o")
print(bot.chat("What's in this photo?", image_path="photo.jpg"))
print(bot.chat("Can you describe the colours in more detail?"))  # refers back to image
```
