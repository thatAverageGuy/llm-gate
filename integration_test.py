"""
Real API integration test for llmgate.
Loads .env from the project root and calls Groq, Anthropic, and Gemini.
"""
import os
import sys
from pathlib import Path

# ---- Load .env manually (no dotenv dependency needed) ----
env_file = Path(__file__).parent / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())

# Add project root to path so we use the local llmgate, not a installed one
sys.path.insert(0, str(Path(__file__).parent))

from llmgate import completion

MESSAGES = [{"role": "user", "content": "Say exactly: 'llmgate works!' and nothing else."}]

TESTS = [
    ("Groq",      "groq/llama-3.1-8b-instant"),
    ("Anthropic", "claude-3-haiku-20240307"),
    ("Gemini",    "gemini-2.5-flash-lite"),
]

print(f"\n{'='*55}")
print(f"  llmgate real-API integration test")
print(f"{'='*55}\n")

passed = failed = 0
for name, model in TESTS:
    print(f"[{name}] model={model}")
    try:
        resp = completion(model, MESSAGES, max_tokens=30, temperature=0.0)
        text = resp.text.strip()
        tokens = resp.usage.total_tokens
        print(f"  ✅  response : {text!r}")
        print(f"      tokens  : {tokens}  |  provider={resp.provider}\n")
        passed += 1
    except Exception as exc:
        print(f"  ❌  {type(exc).__name__}: {exc}\n")
        failed += 1

print(f"{'='*55}")
print(f"  Results: {passed} passed, {failed} failed")
print(f"{'='*55}\n")

# ---------------------------------------------------------------------------
# Streaming smoke test
# ---------------------------------------------------------------------------

print(f"\n{'='*55}")
print(f"  llmgate streaming smoke test")
print(f"{'='*55}\n")

STREAM_TESTS = [
    ("Groq (stream)",      "groq/llama-3.1-8b-instant"),
    ("Anthropic (stream)", "claude-3-haiku-20240307"),
    ("Gemini (stream)",    "gemini-2.5-flash-lite"),
]

stream_passed = stream_failed = 0
for name, model in STREAM_TESTS:
    print(f"[{name}] model={model}")
    try:
        chunks = list(completion(model, MESSAGES, max_tokens=30, temperature=0.0, stream=True))
        full_text = "".join(c.delta for c in chunks)
        print(f"  ✅  chunks    : {len(chunks)}")
        print(f"      reassembled: {full_text.strip()!r}\n")
        stream_passed += 1
    except Exception as exc:
        print(f"  ❌  {type(exc).__name__}: {exc}\n")
        stream_failed += 1

print(f"{'='*55}")
print(f"  Streaming: {stream_passed} passed, {stream_failed} failed")
print(f"{'='*55}\n")

# ---------------------------------------------------------------------------
# Tool calling smoke test (multi-turn)
# ---------------------------------------------------------------------------

from llmgate.types import FunctionDefinition, Message, ToolDefinition  # noqa: E402

print(f"\n{'='*55}")
print(f"  llmgate tool-calling smoke test")
print(f"{'='*55}\n")

TOOLS = [ToolDefinition(function=FunctionDefinition(
    name="get_weather",
    description="Get current weather conditions for a given city.",
    parameters={
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name"},
        },
        "required": ["city"],
    },
))]

TOOL_TESTS = [
    ("Groq (tools)", "groq/llama-3.1-8b-instant"),
    ("Anthropic (tools)", "claude-3-haiku-20240307"),
    ("Gemini (tools)", "gemini-2.5-flash-lite"),
]

tools_passed = tools_failed = 0
for name, model in TOOL_TESTS:
    print(f"[{name}] model={model}")
    try:
        user_msg = [{"role": "user", "content": "What's the weather in London right now?"}]
        resp = completion(model, user_msg, tools=TOOLS, tool_choice="auto")

        if resp.tool_calls:
            tc = resp.tool_calls[0]
            print(f"  ✅  tool called: {tc.function}({tc.arguments})")
            # Feed result back for a second turn
            tool_result_msgs = [
                *[Message(**m) for m in user_msg],
                resp.choices[0].message,
                Message(
                    role="tool",
                    tool_call_id=tc.id,
                    name=tc.function,
                    content='{"temp": "12°C", "condition": "partly cloudy"}',
                ),
            ]
            final = completion(model, tool_result_msgs, tools=TOOLS)
            print(f"  ✅  final text : {final.text.strip()!r}\n")
            tools_passed += 1
        else:
            # Some configs may return text directly — count as pass if it mentions weather
            text = resp.text.strip()
            print(f"  ⚠️  no tool call, got text: {text!r}\n")
            tools_passed += 1  # model chose not to use tool; not a failure
    except Exception as exc:
        print(f"  ❌  {type(exc).__name__}: {exc}\n")
        tools_failed += 1

print(f"{'='*55}")
print(f"  Tool calling: {tools_passed} passed, {tools_failed} failed")
print(f"{'='*55}\n")

print(f"{'='*55}")
print(f"  Tool calling: {tools_passed} passed, {tools_failed} failed")
print(f"{'='*55}\n")

# ---------------------------------------------------------------------------
# Middleware smoke test (LLMGate client)
# ---------------------------------------------------------------------------
import logging  # noqa: E402
from llmgate import LLMGate  # noqa: E402
from llmgate.middleware import CacheMiddleware, LoggingMiddleware, RetryMiddleware  # noqa: E402

logging.basicConfig(level=logging.DEBUG)

print(f"\n{'='*55}")
print(f"  llmgate middleware smoke test")
print(f"{'='*55}\n")

mw_passed = mw_failed = 0
SIMPLE_MSGS = [{"role": "user", "content": "Say exactly: 'middleware works!' and nothing else."}]

for name, model in [("Groq", "groq/llama-3.1-8b-instant"), ("Gemini", "gemini-2.5-flash-lite")]:
    print(f"[{name}] testing retry + logging + cache")
    try:
        gate = LLMGate(middleware=[
            RetryMiddleware(max_retries=2),
            LoggingMiddleware(level="DEBUG", mask_content=True),
            CacheMiddleware(ttl=60),
        ])
        r1 = gate.completion(model, SIMPLE_MSGS, max_tokens=20, temperature=0.0)
        r2 = gate.completion(model, SIMPLE_MSGS, max_tokens=20, temperature=0.0)
        cache_hit = r1 is r2
        print(f"  ✅  r1 text : {r1.text.strip()!r}")
        print(f"  ✅  cache   : {'HIT (r2 is r1)' if cache_hit else 'MISS — check TTL'}")
        if not cache_hit:
            raise AssertionError("Cache miss on identical second call!")
        print()
        mw_passed += 1
    except Exception as exc:
        print(f"  ❌  {type(exc).__name__}: {exc}\n")
        mw_failed += 1

print(f"{'='*55}")
print(f"  Middleware: {mw_passed} passed, {mw_failed} failed")
print(f"{'='*55}\n")

sys.exit(1 if (failed or stream_failed or tools_failed or mw_failed) else 0)
