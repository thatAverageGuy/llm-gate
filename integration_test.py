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

sys.exit(1 if (failed or stream_failed) else 0)
