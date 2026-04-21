# Configuration

## API Keys

llmgate reads keys from environment variables. Set them once at the process level and every call picks them up automatically.

| Provider | Environment variable(s) |
|---|---|
| OpenAI | `OPENAI_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY` |
| Google Gemini | `GEMINI_API_KEY` or `GOOGLE_API_KEY` |
| Groq | `GROQ_API_KEY` |
| Mistral | `MISTRAL_API_KEY` |
| Cohere | `COHERE_API_KEY` |
| Azure OpenAI | `AZURE_OPENAI_API_KEY` · `AZURE_OPENAI_ENDPOINT` · `AZURE_OPENAI_API_VERSION` |
| AWS Bedrock | `AWS_ACCESS_KEY_ID` · `AWS_SECRET_ACCESS_KEY` · `AWS_DEFAULT_REGION` |
| Ollama | `OLLAMA_HOST` (default: `http://localhost:11434`) |

---

## Setting Keys

=== "Shell export"

    ```bash
    export OPENAI_API_KEY="sk-..."
    export GROQ_API_KEY="gsk_..."
    export ANTHROPIC_API_KEY="sk-ant-..."
    ```

=== ".env + python-dotenv"

    ```bash
    # .env  (never commit this file)
    OPENAI_API_KEY=sk-...
    ANTHROPIC_API_KEY=sk-ant-...
    GEMINI_API_KEY=AIza...
    ```

    ```python
    from dotenv import load_dotenv
    load_dotenv()   # call before any llmgate import
    from llmgate import completion
    ```

=== "Inline per call"

    Useful for multi-tenant apps where each user has their own key:

    ```python
    resp = completion("gpt-4o-mini", messages, api_key="sk-user-key")
    ```

---

## Provider Overrides

llmgate auto-detects the provider from the model string. Override it explicitly when needed:

```python
# Auto-detected from model prefix
resp = completion("groq/llama-3.1-8b-instant", messages)

# Explicit override — useful for custom deployments
resp = completion("llama-3.1-8b-instant", messages, provider="groq")
```

---

## Azure-Specific Configuration

Azure requires an endpoint and deployment name in addition to an API key:

```bash
export AZURE_OPENAI_API_KEY="..."
export AZURE_OPENAI_ENDPOINT="https://myinstance.openai.azure.com"
export AZURE_OPENAI_API_VERSION="2024-02-01"   # optional, this is the default
```

```python
resp = completion("azure/my-gpt4-deployment", messages)
```

Or pass them inline:

```python
from llmgate.providers.azure import AzureOpenAIProvider
from llmgate import LLMGate

gate = LLMGate(
    provider=AzureOpenAIProvider(
        api_key="...",
        azure_endpoint="https://myinstance.openai.azure.com",
        api_version="2024-06-01",
    )
)
resp = gate.completion("azure/my-gpt4-deployment", messages)
```

---

## Bedrock-Specific Configuration

Bedrock uses standard AWS credentials — the same ones `boto3` reads:

```bash
export AWS_ACCESS_KEY_ID="AKIA..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_DEFAULT_REGION="us-east-1"
```

Or use an IAM role / instance profile — llmgate delegates entirely to `boto3` for credential resolution.
