# 03 — OpenAI API Adapter Spec

## 1. Goal

DRAG code currently assumes `generator.generate(prompt_or_prompts)` returns a list of strings. Implement an adapter with the same interface but backed by OpenAI API.

## 2. File

```text
model/openai_generator.py
```

## 3. Class interface

```python
from dataclasses import dataclass
from typing import Any, Optional

@dataclass
class OpenAIGeneratorConfig:
    model: str
    temperature: float = 0.0
    max_output_tokens: int = 512
    timeout_seconds: int = 60
    max_retries: int = 3
    reasoning_effort: Optional[str] = None
    organization: Optional[str] = None
    project: Optional[str] = None

class OpenAIGenerator:
    def __init__(self, config: OpenAIGeneratorConfig): ...

    def generate(self, prompts: str | list[str], **kwargs) -> list[str]: ...

    def generate_json(
        self,
        prompt: str,
        schema: dict[str, Any],
        schema_name: str,
        instructions: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
    ) -> dict[str, Any]: ...
```

## 4. Environment

Use:

```bash
export OPENAI_API_KEY="..."
```

Do not store API key in config file or logs.

## 5. Recommended Responses API call

Pseudo:

```python
from openai import OpenAI
client = OpenAI(timeout=config.timeout_seconds)

response = client.responses.create(
    model=config.model,
    instructions=instructions,
    input=prompt,
    temperature=config.temperature,
    max_output_tokens=config.max_output_tokens,
)
return response.output_text
```

For structured JSON:

```python
response = client.responses.create(
    model=config.model,
    instructions=instructions,
    input=prompt,
    text={
        "format": {
            "type": "json_schema",
            "name": schema_name,
            "strict": True,
            "schema": schema,
        }
    },
    max_output_tokens=max_output_tokens or config.max_output_tokens,
)
parsed = json.loads(response.output_text)
```

If selected OpenAI model/API version does not support `text.format`, fallback to `response_format` or plain JSON mode only behind a config flag.

## 6. Retry logic

Implement retry for:

- rate limits
- API connection errors
- timeouts
- transient 5xx

Pseudo:

```python
for attempt in range(max_retries):
    try:
        return call()
    except Exception as e:
        if attempt == max_retries - 1:
            raise
        sleep(backoff_seconds(attempt))
```

Backoff:

```python
sleep = min(2 ** attempt + random.random(), 20)
```

## 7. Usage tracking

Store per call:

```python
call_record = {
    "call_id": uuid,
    "model": model,
    "prompt_hash": sha256(prompt),
    "schema_name": schema_name or None,
    "created_at": timestamp,
    "input_tokens": response.usage.input_tokens if available else None,
    "output_tokens": response.usage.output_tokens if available else None,
}
```

Do not store full prompt inside API adapter by default. Pipeline may store prompt if `save_prompts=true`.

## 8. Safety against malformed JSON

`generate_json` must:

1. Request strict structured output.
2. Parse JSON.
3. Validate with Pydantic model or `jsonschema`.
4. On validation failure, retry once with a repair instruction.
5. If still fails, raise `StructuredOutputError` with raw text.

## 9. Test cases

Create `tests/test_openai_generator.py` with mocked client.

Tests:

- `generate("hi")` returns `[text]`.
- `generate(["a", "b"])` returns 2 outputs in order.
- `generate_json` returns dict.
- retry is called on transient exception.
- API key is never printed.
