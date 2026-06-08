import hashlib
import json
import os
import random
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Optional
from urllib import request as urllib_request

from jsonschema import Draft202012Validator, ValidationError


class StructuredOutputError(RuntimeError):
    def __init__(self, message, raw_text=None):
        super().__init__(message)
        self.raw_text = raw_text


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
    api_key: Optional[str] = None


class OpenAIGenerator:
    def __init__(self, config: OpenAIGeneratorConfig, client=None):
        self.config = config
        self.call_records = []
        if client is not None:
            self.client = client
        else:
            try:
                from openai import OpenAI

                kwargs = {"timeout": config.timeout_seconds}
                if config.api_key:
                    kwargs["api_key"] = config.api_key
                if config.organization:
                    kwargs["organization"] = config.organization
                if config.project:
                    kwargs["project"] = config.project
                self.client = OpenAI(**kwargs)
            except ModuleNotFoundError:
                self.client = _UrllibOpenAIClient(
                    api_key=config.api_key or os.getenv("OPENAI_API_KEY"),
                    timeout=config.timeout_seconds,
                )
        if not (config.api_key or os.getenv("OPENAI_API_KEY")) and client is None:
            raise ValueError("OPENAI_API_KEY is required for OpenAIGenerator.")

    def generate(self, prompts: str | list[str], **kwargs) -> list[str]:
        single = isinstance(prompts, str)
        prompt_list = [prompts] if single else list(prompts)
        return [self._complete_text(prompt, **kwargs) for prompt in prompt_list]

    def generate_json(
        self,
        prompt: str,
        schema: dict[str, Any],
        schema_name: str,
        instructions: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
    ) -> dict[str, Any]:
        Draft202012Validator.check_schema(schema)
        validator = Draft202012Validator(schema)
        raw_text = None
        last_error = None
        for repair in [False, True]:
            call_prompt = prompt
            call_instructions = instructions or "Return only JSON matching the supplied schema."
            if repair:
                call_instructions += " Repair the previous invalid output and return valid JSON only."
                call_prompt += f"\n\nPrevious invalid output:\n{raw_text}\n\nValidation error:\n{last_error}"
            raw_text = self._complete_json_text(
                call_prompt,
                schema=schema,
                schema_name=schema_name,
                instructions=call_instructions,
                max_output_tokens=max_output_tokens,
            )
            try:
                parsed = json.loads(raw_text)
                validator.validate(parsed)
                return parsed
            except (json.JSONDecodeError, ValidationError) as exc:
                last_error = str(exc)
        raise StructuredOutputError(
            f"Could not produce valid structured output for {schema_name}: {last_error}",
            raw_text=raw_text,
        )

    def _complete_text(self, prompt, instructions=None, max_output_tokens=None, **kwargs):
        return self._retry(
            lambda: self._call_openai(
                prompt=prompt,
                instructions=instructions,
                max_output_tokens=max_output_tokens,
                **kwargs,
            )
        )

    def _complete_json_text(self, prompt, schema, schema_name, instructions, max_output_tokens=None):
        return self._retry(
            lambda: self._call_openai(
                prompt=prompt,
                instructions=instructions,
                schema=schema,
                schema_name=schema_name,
                max_output_tokens=max_output_tokens,
            )
        )

    def _retry(self, fn):
        last_exc = None
        for attempt in range(max(1, self.config.max_retries)):
            try:
                return fn()
            except Exception as exc:
                last_exc = exc
                if attempt >= self.config.max_retries - 1:
                    raise
                time.sleep(min(2**attempt + random.random(), 20))
        raise last_exc

    def _call_openai(
        self,
        prompt,
        instructions=None,
        schema=None,
        schema_name=None,
        max_output_tokens=None,
        **kwargs,
    ):
        if hasattr(self.client, "responses"):
            response = self._responses_create(
                prompt, instructions, schema, schema_name, max_output_tokens, **kwargs
            )
            text = getattr(response, "output_text", None) or self._extract_response_text(response)
        else:
            response = self._chat_completions_create(
                prompt, instructions, schema, max_output_tokens, **kwargs
            )
            text = response.choices[0].message.content
        self._record_usage(prompt, response, schema_name)
        return text or ""

    def _responses_create(
        self,
        prompt,
        instructions=None,
        schema=None,
        schema_name=None,
        max_output_tokens=None,
        **kwargs,
    ):
        request = {
            "model": self.config.model,
            "input": prompt,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_output_tokens": max_output_tokens or self.config.max_output_tokens,
        }
        if instructions:
            request["instructions"] = instructions
        if self.config.reasoning_effort:
            request["reasoning"] = {"effort": self.config.reasoning_effort}
        if schema:
            request["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "strict": True,
                    "schema": schema,
                }
            }
        try:
            return self.client.responses.create(**request)
        except TypeError:
            request.pop("temperature", None)
            return self.client.responses.create(**request)

    def _chat_completions_create(
        self,
        prompt,
        instructions=None,
        schema=None,
        max_output_tokens=None,
        **kwargs,
    ):
        messages = []
        if instructions:
            messages.append({"role": "system", "content": instructions})
        messages.append({"role": "user", "content": prompt})
        request = {
            "model": self.config.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": max_output_tokens or self.config.max_output_tokens,
        }
        if schema:
            request["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_output",
                    "strict": True,
                    "schema": schema,
                },
            }
        return self.client.chat.completions.create(**request)

    def _extract_response_text(self, response):
        chunks = []
        for item in getattr(response, "output", []) or []:
            for content in getattr(item, "content", []) or []:
                text = getattr(content, "text", None)
                if text:
                    chunks.append(text)
        return "\n".join(chunks)

    def _record_usage(self, prompt, response, schema_name):
        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "input_tokens", None) or getattr(usage, "prompt_tokens", None)
        output_tokens = getattr(usage, "output_tokens", None) or getattr(usage, "completion_tokens", None)
        self.call_records.append(
            {
                "call_id": str(uuid.uuid4()),
                "model": self.config.model,
                "prompt_hash": hashlib.sha256(str(prompt).encode("utf-8")).hexdigest(),
                "schema_name": schema_name,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }
        )


class _UrllibOpenAIClient:
    def __init__(self, api_key, timeout=60):
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI HTTP fallback.")
        self.api_key = api_key
        self.timeout = timeout
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.chat = SimpleNamespace(completions=_UrllibChatCompletions(self))


class _UrllibChatCompletions:
    def __init__(self, outer):
        self.outer = outer

    def create(self, **kwargs):
        url = f"{self.outer.base_url}/chat/completions"
        body = json.dumps(kwargs).encode("utf-8")
        req = urllib_request.Request(
            url,
            data=body,
            headers={
                "Authorization": f"Bearer {self.outer.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib_request.urlopen(req, timeout=self.outer.timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        choice = payload["choices"][0]
        usage = payload.get("usage") or {}
        return SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content=choice["message"].get("content", "")))
            ],
            usage=SimpleNamespace(
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
            ),
        )
