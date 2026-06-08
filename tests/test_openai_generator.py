import json

from model.openai_generator import OpenAIGenerator, OpenAIGeneratorConfig


class Usage:
    input_tokens = 1
    output_tokens = 2


class Response:
    def __init__(self, text):
        self.output_text = text
        self.usage = Usage()


class Responses:
    def __init__(self):
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        if kwargs.get("text"):
            return Response(json.dumps({"ok": True}))
        return Response("hello")


class Client:
    def __init__(self):
        self.responses = Responses()


def test_generate_string_and_list():
    generator = OpenAIGenerator(OpenAIGeneratorConfig(model="test"), client=Client())
    assert generator.generate("hi") == ["hello"]
    assert generator.generate(["a", "b"]) == ["hello", "hello"]


def test_generate_json():
    generator = OpenAIGenerator(OpenAIGeneratorConfig(model="test"), client=Client())
    result = generator.generate_json(
        "hi",
        {
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
            "additionalProperties": False,
        },
        "ok_schema",
    )
    assert result == {"ok": True}
