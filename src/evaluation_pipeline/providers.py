from __future__ import annotations

import json
import re
from typing import Any

from .config import MODEL_REGISTRY
from .local_eval import evaluate_locally


def parse_json_object(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def call_model(*, model_key: str, prompt: str, provider: str = "auto") -> tuple[dict[str, Any], str]:
    spec = MODEL_REGISTRY[model_key]
    chosen_provider = spec.provider if provider == "auto" else provider

    if chosen_provider == "openai":
        from openai import OpenAI

        client = OpenAI()
        response = client.responses.create(model=spec.runtime_name, input=prompt)
        raw = response.output_text
        return parse_json_object(raw), raw

    if chosen_provider == "gemini":
        from google import genai

        client = genai.Client()
        response = client.models.generate_content(model=spec.runtime_name, contents=prompt)
        raw = response.text or ""
        return parse_json_object(raw), raw

    if chosen_provider == "ollama":
        import ollama

        response = ollama.chat(
            model=spec.runtime_name,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0},
        )
        raw = response["message"]["content"]
        return parse_json_object(raw), raw

    if chosen_provider == "encoder":
        raise RuntimeError(
            f"{model_key} este baseline encoder, nu evaluator generativ. "
            "Folosește --provider local pentru demo sau un model generativ pentru prompturi."
        )

    raise ValueError(f"Provider necunoscut: {chosen_provider}")


def evaluate_with_provider(
    *,
    task: str,
    conversation: dict[str, Any] | list[dict[str, str]],
    model_key: str,
    prompt: str,
    provider: str,
) -> tuple[dict[str, Any], str | None]:
    if provider == "local":
        return evaluate_locally(task, conversation), None
    return call_model(model_key=model_key, prompt=prompt, provider=provider)
