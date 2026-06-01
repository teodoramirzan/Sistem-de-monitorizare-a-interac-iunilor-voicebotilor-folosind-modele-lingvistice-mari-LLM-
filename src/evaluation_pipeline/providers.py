from __future__ import annotations

import json
import os
import re
from typing import Any

from .config import MODEL_REGISTRY
from .local_eval import evaluate_locally


OLLAMA_MODEL_ALIASES = {
    "aya_expanse_8b": ["aya-expanse:8b"],
    "rollama2_7b": ["rollama2:7b", "rollama2"],
    "mistral_7b": ["mistral:7b-instruct", "mistral:7b", "mistral"],
    "qwen2.5_3b": ["qwen2.5:3b", "qwen2.5:3b-instruct", "qwen2.5"],
}


class OllamaModelMissingError(RuntimeError):
    def __init__(self, model_key: str, candidates: list[str], installed: list[str]) -> None:
        self.model_key = model_key
        self.candidates = candidates
        self.installed = installed
        install = candidates[0] if candidates else model_key
        super().__init__(
            f"Modelul Ollama pentru {model_key} nu este instalat. "
            f"Am cautat: {', '.join(candidates) or model_key}. "
            f"Instalate: {', '.join(installed) or 'niciun model detectat'}. "
            f"Ruleaza: ollama pull {install}"
        )


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

        runtime_name = resolve_ollama_runtime_name(ollama, model_key, spec.runtime_name)
        response = ollama.chat(
            model=runtime_name,
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


def resolve_ollama_runtime_name(ollama_module, model_key: str, configured_name: str) -> str:
    env_key = f"OLLAMA_MODEL_{model_key.upper().replace('.', '_')}"
    candidates = []
    if os.getenv(env_key):
        candidates.append(os.environ[env_key])
    candidates.extend(OLLAMA_MODEL_ALIASES.get(model_key, []))
    if configured_name not in candidates:
        candidates.append(configured_name)

    installed = list_ollama_model_names(ollama_module)
    if not installed:
        raise OllamaModelMissingError(model_key, candidates, installed)
    for candidate in candidates:
        if candidate in installed:
            return candidate
    raise OllamaModelMissingError(model_key, candidates, installed)


def list_ollama_model_names(ollama_module) -> list[str]:
    try:
        response = ollama_module.list()
    except Exception:
        return []
    models = response.get("models", []) if isinstance(response, dict) else getattr(response, "models", [])
    names: list[str] = []
    for model in models:
        if isinstance(model, dict):
            name = model.get("name") or model.get("model")
        else:
            name = getattr(model, "name", None) or getattr(model, "model", None)
        if name:
            names.append(str(name))
    return names


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
