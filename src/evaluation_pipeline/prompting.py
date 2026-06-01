from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import CONFIGS_DIR, PROMPT_FILES, PROMPTS_DIR


def normalize_turns(conversation: dict[str, Any] | list[dict[str, str]]) -> list[dict[str, str]]:
    if isinstance(conversation, dict):
        turns = conversation.get("turns", [])
    else:
        turns = conversation
    return [
        {"role": str(turn.get("role", "")).lower(), "text": str(turn.get("text", ""))}
        for turn in turns
    ]


def conversation_to_text(turns: list[dict[str, str]]) -> str:
    return "\n".join(f"{turn['role'].upper()}: {turn['text']}" for turn in turns)


def load_few_shot_examples(task: str, lang: str) -> list[dict[str, Any]]:
    from .config import FEW_SHOT_FILES

    file_name = FEW_SHOT_FILES.get((task, lang))
    if not file_name:
        return []
    path = CONFIGS_DIR / file_name
    if not path.exists():
        raise FileNotFoundError(
            f"Promptul v4 pentru task-ul {task!r} cere exemple few-shot, dar lipsește {path}."
        )
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("examples", [])
    raise ValueError(f"Format few-shot necunoscut în {path}.")


def get_prompt_path(task: str, lang: str, version: str) -> Path:
    try:
        rel_path = PROMPT_FILES[task][(lang, version)]
    except KeyError as exc:
        raise ValueError(
            f"Nu există prompt pentru task={task}, lang={lang}, version={version}."
        ) from exc
    path = PROMPTS_DIR / rel_path
    if not path.exists():
        raise FileNotFoundError(f"Promptul configurat nu există: {path}")
    return path


def render_prompt(
    *,
    task: str,
    conversation: dict[str, Any] | list[dict[str, str]],
    lang: str,
    version: str,
) -> tuple[str, dict[str, Any]]:
    try:
        from jinja2 import Environment, FileSystemLoader, StrictUndefined
    except ImportError as exc:
        raise RuntimeError("Instalează jinja2 pentru randarea prompturilor: pip install jinja2") from exc

    turns = normalize_turns(conversation)
    prompt_path = get_prompt_path(task, lang, version)
    examples = load_few_shot_examples(task, lang) if version == "v4" else []

    env = Environment(
        loader=FileSystemLoader(str(PROMPTS_DIR)),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template(str(prompt_path.relative_to(PROMPTS_DIR)).replace("\\", "/"))

    context: dict[str, Any] = {"conversation": turns}
    if task == "intent":
        context["examples"] = examples
    elif task in {"final_status", "incongruities"}:
        context["few_shot_examples"] = examples

    prompt = template.render(**context)
    metadata = {
        "prompt_file": str(prompt_path.relative_to(PROMPTS_DIR)).replace("\\", "/"),
        "few_shot_examples_loaded": len(examples),
    }
    return prompt, metadata
