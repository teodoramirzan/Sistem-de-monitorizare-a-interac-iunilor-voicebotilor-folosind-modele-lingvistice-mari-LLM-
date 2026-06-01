from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .config import DATASET_PATH, MODEL_REGISTRY, RECOMMENDATIONS, TASKS
from .prompting import (
    conversation_to_text,
    get_prompt_path,
    load_few_shot_examples,
    normalize_turns,
    render_prompt,
)
from .providers import evaluate_with_provider


def _print_json(data: Any) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


def load_dataset(path: Path = DATASET_PATH) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(data, dict):
        return data.get("conversations", [])
    return data


def find_conversation(conversation_id: str, dataset_path: Path = DATASET_PATH) -> dict[str, Any]:
    for conversation in load_dataset(dataset_path):
        if conversation.get("conversation_id") == conversation_id:
            return conversation
    raise ValueError(f"Nu am găsit conversația {conversation_id!r} în {dataset_path}.")


def parse_text_conversation(text: str) -> dict[str, Any]:
    turns: list[dict[str, str]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if ":" not in line:
            raise ValueError(f"Linie invalidă în conversație: {raw_line!r}")
        role, content = line.split(":", 1)
        role = role.strip().lower()
        if role in {"assistant", "bot", "voicebot"}:
            role = "assistant"
        elif role in {"user", "client", "utilizator"}:
            role = "user"
        else:
            raise ValueError(f"Rol necunoscut în linia: {raw_line!r}")
        turns.append({"role": role, "text": content.strip()})
    return {"conversation_id": "inline", "turns": turns}


def load_conversation(args: argparse.Namespace) -> dict[str, Any]:
    if args.conversation_id:
        return find_conversation(args.conversation_id, Path(args.dataset))
    if args.conversation_file:
        data = json.loads(Path(args.conversation_file).read_text(encoding="utf-8-sig"))
        if isinstance(data, list):
            return {"conversation_id": Path(args.conversation_file).stem, "turns": data}
        return data
    if args.text:
        return parse_text_conversation(args.text)
    raise ValueError("Alege una dintre opțiunile: --conversation-id, --conversation-file sau --text.")


def resolve_task_options(task: str, args: argparse.Namespace) -> tuple[str, str, str]:
    rec = RECOMMENDATIONS[task]
    model = rec["model"] if args.model == "recommended" else args.model
    lang = rec["lang"] if args.lang == "recommended" else args.lang
    version = rec["prompt_version"] if args.prompt_version == "recommended" else args.prompt_version
    return model, lang, version


def evaluate_one(
    *,
    task: str,
    conversation: dict[str, Any],
    model_key: str,
    lang: str,
    prompt_version: str,
    provider: str,
    show_prompt: bool,
    show_raw: bool,
) -> dict[str, Any]:
    if provider == "local" and not show_prompt:
        examples = load_few_shot_examples(task, lang) if prompt_version == "v4" else []
        prompt_path = get_prompt_path(task, lang, prompt_version)
        prompt = ""
        prompt_meta = {
            "prompt_file": str(prompt_path.relative_to(prompt_path.parents[1])).replace("\\", "/"),
            "few_shot_examples_loaded": len(examples),
        }
    else:
        prompt, prompt_meta = render_prompt(
            task=task,
            conversation=conversation,
            lang=lang,
            version=prompt_version,
        )
    result, raw_response = evaluate_with_provider(
        task=task,
        conversation=conversation,
        model_key=model_key,
        prompt=prompt,
        provider=provider,
    )
    output = {
        "task": task,
        "model": model_key,
        "model_label": MODEL_REGISTRY[model_key].label,
        "provider": provider,
        "lang": lang,
        "prompt_version": prompt_version,
        "recommended_configuration": RECOMMENDATIONS[task],
        **prompt_meta,
        "result": result,
    }
    if show_prompt:
        output["prompt"] = prompt
    if show_raw and raw_response is not None:
        output["raw_response"] = raw_response
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pipeline demo pentru evaluarea conversațiilor voicebot pe cele 3 taskuri."
    )
    parser.add_argument("--task", choices=[*TASKS, "all"], default="all")
    parser.add_argument("--model", default="recommended", choices=["recommended", *MODEL_REGISTRY.keys()])
    parser.add_argument("--compare-models", help="Listă separată prin virgulă sau 'all'.")
    parser.add_argument("--lang", default="recommended", choices=["recommended", "ro", "en"])
    parser.add_argument("--prompt-version", default="recommended")
    parser.add_argument("--provider", default="local", choices=["local", "auto", "openai", "gemini", "ollama"])
    parser.add_argument("--conversation-id")
    parser.add_argument("--conversation-file")
    parser.add_argument("--text", help="Conversație inline, cu linii USER:/ASSISTANT:.")
    parser.add_argument("--dataset", default=str(DATASET_PATH))
    parser.add_argument("--show-prompt", action="store_true")
    parser.add_argument("--show-raw", action="store_true")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--recommendations", action="store_true")
    parser.add_argument("--show-conversation", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)

    if args.list_models:
        _print_json({key: spec.__dict__ for key, spec in MODEL_REGISTRY.items()})
        return 0
    if args.recommendations:
        _print_json(RECOMMENDATIONS)
        return 0

    conversation = load_conversation(args)
    tasks = list(TASKS) if args.task == "all" else [args.task]
    models = [args.model]
    if args.compare_models:
        models = list(MODEL_REGISTRY) if args.compare_models == "all" else [
            item.strip() for item in args.compare_models.split(",") if item.strip()
        ]

    evaluations = []
    for task in tasks:
        default_model, lang, prompt_version = resolve_task_options(task, args)
        for model_key in models:
            model = default_model if model_key == "recommended" else model_key
            if model not in MODEL_REGISTRY:
                raise ValueError(f"Model necunoscut: {model}")
            evaluations.append(
                evaluate_one(
                    task=task,
                    conversation=conversation,
                    model_key=model,
                    lang=lang,
                    prompt_version=prompt_version,
                    provider=args.provider,
                    show_prompt=args.show_prompt,
                    show_raw=args.show_raw,
                )
            )

    output = {
        "conversation_id": conversation.get("conversation_id", "inline"),
        "dataset_labels": {
            key: conversation.get(key)
            for key in ("intent", "final_status", "incongruities")
            if key in conversation
        },
        "evaluations": evaluations,
    }
    if args.show_conversation:
        output["conversation"] = conversation_to_text(normalize_turns(conversation))
    _print_json(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
