#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Rulează două modele locale noi, Gemma și Mistral, pe best prompts per task.

Scop:
    - Nu modifică experimentele vechi.
    - Produce fișiere JSON în același stil cu experimentele existente.
    - După rulare, poți folosi evaluate_models.py pentru agregarea metricilor.

Modele:
    - gemma3:12b
    - mistral:latest

Backend:
    - Ollama local API: http://localhost:11434/api/generate

Rulare:
    python src/prompting/run_new_models_best_prompts.py --task all
    python src/prompting/run_new_models_best_prompts.py --task intent
    python src/prompting/run_new_models_best_prompts.py --task incongruities --limit 20

Apoi:
    python src/prompting/evaluate_models.py --task all --save-latex
"""

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from jinja2 import Environment, FileSystemLoader, StrictUndefined


# ============================================================================
# CONFIG
# ============================================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATASET_PATH = BASE_DIR / "data" / "processed" / "master_dataset_refined_20260217_192302.json"

TASKS = ["intent", "final_status", "incongruities"]

MODELS = {
    "gemma3_4b": {
        "ollama_name": "gemma3:4b",
        "display_name": "Gemma 3 4B",
    },
    "mistral_7b": {
        "ollama_name": "mistral:7b-instruct",
        "display_name": "Mistral 7B Instruct",
    },
}

# Aici alegi best prompturile.
# Le-am pus ca default pe v4 RO, fiindcă în lucrare v4 = few-shot construit peste v3.
# Dacă din rezultatele tale best prompt e altul, schimbi aici o singură dată.
BEST_PROMPTS = {
    "intent": {
        "lang": "en",
        "version": "v4",
        "template": BASE_DIR / "prompts" / "intent_extraction" / "en_few_shot_v4.jinja",
    },
    "final_status": {
        "lang": "ro",
        "version": "v4",
        "template": BASE_DIR / "prompts" / "final_status" / "ro_final_status_v4.jinja",
    },
    "incongruities": {
        "lang": "en",
        "version": "v2",
        "template": BASE_DIR / "prompts" / "incongruities" / "en_incongruities_v2.jinja",
    },
}

TASK_OUTPUT_DIRS = {
    "intent": BASE_DIR / "outputs" / "intent",
    "final_status": BASE_DIR / "outputs_final_status",
    "incongruities": BASE_DIR / "outputs_incongruities",
}

LABEL_DEFINITION_PATHS = {
    "intent": BASE_DIR / "configs" / "intent_definitions.json",
    "final_status": BASE_DIR / "configs" / "final_status_definitions.json",
    "incongruities": BASE_DIR / "configs" / "incongruities_definitions.json",
}
FEW_SHOT_EXAMPLES_PATHS = {
    "intent": BASE_DIR / "configs" / "few_shot_examples_intent.json",
    "final_status": BASE_DIR / "configs" / "few_shot_examples_final_status.json",
    "incongruities": BASE_DIR / "configs" / "few_shot_examples_incongruities.json",
}

OLLAMA_URL = "http://localhost:11434/api/generate"


# ============================================================================
# DATA HELPERS
# ============================================================================

def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Nu există fișierul: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_dataset() -> List[Dict[str, Any]]:
    data = load_json(DATASET_PATH)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ["conversations", "data", "items", "dataset"]:
            if key in data and isinstance(data[key], list):
                return data[key]

    raise ValueError("Nu pot identifica lista de conversații din dataset.")


def load_label_definitions(task: str) -> Any:
    path = LABEL_DEFINITION_PATHS[task]
    if not path.exists():
        return None
    return load_json(path)

def load_few_shot_examples(task: str) -> List[Dict[str, Any]]:
    path = FEW_SHOT_EXAMPLES_PATHS.get(task)

    if path is None or not path.exists():
        return []

    data = load_json(path)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ["examples", "few_shot_examples", "items", "data"]:
            if key in data and isinstance(data[key], list):
                return data[key]

    return []


def conversation_to_text(conv: Dict[str, Any]) -> str:
    turns = conv.get("turns", [])
    lines = []

    for idx, turn in enumerate(turns, start=1):
        role = str(turn.get("role", "")).upper()
        text = str(turn.get("text", "")).strip()
        lines.append(f"{idx}. {role}: {text}")

    return "\n".join(lines)


def get_ground_truth(conv: Dict[str, Any], task: str) -> Dict[str, Any]:
    if task == "intent":
        return {
            "dataset_label": conv.get("intent"),
            "dataset_intent": conv.get("intent"),
        }

    if task == "final_status":
        return {
            "dataset_status": conv.get("final_status"),
            "dataset_label": conv.get("final_status"),
        }

    if task == "incongruities":
        incongruities = conv.get("incongruities", [])
        has_inc = bool(incongruities)

        if has_inc:
            first = incongruities[0]
            if isinstance(first, dict):
                inc_type = first.get("type") or first.get("label")
            else:
                inc_type = str(first)
        else:
            inc_type = "none"

        return {
            "dataset_has_incongruity": has_inc,
            "dataset_type": inc_type,
            "dataset_label": inc_type,
        }

    raise ValueError(f"Task necunoscut: {task}")


# ============================================================================
# PROMPT RENDERING
# ============================================================================

def load_template(template_path: Path):
    if not template_path.exists():
        raise FileNotFoundError(
            f"Nu găsesc template-ul: {template_path}\n"
            f"Verifică BEST_PROMPTS din script și numele fișierelor din folderul prompts."
        )

    env = Environment(
        loader=FileSystemLoader(str(template_path.parent)),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    return env.get_template(template_path.name)


def render_prompt(task: str, conv: Dict[str, Any]) -> str:
    cfg = BEST_PROMPTS[task]
    template = load_template(cfg["template"])

    context = {
        "conversation": conv.get("turns", []),
        "conversation_id": conv.get("conversation_id") or conv.get("id"),
        "turns": conv.get("turns", []),
        "intent": conv.get("intent"),
        "final_status": conv.get("final_status"),
        "incongruities": conv.get("incongruities", []),
        "label_definitions": load_label_definitions(task),
        "examples": load_few_shot_examples(task),
    }

    try:
        return template.render(**context)
    except Exception as e:
        raise RuntimeError(
            f"Eroare la randarea promptului pentru task={task}, "
            f"conversation_id={context['conversation_id']}: {e}"
        )


# ============================================================================
# MODEL CALL
# ============================================================================

def call_ollama(model_name: str, prompt: str, temperature: float = 0.0) -> Tuple[str, float]:
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": 256,
        },
    }

    start = time.perf_counter()
    response = requests.post(OLLAMA_URL, json=payload, timeout=180)
    latency_ms = (time.perf_counter() - start) * 1000

    response.raise_for_status()
    data = response.json()

    return data.get("response", ""), latency_ms


# ============================================================================
# PARSING
# ============================================================================

def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    Încearcă să extragă primul obiect JSON din răspuns.
    Merge și dacă modelul pune text înainte/după JSON.
    """
    text = text.strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None

    try:
        parsed = json.loads(match.group(0))
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        return None

    return None


def normalize_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    return str(value).strip().lower().replace(" ", "_")


def parse_prediction(task: str, raw_text: str) -> Dict[str, Any]:
    parsed = extract_json_object(raw_text)

    if parsed is None:
        return {
            "parse_failed": True,
            "raw_response": raw_text,
        }

    result = {
        "parse_failed": False,
        "raw_response": raw_text,
        "parsed_response": parsed,
    }

    if task == "intent":
        pred = (
            parsed.get("predicted_intent")
            or parsed.get("intent")
            or parsed.get("label")
            or parsed.get("prediction")
        )
        result["predicted_intent"] = normalize_str(pred)
        result["predicted_label"] = normalize_str(pred)

    elif task == "final_status":
        pred = (
            parsed.get("predicted_status")
            or parsed.get("predicted_final_status")
            or parsed.get("final_status")
            or parsed.get("label")
            or parsed.get("prediction")
        )
        result["predicted_status"] = normalize_str(pred)
        result["predicted_label"] = normalize_str(pred)

    elif task == "incongruities":
        has_inc = (
            parsed.get("predicted_has_incongruity")
            if "predicted_has_incongruity" in parsed
            else parsed.get("has_incongruity")
        )

        pred_type = (
            parsed.get("predicted_type")
            or parsed.get("predicted_incongruity_type")
            or parsed.get("type")
            or parsed.get("label")
            or parsed.get("prediction")
        )

        if isinstance(has_inc, str):
            has_inc_norm = has_inc.strip().lower() in {"true", "yes", "da", "1", "exista", "există"}
        elif has_inc is None:
            pred_type_norm = normalize_str(pred_type)
            has_inc_norm = pred_type_norm not in {None, "none", "nu", "false", "fara_neconcordanta"}
        else:
            has_inc_norm = bool(has_inc)

        result["predicted_has_incongruity"] = has_inc_norm
        result["predicted_type"] = normalize_str(pred_type) if has_inc_norm else "none"
        result["predicted_label"] = result["predicted_type"]

    return result


# ============================================================================
# RUNNER
# ============================================================================

def run_experiment(
    task: str,
    model_key: str,
    model_cfg: Dict[str, str],
    dataset: List[Dict[str, Any]],
    limit: Optional[int] = None,
) -> Path:
    prompt_cfg = BEST_PROMPTS[task]
    output_dir = TASK_OUTPUT_DIRS[task]
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    lang = prompt_cfg["lang"]
    version = prompt_cfg["version"]

    if task == "intent":
        prefix = "exp"
    elif task == "final_status":
        prefix = "exp_fst"
    else:
        prefix = "exp_inc"

    experiment_name = f"{prefix}_{model_key}__{lang}__{version}"

    predictions = []
    subset = dataset[:limit] if limit else dataset

    print(f"\n=== {task.upper()} | {model_key} | {lang} {version} | N={len(subset)} ===")

    for idx, conv in enumerate(subset, start=1):
        conv_id = conv.get("conversation_id") or conv.get("id") or f"conv_{idx:04d}"

        try:
            prompt = render_prompt(task, conv)
            raw_response, latency_ms = call_ollama(model_cfg["ollama_name"], prompt)
            pred = parse_prediction(task, raw_response)

        except Exception as e:
            latency_ms = None
            pred = {
                "parse_failed": True,
                "error": str(e),
                "raw_response": "",
            }

        row = {
            "conversation_id": conv_id,
            "model": model_key,
            "prompt_lang": lang,
            "prompt_version": version,
            "latency_ms": latency_ms,
            **get_ground_truth(conv, task),
            **pred,
        }

        predictions.append(row)

        status = "OK" if not row.get("parse_failed") else "PARSE_FAIL"
        print(f"[{idx:04d}/{len(subset)}] {conv_id} -> {status}")

    output = {
        "experiment_name": experiment_name,
        "model": model_key,
        "model_display": model_cfg["display_name"],
        "provider": "Ollama local",
        "ollama_model": model_cfg["ollama_name"],
        "language": lang,
        "prompt_version": version,
        "task": task,
        "timestamp": timestamp,
        "prompt_template": str(prompt_cfg["template"]),
        "n_predictions": len(predictions),
        "results": predictions,
        "predictions": predictions,
    }

    out_path = output_dir / f"{experiment_name}__{timestamp}.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nSaved: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        choices=["all", "intent", "final_status", "incongruities"],
        default="all",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODELS.keys()),
        help=f"Modele disponibile: {list(MODELS.keys())}",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Rulează doar primele N conversații, util pentru test rapid.",
    )

    args = parser.parse_args()

    dataset = load_dataset()
    tasks = TASKS if args.task == "all" else [args.task]

    print(f"Dataset: {DATASET_PATH}")
    print(f"Total conversații: {len(dataset)}")

    generated_files = []

    for task in tasks:
        for model_key in args.models:
            if model_key not in MODELS:
                raise ValueError(f"Model necunoscut: {model_key}. Disponibile: {list(MODELS.keys())}")

            out_path = run_experiment(
                task=task,
                model_key=model_key,
                model_cfg=MODELS[model_key],
                dataset=dataset,
                limit=args.limit,
            )
            generated_files.append(out_path)

    print("\nGata. Fișiere generate:")
    for p in generated_files:
        print(f" - {p}")

    print("\nUrmătorul pas:")
    print("python src/prompting/evaluate_models.py --task all --save-latex")


if __name__ == "__main__":
    main()