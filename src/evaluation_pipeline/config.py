from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
DATASET_PATH = ROOT_DIR / "data" / "master_dataset_refined_180.json"
CONFIGS_DIR = ROOT_DIR / "configs"
PROMPTS_DIR = ROOT_DIR / "prompts"

TASKS = ("intent", "final_status", "incongruities")


@dataclass(frozen=True)
class ModelSpec:
    label: str
    provider: str
    runtime_name: str
    notes: str


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "openai_o3": ModelSpec("OpenAI o3", "openai", "o3", "model API studiat"),
    "gemini_2.5_flash": ModelSpec(
        "Gemini 2.5 Flash", "gemini", "gemini-2.5-flash", "model API studiat"
    ),
    "aya_expanse_8b": ModelSpec(
        "Aya Expanse 8B", "ollama", "aya-expanse:8b", "model local studiat"
    ),
    "rollama2_7b": ModelSpec(
        "RoLLaMA 2 7B", "ollama", "rollama2:7b", "model local studiat"
    ),
    "roberta_encoder": ModelSpec(
        "XLM-RoBERTa encoder",
        "encoder",
        "xlm-roberta-base",
        "baseline encoder; nu este model generativ pentru prompturi libere",
    ),
    "robert_encoder": ModelSpec(
        "RoBERT encoder",
        "encoder",
        "readerbench/RoBERT-base",
        "baseline encoder; alias păstrat din notebook-uri",
    ),
    "mistral_7b": ModelSpec(
        "Mistral 7B", "ollama", "mistral:7b-instruct", "model local adăugat pentru comparație"
    ),
    "qwen2.5_3b": ModelSpec(
        "Qwen2.5 3B", "ollama", "qwen2.5:3b", "model local adăugat pentru comparație"
    ),
}


RECOMMENDATIONS = {
    "intent": {
        "model": "openai_o3",
        "lang": "en",
        "prompt_version": "v4",
        "metric": "accuracy/f1 = 98.3%",
        "source": "evaluation_report_intent.txt și results/latex_tables",
        "reason": "OpenAI o3 EN v4 este la egalitate cu EN v3, dar v4 include few-shot.",
    },
    "final_status": {
        "model": "openai_o3",
        "lang": "ro",
        "prompt_version": "v4",
        "metric": "nu există JSON de rezultate salvat în repo",
        "source": "outputs_final_status/final_status_experiments.ipynb",
        "reason": "Notebook-ul folosește OpenAI o3 ca model implicit; v4 este varianta few-shot adăugată aici pentru demo.",
    },
    "incongruities": {
        "model": "gemini_2.5_flash",
        "lang": "ro",
        "prompt_version": "v4",
        "metric": "binary_f1 = 0.8219, type_macro_f1 = 0.8531",
        "source": "outputs_incongruities/exp_inc_gemini_2.5_flash__ro__v4.json",
        "reason": "Gemini 2.5 Flash RO v4 are cel mai bun F1 binar dintre rezultatele salvate.",
    },
}


PROMPT_FILES = {
    "intent": {
        ("ro", "v1"): "intent_extraction/ro_zero_shot_v1.jinja",
        ("ro", "v2"): "intent_extraction/ro_zero_shot_v2.jinja",
        ("ro", "v3"): "intent_extraction/ro_zero_shot_v3.jinja",
        ("ro", "v4"): "intent_extraction/ro_few_shot_v4.jinja",
        ("en", "v1"): "intent_extraction/en_zero_shot_v1.jinja",
        ("en", "v2"): "intent_extraction/en_zero_shot_v2.jinja",
        ("en", "v3"): "intent_extraction/en_zero_shot_v3.jinja",
        ("en", "v4"): "intent_extraction/en_few_shot_v4.jinja",
    },
    "final_status": {
        ("ro", "v1"): "final_status/fs_ro_zero_shot.jinja",
        ("ro", "zero_shot"): "final_status/fs_ro_zero_shot.jinja",
        ("ro", "v4"): "final_status/fs_ro_few_shot_v4.jinja",
        ("en", "v1"): "final_status/fs_en_zero_shot.jinja",
        ("en", "zero_shot"): "final_status/fs_en_zero_shot.jinja",
        ("en", "v4"): "final_status/fs_en_few_shot_v4.jinja",
    },
    "incongruities": {
        ("ro", "v1"): "incongruities/ro_incongruities_v1.jinja",
        ("ro", "v2"): "incongruities/ro_incongruities_v2.jinja",
        ("ro", "v3"): "incongruities/ro_incongruities_v3.jinja",
        ("ro", "v4"): "incongruities/ro_incongruities_v4.jinja",
        ("en", "v1"): "incongruities/en_incongruities_v1.jinja",
        ("en", "v2"): "incongruities/en_incongruities_v2.jinja",
        ("en", "v3"): "incongruities/en_incongruities_v3.jinja",
        ("en", "v4"): "incongruities/en_incongruities_v4.jinja",
    },
}


FEW_SHOT_FILES = {
    ("intent", "ro"): "few_shot_examples_intent.json",
    ("intent", "en"): "few_shot_examples_intent.json",
    ("final_status", "ro"): "few_shot_examples_final_status.json",
    ("final_status", "en"): "few_shot_examples_final_status_en.json",
    ("incongruities", "ro"): "few_shot_examples_incongruities.json",
    ("incongruities", "en"): "few_shot_examples_incongruities.json",
}
