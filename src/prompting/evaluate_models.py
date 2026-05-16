#!/usr/bin/env python3
# src/prompting/evaluate_thesis.py

"""
══════════════════════════════════════════════════════════════════════════════
  Evaluare completă LLM-uri — Capitolul 5 al dizertației
══════════════════════════════════════════════════════════════════════════════

Unified evaluation script for all 3 tasks:
    1. intent          — Extragerea intenției
    2. final_status    — Clasificarea statusului final
    3. incongruities   — Detecția neconcordanțelor

Produces:
    ─ Per-task tables (T1–T9) with full metrics
    ─ Cross-task comparative analysis (when --task all)
    ─ LaTeX tables ready for thesis inclusion
    ─ Comprehensive text report with researcher commentary
    ─ Highlights: best experiments, key findings, recommendations

Analyses included:
    T1   Overview experiments (all loaded, with parse fail rates)
    T2   All experiments ranked by Macro-F1
    T3   Best prompt per model (headline table)
    T4   Per-class Precision / Recall / F1 for best configs
    T5   Error analysis — confusion patterns for best configs
    T6   Prompt version evolution v1 → v4
    T7   Language impact: RO vs EN (accuracy, F1, latency delta)
    T8   Latency analysis (mean, median, p95, distribution type)
    T9   Output reliability (parse failures, confidence distribution)
    T-BIN   Binary detection metrics (incongruities only)

    C1   API vs Local models — grouped comparison
    C2   Model size vs performance (parameter efficiency)
    C3   Latency–accuracy trade-off
    C4   Cross-task ranking stability (Spearman ρ)  [--task all]
    C5   Statistical agreement — Cohen's κ interpretation

Usage:
    python evaluate_thesis.py --task intent
    python evaluate_thesis.py --task final_status
    python evaluate_thesis.py --task incongruities
    python evaluate_thesis.py --task all
    python evaluate_thesis.py --task intent --model openai_o3
    python evaluate_thesis.py --task all --save-latex
    python evaluate_thesis.py --task incongruities --debug
"""

import json
import argparse
import sys
import os
import numpy as np
from pathlib import Path
from collections import defaultdict, OrderedDict
from datetime import datetime

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    cohen_kappa_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

from tabulate import tabulate

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box
    HAS_RICH = True
    console = Console(width=140)
except ImportError:
    HAS_RICH = False
    console = None


# ════════════════════════════════════════════════════════════════════════════
# ██  CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

BASE_DIR = Path(
    r"C:\Users\Matebook 14s\Documents"
    r"\Sistem-de-monitorizare-a-interac-iunilor-voicebotilor-folosind-modele-lingvistice-mari-LLM-"
)

TASK_CONFIG = {
    "intent": {
        "display_name": "Extragerea intenției",
        "display_name_en": "Intent Extraction",
        "output_dir": BASE_DIR / "outputs" / "intent",
        "report_path": BASE_DIR / "evaluation_reports" / "eval_intent.txt",
        "latex_dir": BASE_DIR / "evaluation_reports" / "latex",
        "labels_path": BASE_DIR / "configs" / "intent_definitions.json",
        "labels_default": None,
        "file_pattern": "*.json",
        "pred_key_candidates": ["predictions", "results"],
        "pred_candidates": [
            "predicted_intent", "predicted_label", "prediction", "pred_label",
        ],
        "true_candidates": [
            "dataset_label", "dataset_intent", "ground_truth", "true_label", "label",
        ],
        "prefixes_to_remove": ["exp_"],
    },
    "final_status": {
        "display_name": "Clasificarea statusului final",
        "display_name_en": "Final Status Classification",
        "output_dir": BASE_DIR / "outputs_final_status",
        "report_path": BASE_DIR / "evaluation_reports" / "eval_final_status.txt",
        "latex_dir": BASE_DIR / "evaluation_reports" / "latex",
        "labels_path": BASE_DIR / "configs" / "final_status_definitions.json",
        "labels_default": [
            "rezolvata", "partial_rezolvata", "nerezolvata",
            "intrerupta", "redirectionata",
        ],
        "file_pattern": "*.json",
        "pred_key_candidates": ["results", "predictions"],
        "pred_candidates": [
            "predicted_status", "predicted_final_status",
            "predicted_label", "prediction", "pred_label",
        ],
        "true_candidates": [
            "dataset_status", "dataset_label", "ground_truth",
            "true_label", "label",
        ],
        "prefixes_to_remove": ["exp_fst_", "fst_", "exp_"],
    },
    "incongruities": {
        "display_name": "Detecția neconcordanțelor",
        "display_name_en": "Incongruity Detection",
        "output_dir": BASE_DIR / "outputs_incongruities",
        "report_path": BASE_DIR / "evaluation_reports" / "eval_incongruities.txt",
        "latex_dir": BASE_DIR / "evaluation_reports" / "latex",
        "labels_path": BASE_DIR / "configs" / "incongruities_definitions.json",
        "labels_default": [
            "incomplet", "irelevant", "contradictoriu",
            "nealiniat_context", "halucinatie",
        ],
        "file_pattern": "*.json",
        "pred_key_candidates": ["predictions", "results"],
        "pred_candidates": [
            "predicted_type", "predicted_incongruity_type",
            "predicted_label", "prediction", "pred_label",
        ],
        "true_candidates": [
            "dataset_label", "dataset_type", "dataset_incongruity_type",
            "ground_truth", "true_label", "label",
        ],
        "prefixes_to_remove": ["exp_inc_", "inc_", "exp_"],
        "binary": {
            "pred_candidates": [
                "predicted_has_incongruity", "has_incongruity",
                "predicted_binary", "prediction_binary",
            ],
            "true_candidates": [
                "dataset_has_incongruity", "ground_truth_has_incongruity",
                "true_has_incongruity", "label_binary",
            ],
        },
    },
}


# ── Model metadata ────────────────────────────────────────────────────────

MODEL_TYPE = {
    "openai_o3":        "API",
    "gemini_2.5_flash": "API",
    "aya_expanse_8b":   "Local",
    "rollama2_7b":      "Local",
    "roberta_encoder":  "Local",
    "robert_encoder":   "Local",
}

MODEL_DISPLAY = {
    "openai_o3":        "OpenAI o3",
    "gemini_2.5_flash": "Gemini 2.5 Flash",
    "aya_expanse_8b":   "Aya Expanse 8B",
    "rollama2_7b":      "RoLLaMA 2 7B",
    "roberta_encoder":  "XLM-RoBERTa",
    "robert_encoder":   "RoBERT-base",
}

MODEL_PARAMS_B = {
    "openai_o3":        200.0,      # estimated, undisclosed
    "gemini_2.5_flash": None,       # undisclosed
    "aya_expanse_8b":   8.0,
    "rollama2_7b":      7.0,
    "roberta_encoder":  0.56,
    "robert_encoder":   0.125,
}

MODEL_PARAMS_DISPLAY = {
    "openai_o3":        "~200B+",
    "gemini_2.5_flash": "undisclosed",
    "aya_expanse_8b":   "8B",
    "rollama2_7b":      "7B",
    "roberta_encoder":  "~560M",
    "robert_encoder":   "~125M",
}

# Inference environment metadata
MODEL_INFERENCE_INFO = {
    "openai_o3":        {"provider": "OpenAI API", "quantization": "N/A", "hardware": "cloud GPU cluster"},
    "gemini_2.5_flash": {"provider": "Google AI API", "quantization": "N/A", "hardware": "cloud TPU/GPU"},
    "aya_expanse_8b":   {"provider": "Ollama (local)", "quantization": "Q4_K_M", "hardware": "local GPU"},
    "rollama2_7b":      {"provider": "Ollama (local)", "quantization": "Q4_K_M", "hardware": "local GPU"},
    "roberta_encoder":  {"provider": "HuggingFace (local)", "quantization": "FP16/FP32", "hardware": "local GPU"},
    "robert_encoder":   {"provider": "HuggingFace (local)", "quantization": "FP16/FP32", "hardware": "local GPU"},
}

# Kappa interpretation (Landis & Koch, 1977)
KAPPA_INTERPRETATION = [
    (0.0,  "Poor"),
    (0.20, "Slight"),
    (0.40, "Fair"),
    (0.60, "Moderate"),
    (0.80, "Substantial"),
    (1.01, "Almost Perfect"),
]


def interpret_kappa(k):
    if k is None:
        return "—"
    for threshold, label in KAPPA_INTERPRETATION:
        if k < threshold:
            return label
    return "Almost Perfect"


# ════════════════════════════════════════════════════════════════════════════
# ██  OUTPUT SYSTEM
# ════════════════════════════════════════════════════════════════════════════

_report_lines = []
_latex_blocks = {}   # key → LaTeX string, collected for file export


def rprint(text="", style=None):
    """Print to console + accumulate for report file."""
    _report_lines.append(str(text) + "\n")
    if HAS_RICH and console:
        if style:
            console.print(text, style=style)
        else:
            console.print(text)
    else:
        print(text)


def rprint_table(rows, headers, title=None):
    """Print table (rich if available) + accumulate plain text."""
    plain = tabulate(rows, headers=headers, tablefmt="rounded_outline")
    _report_lines.append(f"\n{title}\n" if title else "")
    _report_lines.append(plain + "\n\n")
    if HAS_RICH and console:
        table = Table(
            title=title, box=box.ROUNDED, header_style="bold cyan",
            border_style="blue", show_lines=True,
        )
        for h in headers:
            table.add_column(str(h), no_wrap=False)
        for row in rows:
            table.add_row(*[str(c) for c in row])
        console.print(table)
    else:
        if title:
            print(f"\n{title}")
        print(plain)


def section(title):
    line = f"\n{'═' * 100}\n  {title}\n{'═' * 100}"
    _report_lines.append(line + "\n")
    if HAS_RICH and console:
        console.print()
        console.print(Panel(f"[bold white]{title}[/bold white]",
                            border_style="cyan", expand=True))
    else:
        print(line)


def commentary(text):
    """Research commentary — insight paragraph for the report."""
    wrapped = f"\n  📝 {text}\n"
    _report_lines.append(wrapped + "\n")
    if HAS_RICH and console:
        console.print(f"\n  [italic bright_yellow]{text}[/italic bright_yellow]\n")
    else:
        print(wrapped)


def save_report(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(_report_lines)
    rprint(f"\n✓ Report saved: {path}", style="bold green")


def save_latex_files(latex_dir):
    latex_dir = Path(latex_dir)
    latex_dir.mkdir(parents=True, exist_ok=True)
    for name, content in _latex_blocks.items():
        p = latex_dir / f"{name}.tex"
        with open(p, "w", encoding="utf-8") as f:
            f.write(content)
        rprint(f"  ✓ LaTeX: {p}", style="green")


# ════════════════════════════════════════════════════════════════════════════
# ██  FORMAT HELPERS
# ════════════════════════════════════════════════════════════════════════════

def pct(v):
    return f"{v * 100:.1f}%" if v is not None else "—"

def pct_latex(v):
    return f"{v * 100:.1f}\\%" if v is not None else "—"

def fmt(v, d=3):
    if v is None:
        return "—"
    try:
        if np.isnan(v):
            return "—"
    except TypeError:
        pass
    return f"{v:.{d}f}"

def ms(v):
    return f"{v:.0f} ms" if v is not None else "—"

def sec(v):
    return f"{v:.2f} s" if v is not None else "—"

def delta_pct(a, b):
    if a is None or b is None:
        return "—"
    d = (a - b) * 100
    return f"{'+' if d >= 0 else ''}{d:.1f}%"

def delta_ms(a, b):
    if a is None or b is None:
        return "—"
    d = a - b
    return f"{'+' if d >= 0 else ''}{d:.0f} ms"

def get_first_existing(d, candidates):
    for key in candidates:
        if key in d and d.get(key) is not None:
            return d.get(key)
    return None


# ════════════════════════════════════════════════════════════════════════════
# ██  NORMALIZATION
# ════════════════════════════════════════════════════════════════════════════

def normalize_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    if isinstance(value, float):
        if value == 1.0:
            return True
        if value == 0.0:
            return False
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "1", "yes", "da", "present", "exists", "existent"}:
            return True
        if v in {"false", "0", "no", "nu", "absent", "none", "null"}:
            return False
    return None


# Comprehensive label normalization covering all 3 tasks
LABEL_MAP = {
    # ── Incongruities ──
    "halucinație": "halucinatie", "halucinatie": "halucinatie",
    "hallucination": "halucinatie",
    "incomplet": "incomplet", "incomplete": "incomplet",
    "irelevant": "irelevant", "irrelevant": "irelevant",
    "contradictoriu": "contradictoriu", "contradiction": "contradictoriu",
    "contradictory": "contradictoriu",
    "nealiniat_context": "nealiniat_context",
    "context_misalignment": "nealiniat_context",
    "misaligned_context": "nealiniat_context",
    "none": "none", "no_incongruity": "none",
    "fara_neconcordanta": "none", "fără_neconcordanță": "none",
    "nu_exista": "none", "absent": "none",
    # ── Final status ──
    "rezolvată": "rezolvata", "rezolvata": "rezolvata", "resolved": "rezolvata",
    "parțial_rezolvată": "partial_rezolvata",
    "partial_rezolvata": "partial_rezolvata",
    "partially_resolved": "partial_rezolvata",
    "nerezolvată": "nerezolvata", "nerezolvata": "nerezolvata",
    "unresolved": "nerezolvata",
    "întreruptă": "intrerupta", "intrerupta": "intrerupta",
    "interrupted": "intrerupta",
    "redirecționată": "redirectionata", "redirectionata": "redirectionata",
    "redirected": "redirectionata",
    # ── Romanian intent aliases seen in local model outputs ──
    "blocare_card": "block_card", "deblocare_card": "unblock_card",
    "inchidere_cont": "close_account", "închidere_cont": "close_account",
    "deschidere_cont": "open_account", "sold_cont": "check_balance",
    "extras_cont": "get_account_statement",
    "programare_consultant": "schedule_advisor_meeting",
}


def normalize_label(value):
    if value is None:
        return None
    value = str(value).strip().lower().strip("'\"").replace(" ", "_")
    return LABEL_MAP.get(value, value)


def normalize_version(version):
    return str(version).lower().strip()


# ════════════════════════════════════════════════════════════════════════════
# ██  METADATA EXTRACTION & LOADING
# ════════════════════════════════════════════════════════════════════════════

def remove_prefixes(name, prefixes):
    for p in prefixes:
        if name.startswith(p):
            name = name[len(p):]
    return name


def extract_metadata(path, data, cfg):
    """Extract (model, lang, version, timestamp) with fallback chain."""
    meta = data.get("experiment", {})
    if meta:
        model = meta.get("model", data.get("model", "unknown"))
        lang = meta.get("lang", meta.get("language", data.get("language", "unknown")))
        version = meta.get("prompt_version", data.get("prompt_version", "unknown"))
        ts = meta.get("timestamp", data.get("timestamp", "00000000"))
    else:
        model = data.get("model", "unknown")
        lang = data.get("language", "unknown")
        version = data.get("prompt_version", "unknown")
        ts = data.get("timestamp", "00000000")

    # Fallback: parse experiment_name  (prefix_model__lang__version)
    exp_name = data.get("experiment_name", meta.get("name", path.stem))
    parts = exp_name.split("__")
    if len(parts) >= 3:
        m_name = remove_prefixes(parts[0], cfg.get("prefixes_to_remove", []))
        l_name = parts[1]
        v_name = parts[2].replace(".json", "")
        if model == "unknown":
            model = m_name
        if lang == "unknown":
            lang = l_name
        if version == "unknown":
            version = v_name

    return (
        str(model).lower().strip(),
        str(lang).lower().strip(),
        normalize_version(version),
        str(ts),
    )


def get_predictions_from_data(data, cfg):
    for key in cfg["pred_key_candidates"]:
        if key in data and isinstance(data[key], list):
            return data[key]
    return []


def load_labels(cfg):
    labels_path = cfg["labels_path"]
    if not labels_path.exists():
        if cfg.get("labels_default"):
            rprint(f"  ⚠ Labels file not found ({labels_path}). Using defaults.")
            return [normalize_label(x) for x in cfg["labels_default"]]
        raise FileNotFoundError(f"No labels file: {labels_path}")

    with open(labels_path, encoding="utf-8") as f:
        defs = json.load(f)

    if "labels" in defs:
        labels = [l["name"] for l in defs["labels"]]
    elif isinstance(defs, list):
        labels = defs
    else:
        labels = cfg.get("labels_default")

    if not labels:
        raise ValueError(f"Cannot extract labels from {labels_path}")
    return [normalize_label(x) for x in labels]


def load_all_experiments(cfg, filter_model=None, filter_lang=None, filter_version=None):
    """Load every experiment file, returning list of dicts."""
    output_dir = cfg["output_dir"]
    exp_files = sorted(output_dir.rglob(cfg["file_pattern"]))

    if not exp_files:
        raise FileNotFoundError(f"No JSON files found in {output_dir}")

    experiments = []
    for path in exp_files:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        model, lang, version, ts = extract_metadata(path, data, cfg)

        if filter_model and model != filter_model:
            continue
        if filter_lang and lang != filter_lang:
            continue
        if filter_version and version != filter_version:
            continue

        preds = get_predictions_from_data(data, cfg)
        if not preds:
            continue

        experiments.append({
            "model": model, "lang": lang, "version": version,
            "timestamp": ts, "path": path, "data": data,
            "predictions": preds,
        })

    if not experiments:
        raise ValueError("No experiments match the given filters.")

    rprint(f"\n  Loaded {len(experiments)} experiment files:", style="bold")
    for exp in sorted(experiments, key=lambda e: (e["model"], e["lang"], e["version"])):
        rprint(f"    {MODEL_DISPLAY.get(exp['model'], exp['model']):<25} "
               f"[{exp['lang'].upper()}] {exp['version']:<10} ← {exp['path'].name}")

    return experiments


# ════════════════════════════════════════════════════════════════════════════
# ██  PREDICTION NORMALIZATION
# ════════════════════════════════════════════════════════════════════════════

def normalize_prediction_row(row, exp, cfg):
    """Normalize a single prediction row into a canonical form."""
    row = dict(row)

    pred_label = get_first_existing(row, cfg["pred_candidates"])
    true_label = get_first_existing(row, cfg["true_candidates"])

    pred_norm = normalize_label(pred_label)
    true_norm = normalize_label(true_label)

    normalized = {
        **row,
        "model_name": exp["model"],
        "prompt_lang": exp["lang"],
        "prompt_version": exp["version"],
        "source_file": exp["path"].name,
        "predicted_label_norm": pred_norm,
        "true_label_norm": true_norm,
    }

    # Binary normalization (incongruities)
    if "binary" in cfg:
        pred_binary_raw = get_first_existing(row, cfg["binary"]["pred_candidates"])
        true_binary_raw = get_first_existing(row, cfg["binary"]["true_candidates"])
        pred_binary = normalize_bool(pred_binary_raw)
        true_binary = normalize_bool(true_binary_raw)
        # Derive from multiclass if needed
        if true_binary is None and true_norm is not None:
            true_binary = true_norm != "none"
        if pred_binary is None and pred_norm is not None:
            pred_binary = pred_norm != "none"
        normalized["predicted_binary_norm"] = pred_binary
        normalized["true_binary_norm"] = true_binary

    return normalized


def build_by_experiment(experiments, cfg):
    """Group normalized predictions by (model, lang, version)."""
    by_key = defaultdict(list)
    for exp in experiments:
        key = (exp["model"], exp["lang"], exp["version"])
        for row in exp["predictions"]:
            norm = normalize_prediction_row(row, exp, cfg)
            by_key[key].append(norm)

    total = sum(len(v) for v in by_key.values())
    rprint(f"\n  Total predictions across all experiments: {total}\n", style="bold")
    return by_key


# ════════════════════════════════════════════════════════════════════════════
# ██  METRICS COMPUTATION
# ════════════════════════════════════════════════════════════════════════════

def compute_latency(rows):
    lats = [r.get("latency_ms", 0) for r in rows if r.get("latency_ms", 0) > 0]
    if not lats:
        return {"mean": None, "median": None, "p95": None, "p99": None,
                "min": None, "max": None, "std": None, "n_samples": 0}
    return {
        "mean": float(np.mean(lats)),
        "median": float(np.median(lats)),
        "p95": float(np.percentile(lats, 95)),
        "p99": float(np.percentile(lats, 99)),
        "min": float(np.min(lats)),
        "max": float(np.max(lats)),
        "std": float(np.std(lats)),
        "n_samples": len(lats),
    }


def safe_kappa(y_true, y_pred, labels):
    """Cohen's Kappa that penalizes __invalid__ predictions."""
    try:
        kappa_labels = list(labels)
        if "__invalid__" in y_pred and "__invalid__" not in kappa_labels:
            kappa_labels.append("__invalid__")
        return cohen_kappa_score(y_true, y_pred, labels=kappa_labels)
    except Exception:
        return None


def get_multiclass_yt_yp(rows, labels, task):
    y_true, y_pred, valid_rows = [], [], []
    for r in rows:
        true_label = r.get("true_label_norm")
        pred_label = r.get("predicted_label_norm")
        if true_label is None:
            continue
        # For incongruities multiclass: only samples WITH an incongruity
        if task == "incongruities" and true_label not in labels:
            continue
        if task != "incongruities" and true_label not in labels:
            continue
        y_true.append(true_label)
        y_pred.append(pred_label if pred_label in labels else "__invalid__")
        valid_rows.append(r)
    return y_true, y_pred, valid_rows


def compute_multiclass_metrics(rows, labels, task):
    y_true, y_pred, valid_rows = get_multiclass_yt_yp(rows, labels, task)
    if not y_true:
        return None
    n_invalid = sum(1 for p in y_pred if p == "__invalid__")
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0),
        "kappa": safe_kappa(y_true, y_pred, labels),
        "n": len(y_true),
        "n_invalid": n_invalid,
        "invalid_rate": n_invalid / len(y_true) if y_true else 0,
        "y_true": y_true,
        "y_pred": y_pred,
        "valid_rows": valid_rows,
    }


def compute_binary_metrics(rows):
    y_true, y_pred = [], []
    for r in rows:
        t = r.get("true_binary_norm")
        p = r.get("predicted_binary_norm")
        if t is not None and p is not None:
            y_true.append(t)
            y_pred.append(p)
    if not y_true:
        return None
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average="binary", pos_label=True, zero_division=0),
        "kappa": cohen_kappa_score(y_true, y_pred),
        "n": len(y_true),
        "y_true": y_true,
        "y_pred": y_pred,
    }


# ════════════════════════════════════════════════════════════════════════════
# ██  RECORD COLLECTION
# ════════════════════════════════════════════════════════════════════════════

def collect_multiclass_records(by_key, labels, task):
    records = []
    for key in sorted(by_key):
        model, lang, version = key
        rows = by_key[key]
        metrics = compute_multiclass_metrics(rows, labels, task)
        if not metrics:
            continue
        lat = compute_latency(metrics["valid_rows"])
        records.append({
            "model": model,
            "model_display": MODEL_DISPLAY.get(model, model),
            "type": MODEL_TYPE.get(model, "?"),
            "params": MODEL_PARAMS_DISPLAY.get(model, "?"),
            "params_b": MODEL_PARAMS_B.get(model),
            "lang": lang,
            "version": version,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "kappa": metrics["kappa"],
            "n_invalid": metrics["n_invalid"],
            "invalid_rate": metrics["invalid_rate"],
            "latency": lat,
            "n": metrics["n"],
            "y_true": metrics["y_true"],
            "y_pred": metrics["y_pred"],
            "valid_rows": metrics["valid_rows"],
        })
    return records


def collect_binary_records(by_key):
    records = []
    for key in sorted(by_key):
        model, lang, version = key
        rows = by_key[key]
        metrics = compute_binary_metrics(rows)
        if not metrics:
            continue
        lat = compute_latency(rows)
        records.append({
            "model": model,
            "model_display": MODEL_DISPLAY.get(model, model),
            "type": MODEL_TYPE.get(model, "?"),
            "lang": lang,
            "version": version,
            "accuracy": metrics["accuracy"],
            "f1": metrics["f1"],
            "kappa": metrics["kappa"],
            "latency": lat,
            "n": metrics["n"],
            "y_true": metrics["y_true"],
            "y_pred": metrics["y_pred"],
        })
    return records


def choose_best_per_model(records, metric_name="macro_f1"):
    """Pick the best (lang, version) config per model, ranked by metric."""
    best_by = {}
    for rec in records:
        m = rec["model"]
        if m not in best_by:
            best_by[m] = rec
            continue
        curr = best_by[m]
        if (rec[metric_name] > curr[metric_name] or
            (rec[metric_name] == curr[metric_name] and
             rec["accuracy"] > curr["accuracy"])):
            best_by[m] = rec
    return sorted(best_by.values(), key=lambda r: r[metric_name], reverse=True)


# ════════════════════════════════════════════════════════════════════════════
# ██  TABLE PRINTERS — PER-TASK ANALYSIS
# ════════════════════════════════════════════════════════════════════════════

def print_t1_overview(by_key, cfg):
    section(f"T1 — Experiment Overview: {cfg['display_name']}")
    rows = []
    for key in sorted(by_key):
        model, lang, version = key
        preds = by_key[key]
        total = len(preds)
        fails = sum(1 for r in preds if r.get("parse_failed", False))
        rows.append([
            MODEL_DISPLAY.get(model, model), MODEL_TYPE.get(model, "?"),
            MODEL_PARAMS_DISPLAY.get(model, "?"), lang.upper(), version,
            total, fails, pct(fails / total) if total else "—",
        ])
    rprint_table(rows,
                 headers=["Model", "Type", "Params", "Lang", "Ver", "N", "Fails", "Fail%"],
                 title="T1 — All loaded experiments")


def print_t2_all_experiments(records, cfg):
    section(f"T2 — All Experiments Ranked: {cfg['display_name']}")
    rows = []
    for rec in records:
        rows.append([
            rec["model_display"], rec["type"], rec["lang"].upper(), rec["version"],
            pct(rec["accuracy"]), fmt(rec["macro_f1"]), fmt(rec["weighted_f1"]),
            fmt(rec["kappa"]), interpret_kappa(rec["kappa"]),
            ms(rec["latency"]["mean"]), rec["n"],
        ])
    rows.sort(key=lambda r: float(r[5]) if r[5] != "—" else -1, reverse=True)
    rprint_table(rows,
                 headers=["Model", "Type", "Lang", "Ver", "Acc", "M-F1",
                          "W-F1", "κ", "κ interp", "Latency", "N"],
                 title="T2 — All experiments, ranked by Macro-F1")
    commentary("This table shows every (model, language, prompt version) combination. "
               "The best configuration per model is selected in T3.")


def print_t3_best_per_model(records, cfg, task):
    section(f"T3 — Best Configuration per Model: {cfg['display_name']}")
    best = choose_best_per_model(records, "macro_f1")
    rows = []
    for i, rec in enumerate(best):
        marker = " ★" if i == 0 else ""
        rows.append([
            rec["model_display"] + marker, rec["type"], rec["params"],
            f"{rec['lang'].upper()} {rec['version']}",
            pct(rec["accuracy"]), fmt(rec["macro_f1"]), fmt(rec["weighted_f1"]),
            fmt(rec["kappa"]), interpret_kappa(rec["kappa"]),
            ms(rec["latency"]["mean"]), rec["n"],
        ])
    rprint_table(rows,
                 headers=["Model", "Type", "Params", "Config", "Acc", "M-F1",
                          "W-F1", "κ", "κ interp", "Latency", "N"],
                 title=f"T3 — Best prompt per model ({cfg['display_name']})")

    if best:
        top = best[0]
        commentary(f"Best overall: {top['model_display']} with config "
                   f"{top['lang'].upper()} {top['version']} — "
                   f"Accuracy={pct(top['accuracy'])}, Macro-F1={fmt(top['macro_f1'])}, "
                   f"κ={fmt(top['kappa'])} ({interpret_kappa(top['kappa'])}).")

    # ── Generate LaTeX ──
    _generate_latex_best(best, cfg, task)

    return best


def print_t4_per_class(best_records, labels, cfg):
    section(f"T4 — Per-Class Metrics (Best Configs): {cfg['display_name']}")
    for rec in best_records:
        p, r, f1, sup = precision_recall_fscore_support(
            rec["y_true"], rec["y_pred"],
            labels=labels, zero_division=0,
        )
        rows = []
        for label, prec, recall, f, s in zip(labels, p, r, f1, sup):
            rows.append([label, fmt(prec), fmt(recall), fmt(f), int(s)])
        rprint_table(rows,
                     headers=["Class", "Precision", "Recall", "F1", "Support"],
                     title=f"{rec['model_display']} [{rec['lang'].upper()} {rec['version']}]")

    # Generate per-class LaTeX
    _generate_latex_per_class(best_records, labels, cfg)


def print_t5_error_analysis(best_records, cfg):
    section(f"T5 — Error Analysis (Best Configs): {cfg['display_name']}")
    for rec in best_records:
        errors = defaultdict(lambda: defaultdict(int))
        for true, pred in zip(rec["y_true"], rec["y_pred"]):
            if true != pred:
                errors[true][pred] += 1

        title = f"{rec['model_display']} [{rec['lang'].upper()} {rec['version']}]"
        total_errors = sum(cnt for preds in errors.values() for cnt in preds.values())
        total_preds = len(rec["y_true"])

        if not errors:
            rprint(f"\n  ▶ {title} — 0 errors (perfect)", style="green")
            continue

        rprint(f"\n  ▶ {title} — {total_errors} errors out of {total_preds} "
               f"({pct(total_errors / total_preds)} error rate)", style="bold")

        rows = []
        for true_label, preds in errors.items():
            for pred_label, cnt in preds.items():
                rows.append([true_label, pred_label, cnt,
                             pct(cnt / total_preds)])
        rows.sort(key=lambda x: -x[2])
        rprint_table(rows[:15],
                     headers=["True Label", "Predicted As", "Count", "% of Total"],
                     title=title)


def print_t6_version_evolution(by_key, labels, cfg, task):
    section(f"T6 — Prompt Version Evolution: {cfg['display_name']}")
    # Collect (model, lang) pairs and their versions
    model_lang_versions = defaultdict(dict)
    for (model, lang, version), rows in by_key.items():
        metrics = compute_multiclass_metrics(rows, labels, task)
        if metrics:
            model_lang_versions[(model, lang)][version] = {
                "acc": metrics["accuracy"],
                "mf1": metrics["macro_f1"],
                "kappa": metrics["kappa"],
            }

    versions_seen = sorted({v for vdict in model_lang_versions.values() for v in vdict})

    rows = []
    for (model, lang) in sorted(model_lang_versions):
        vdata = model_lang_versions[(model, lang)]
        row = [MODEL_DISPLAY.get(model, model), lang.upper()]
        for v in versions_seen:
            if v in vdata:
                row.append(f"{pct(vdata[v]['acc'])} / {fmt(vdata[v]['mf1'])}")
            else:
                row.append("—")
        # Delta first→last
        ordered = [vdata[v]["mf1"] for v in versions_seen if v in vdata]
        if len(ordered) >= 2:
            d = ordered[-1] - ordered[0]
            row.append(f"{'+' if d >= 0 else ''}{d * 100:.1f}%")
        else:
            row.append("—")
        rows.append(row)

    rprint_table(rows,
                 headers=["Model", "Lang"] +
                         [f"Acc/F1 {v}" for v in versions_seen] +
                         ["ΔF1 first→last"],
                 title=f"T6 — Version evolution ({cfg['display_name']})")

    commentary("Each cell shows Accuracy / Macro-F1 for that prompt version. "
               "The ΔF1 column measures the net improvement from the earliest "
               "to the latest prompt version.")


def print_t7_language_impact(records, cfg):
    section(f"T7 — Language Impact (RO vs EN): {cfg['display_name']}")
    # Group best by model+lang
    by_model = defaultdict(dict)
    for rec in records:
        key = (rec["model"], rec["lang"])
        if rec["model"] not in by_model or rec["lang"] not in by_model[rec["model"]]:
            by_model[rec["model"]][rec["lang"]] = rec
        else:
            # Keep the best version per lang
            existing = by_model[rec["model"]][rec["lang"]]
            if rec["macro_f1"] > existing["macro_f1"]:
                by_model[rec["model"]][rec["lang"]] = rec

    rows = []
    for model in sorted(by_model):
        lang_data = by_model[model]
        ro = lang_data.get("ro")
        en = lang_data.get("en")
        if not ro and not en:
            continue

        acc_ro = ro["accuracy"] if ro else None
        acc_en = en["accuracy"] if en else None
        f1_ro = ro["macro_f1"] if ro else None
        f1_en = en["macro_f1"] if en else None
        lat_ro = ro["latency"]["mean"] if ro else None
        lat_en = en["latency"]["mean"] if en else None

        # Determine winner
        if f1_ro is not None and f1_en is not None:
            d = f1_ro - f1_en
            winner = "RO" if d > 0.01 else "EN" if d < -0.01 else "≈"
        elif f1_ro is not None:
            winner = "RO (only)"
        elif f1_en is not None:
            winner = "EN (only)"
        else:
            winner = "—"

        rows.append([
            MODEL_DISPLAY.get(model, model), MODEL_TYPE.get(model, "?"),
            pct(acc_ro), pct(acc_en), delta_pct(acc_ro, acc_en),
            fmt(f1_ro), fmt(f1_en), delta_pct(f1_ro, f1_en),
            ms(lat_ro), ms(lat_en), delta_ms(lat_ro, lat_en),
            winner,
        ])

    rprint_table(rows,
                 headers=["Model", "Type", "Acc RO", "Acc EN", "ΔAcc",
                          "F1 RO", "F1 EN", "ΔF1",
                          "Lat RO", "Lat EN", "ΔLat", "Winner"],
                 title=f"T7 — Language impact ({cfg['display_name']})")

    commentary("For tasks on Romanian banking conversations, RO prompts often align "
               "better with the conversational register. EN prompts can help with models "
               "that have stronger English pretraining. The winner column uses a 1% "
               "threshold on Macro-F1.")

    _generate_latex_language(rows, cfg)


def print_t8_latency(records, cfg):
    section(f"T8 — Latency Analysis: {cfg['display_name']}")
    best = choose_best_per_model(records, "macro_f1")
    rows = []
    for rec in best:
        lat = rec["latency"]
        rows.append([
            rec["model_display"], rec["type"],
            f"{rec['lang'].upper()} {rec['version']}",
            ms(lat["mean"]), ms(lat["median"]), ms(lat["std"]),
            ms(lat["p95"]), ms(lat["min"]), ms(lat["max"]),
            lat["n_samples"],
        ])
    rows.sort(key=lambda r: float(r[3].replace(" ms", "")) if r[3] != "—" else 1e9)
    rprint_table(rows,
                 headers=["Model", "Type", "Config", "Mean", "Median", "Std",
                          "p95", "Min", "Max", "N"],
                 title=f"T8 — Latency for best configs ({cfg['display_name']})")

    commentary("API models include network round-trip time. Local models run on "
               "consumer hardware with quantized weights (Q4_K_M for Ollama models). "
               "p95 is most relevant for production SLA considerations.")

    _generate_latex_latency(rows, cfg)


def print_t9_reliability(by_key, cfg):
    section(f"T9 — Output Reliability: {cfg['display_name']}")
    rows = []
    for key in sorted(by_key):
        model, lang, version = key
        preds = by_key[key]
        total = len(preds)
        if total == 0:
            continue
        fails = sum(1 for r in preds if r.get("parse_failed", False))
        high = sum(1 for r in preds if r.get("confidence") == "high")
        med = sum(1 for r in preds if r.get("confidence") == "medium")
        low = sum(1 for r in preds if r.get("confidence") == "low")
        conf_available = high + med + low > 0
        rows.append([
            MODEL_DISPLAY.get(model, model), lang.upper(), version,
            total, fails, pct(fails / total),
            pct(high / total) if conf_available else "—",
            pct(med / total) if conf_available else "—",
            pct(low / total) if conf_available else "—",
        ])
    rprint_table(rows,
                 headers=["Model", "Lang", "Ver", "Total", "Fails", "Fail%",
                          "High%", "Med%", "Low%"],
                 title=f"T9 — Output reliability ({cfg['display_name']})")

    commentary("Parse failures indicate the model returned output that could not be "
               "parsed into the expected JSON schema. High failure rates correlate "
               "with weaker instruction-following capability.")


def print_binary_tables(by_key, cfg, task):
    """Binary detection tables (incongruities only)."""
    if "binary" not in cfg:
        return None

    section(f"T-BIN — Binary Detection: {cfg['display_name']}")
    records = collect_binary_records(by_key)

    rows = []
    for rec in records:
        rows.append([
            rec["model_display"], rec["type"], rec["lang"].upper(), rec["version"],
            pct(rec["accuracy"]), fmt(rec["f1"]), fmt(rec["kappa"]),
            interpret_kappa(rec["kappa"]),
            ms(rec["latency"]["mean"]), rec["n"],
        ])
    rows.sort(key=lambda r: float(r[5]) if r[5] != "—" else -1, reverse=True)
    rprint_table(rows,
                 headers=["Model", "Type", "Lang", "Ver", "Acc", "F1", "κ",
                          "κ interp", "Lat", "N"],
                 title="T-BIN — All binary detection results")

    best = choose_best_per_model(records, "f1")
    rows2 = []
    for rec in best:
        rows2.append([
            rec["model_display"], rec["type"],
            f"{rec['lang'].upper()} {rec['version']}",
            pct(rec["accuracy"]), fmt(rec["f1"]), fmt(rec["kappa"]),
            interpret_kappa(rec["kappa"]),
        ])
    rprint_table(rows2,
                 headers=["Model", "Type", "Config", "Acc", "F1", "κ", "κ interp"],
                 title="T-BIN-BEST — Best binary detection per model")
    return best


# ════════════════════════════════════════════════════════════════════════════
# ██  COMPARATIVE ANALYSES
# ════════════════════════════════════════════════════════════════════════════

def print_c1_api_vs_local(records, cfg):
    section(f"C1 — API vs Local Comparison: {cfg['display_name']}")

    best = choose_best_per_model(records, "macro_f1")
    api_models = [r for r in best if r["type"] == "API"]
    local_models = [r for r in best if r["type"] == "Local"]

    def avg_metric(group, key):
        vals = [r[key] for r in group if r[key] is not None]
        return np.mean(vals) if vals else None

    def avg_latency(group):
        vals = [r["latency"]["mean"] for r in group if r["latency"]["mean"] is not None]
        return np.mean(vals) if vals else None

    rows = []
    for label, group in [("API models", api_models), ("Local models", local_models)]:
        if not group:
            continue
        rows.append([
            label, len(group),
            pct(avg_metric(group, "accuracy")),
            fmt(avg_metric(group, "macro_f1")),
            fmt(avg_metric(group, "kappa")),
            ms(avg_latency(group)),
            ", ".join(r["model_display"] for r in group),
        ])

    rprint_table(rows,
                 headers=["Category", "N models", "Avg Acc", "Avg M-F1",
                          "Avg κ", "Avg Lat", "Models"],
                 title=f"C1 — API vs Local ({cfg['display_name']})")

    # Individual comparison
    rows2 = []
    for rec in best:
        rows2.append([
            rec["model_display"], rec["type"], rec["params"],
            pct(rec["accuracy"]), fmt(rec["macro_f1"]), fmt(rec["kappa"]),
            ms(rec["latency"]["mean"]),
            MODEL_INFERENCE_INFO.get(rec["model"], {}).get("provider", "?"),
            MODEL_INFERENCE_INFO.get(rec["model"], {}).get("quantization", "?"),
        ])
    rprint_table(rows2,
                 headers=["Model", "Type", "Params", "Acc", "M-F1", "κ",
                          "Latency", "Provider", "Quantization"],
                 title=f"C1b — Individual model details ({cfg['display_name']})")

    if api_models and local_models:
        best_api = api_models[0]
        best_local = local_models[0]
        gap = best_api["macro_f1"] - best_local["macro_f1"]
        commentary(
            f"API vs Local gap (Macro-F1): {gap * 100:+.1f}%. "
            f"Best API: {best_api['model_display']} ({fmt(best_api['macro_f1'])}), "
            f"Best Local: {best_local['model_display']} ({fmt(best_local['macro_f1'])}). "
            f"This {'confirms' if gap > 0.05 else 'suggests a narrow gap, indicating'} "
            f"{'the advantage of large proprietary models' if gap > 0.05 else 'local models are competitive'} "
            f"for this task."
        )

    _generate_latex_api_vs_local(best, cfg)


def print_c2_param_efficiency(records, cfg):
    section(f"C2 — Parameter Efficiency: {cfg['display_name']}")
    best = choose_best_per_model(records, "macro_f1")
    rows = []
    for rec in best:
        p = rec["params_b"]
        f1 = rec["macro_f1"]
        if p is not None and f1 is not None and p > 0:
            efficiency = f1 / np.log10(p * 1e9)  # F1 per log10(params)
        else:
            efficiency = None
        rows.append([
            rec["model_display"], rec["type"], rec["params"],
            pct(rec["accuracy"]), fmt(rec["macro_f1"]),
            fmt(efficiency) if efficiency else "—",
        ])
    rprint_table(rows,
                 headers=["Model", "Type", "Params", "Acc", "M-F1",
                          "F1/log₁₀(params)"],
                 title=f"C2 — Parameter efficiency ({cfg['display_name']})")

    commentary("F1/log₁₀(params) measures how much performance a model squeezes "
               "per order of magnitude of parameters. Smaller models with high "
               "scores are exceptionally parameter-efficient.")


def print_c3_latency_accuracy_tradeoff(records, cfg):
    section(f"C3 — Latency–Accuracy Trade-off: {cfg['display_name']}")
    best = choose_best_per_model(records, "macro_f1")
    rows = []
    for rec in best:
        lat = rec["latency"]["mean"]
        acc = rec["accuracy"]
        if lat is not None and lat > 0 and acc is not None:
            # "Points per second" — accuracy gained per second of latency
            points_per_sec = acc / (lat / 1000)
        else:
            points_per_sec = None
        rows.append([
            rec["model_display"], rec["type"],
            pct(rec["accuracy"]), fmt(rec["macro_f1"]),
            ms(lat),
            sec(lat / 1000 if lat else None),
            fmt(points_per_sec, 1) if points_per_sec else "—",
        ])
    rows.sort(key=lambda r: float(r[6]) if r[6] != "—" else 0, reverse=True)
    rprint_table(rows,
                 headers=["Model", "Type", "Acc", "M-F1", "Latency (ms)",
                          "Latency (s)", "Acc/sec"],
                 title=f"C3 — Latency vs Accuracy ({cfg['display_name']})")

    commentary("Acc/sec measures how much accuracy is delivered per second of "
               "inference time. Models with high accuracy AND low latency rank "
               "highest — relevant for real-time monitoring pipelines.")


def print_c5_kappa_interpretation(records, cfg):
    section(f"C5 — Agreement Analysis (Cohen's κ): {cfg['display_name']}")
    best = choose_best_per_model(records, "macro_f1")
    rows = []
    for rec in best:
        k = rec["kappa"]
        rows.append([
            rec["model_display"], rec["type"],
            f"{rec['lang'].upper()} {rec['version']}",
            fmt(k), interpret_kappa(k),
            pct(rec["accuracy"]), fmt(rec["macro_f1"]),
            pct(rec.get("invalid_rate", 0)),
        ])
    rprint_table(rows,
                 headers=["Model", "Type", "Config", "κ", "Interpretation",
                          "Acc", "M-F1", "Invalid%"],
                 title=f"C5 — κ interpretation per Landis & Koch (1977)")

    commentary("Cohen's κ corrects for chance agreement, making it stricter than "
               "raw accuracy. Invalid predictions (unparseable outputs mapped to "
               "__invalid__) are penalized in the κ computation. "
               "Interpretation: <0.20=Slight, 0.21-0.40=Fair, 0.41-0.60=Moderate, "
               "0.61-0.80=Substantial, 0.81-1.00=Almost Perfect.")


# ════════════════════════════════════════════════════════════════════════════
# ██  CROSS-TASK ANALYSIS (--task all)
# ════════════════════════════════════════════════════════════════════════════

def print_c4_cross_task(all_task_records):
    """Compare model rankings across tasks — Spearman rank correlation."""
    section("C4 — Cross-Task Model Ranking Stability")

    tasks = list(all_task_records.keys())
    if len(tasks) < 2:
        rprint("  Need at least 2 tasks for cross-task comparison.")
        return

    # Get best per model for each task
    task_rankings = {}
    all_models = set()
    for task_name, records in all_task_records.items():
        best = choose_best_per_model(records, "macro_f1")
        task_rankings[task_name] = {r["model"]: r["macro_f1"] for r in best}
        all_models.update(r["model"] for r in best)

    # Build ranking table
    rows = []
    for model in sorted(all_models):
        row = [MODEL_DISPLAY.get(model, model), MODEL_TYPE.get(model, "?")]
        for task_name in tasks:
            f1 = task_rankings[task_name].get(model)
            row.append(fmt(f1) if f1 is not None else "—")
        # Average F1 across tasks
        vals = [task_rankings[t].get(model) for t in tasks if task_rankings[t].get(model)]
        avg = np.mean(vals) if vals else None
        row.append(fmt(avg))
        rows.append(row)

    rows.sort(key=lambda r: float(r[-1]) if r[-1] != "—" else 0, reverse=True)

    task_headers = [TASK_CONFIG[t]["display_name_en"] for t in tasks]
    rprint_table(rows,
                 headers=["Model", "Type"] + task_headers + ["Avg M-F1"],
                 title="C4 — Cross-task comparison (best M-F1 per model)")

    # Spearman rank correlation between tasks
    from itertools import combinations as combs
    common_models = [m for m in all_models
                     if all(m in task_rankings[t] for t in tasks)]

    if len(common_models) >= 3:
        rprint("\n  Rank correlation (Spearman ρ) between tasks:", style="bold")
        for t1, t2 in combs(tasks, 2):
            r1 = [task_rankings[t1][m] for m in common_models]
            r2 = [task_rankings[t2][m] for m in common_models]
            from scipy.stats import spearmanr
            try:
                rho, p = spearmanr(r1, r2)
                rprint(f"    {t1} vs {t2}: ρ = {rho:.3f} (p = {p:.3f})")
            except Exception:
                rprint(f"    {t1} vs {t2}: cannot compute (too few models)")

    _generate_latex_cross_task(rows, tasks)

    commentary("A high Spearman ρ between tasks indicates that model quality is "
               "consistent: models that perform well on intent extraction also "
               "tend to perform well on status classification and incongruity "
               "detection. Low ρ would suggest task-specific strengths.")


def print_cross_task_summary(all_task_records):
    """Master summary across all tasks."""
    section("SUMMARY — Cross-Task Overview")

    rows = []
    for task_name, records in all_task_records.items():
        best = choose_best_per_model(records, "macro_f1")
        if not best:
            continue
        top = best[0]
        api_best = [r for r in best if r["type"] == "API"]
        local_best = [r for r in best if r["type"] == "Local"]

        rows.append([
            TASK_CONFIG[task_name]["display_name"],
            len(records),
            top["model_display"],
            f"{top['lang'].upper()} {top['version']}",
            pct(top["accuracy"]),
            fmt(top["macro_f1"]),
            fmt(top["kappa"]),
            ms(top["latency"]["mean"]),
            api_best[0]["model_display"] if api_best else "—",
            local_best[0]["model_display"] if local_best else "—",
        ])

    rprint_table(rows,
                 headers=["Task", "Exps", "Best Model", "Config", "Acc",
                          "M-F1", "κ", "Latency", "Best API", "Best Local"],
                 title="MASTER SUMMARY — Best results per task")


# ════════════════════════════════════════════════════════════════════════════
# ██  LATEX GENERATORS
# ════════════════════════════════════════════════════════════════════════════

def _latex_escape(s):
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def _generate_latex_best(best, cfg, task):
    label_map = {
        "intent": "tab:intent-best",
        "final_status": "tab:final-status-best",
        "incongruities": "tab:incongruity-best",
    }
    caption_map = {
        "intent": "Best prompt configuration per model — Intent Extraction",
        "final_status": "Best prompt configuration per model — Final Status Classification",
        "incongruities": "Best prompt configuration per model — Incongruity Type Classification",
    }

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\footnotesize",
        fr"\caption{{{caption_map.get(task, cfg['display_name'])}}}",
        fr"\label{{{label_map.get(task, 'tab:' + task)}}}",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Config} & \textbf{Acc} & \textbf{M-F1} & \textbf{W-F1} & $\boldsymbol{\kappa}$ & \textbf{Latency} \\",
        r"\midrule",
    ]
    for i, rec in enumerate(best):
        model = _latex_escape(rec["model_display"])
        cfg_txt = f"{rec['lang'].upper()} {rec['version']}"
        acc = pct_latex(rec["accuracy"])
        mf1 = fmt(rec["macro_f1"])
        wf1 = fmt(rec["weighted_f1"])
        kappa = fmt(rec["kappa"])
        lat = ms(rec["latency"]["mean"])
        bold_start = r"\textbf{" if i == 0 else ""
        bold_end = "}" if i == 0 else ""
        lines.append(
            f"{bold_start}{model}{bold_end} & {cfg_txt} & {acc} & {mf1} & {wf1} & {kappa} & {lat} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex = "\n".join(lines)
    _latex_blocks[f"best_{task}"] = tex
    rprint(f"\n  [LaTeX block saved: best_{task}]", style="dim")


def _generate_latex_per_class(best_records, labels, cfg):
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\footnotesize",
        fr"\caption{{Per-class metrics for best configurations — {cfg['display_name']}}}",
        r"\label{tab:per-class-" + cfg['display_name_en'].lower().replace(' ', '-') + "}",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Class} & \textbf{Precision} & \textbf{Recall} & \textbf{F1} & \textbf{Support} \\",
        r"\midrule",
    ]
    for rec in best_records:
        p, r, f1, sup = precision_recall_fscore_support(
            rec["y_true"], rec["y_pred"], labels=labels, zero_division=0)
        model = _latex_escape(rec["model_display"])
        for j, (label, prec, recall, f, s) in enumerate(zip(labels, p, r, f1, sup)):
            m_col = model if j == 0 else ""
            lines.append(
                f"{m_col} & {_latex_escape(label)} & {fmt(prec)} & {fmt(recall)} & {fmt(f)} & {int(s)} \\\\"
            )
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table}"]
    key = f"per_class_{cfg['display_name_en'].lower().replace(' ', '_')}"
    _latex_blocks[key] = "\n".join(lines)


def _generate_latex_language(rows, cfg):
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\footnotesize",
        fr"\caption{{Language impact analysis — {cfg['display_name']}}}",
        r"\label{tab:lang-impact-" + cfg['display_name_en'].lower().replace(' ', '-') + "}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Acc RO} & \textbf{Acc EN} & \textbf{$\Delta$Acc} & \textbf{F1 RO} & \textbf{F1 EN} & \textbf{Winner} \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{_latex_escape(row[0])} & {row[2]} & {row[3]} & {row[4]} & "
            f"{row[5]} & {row[6]} & {row[11]} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    key = f"lang_impact_{cfg['display_name_en'].lower().replace(' ', '_')}"
    _latex_blocks[key] = "\n".join(lines)


def _generate_latex_latency(rows, cfg):
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\footnotesize",
        fr"\caption{{Latency analysis — {cfg['display_name']}}}",
        r"\label{tab:latency-" + cfg['display_name_en'].lower().replace(' ', '-') + "}",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Type} & \textbf{Mean} & \textbf{Median} & \textbf{Std} & \textbf{p95} & \textbf{N} \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{_latex_escape(row[0])} & {row[1]} & {row[3]} & {row[4]} & "
            f"{row[5]} & {row[6]} & {row[9]} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    key = f"latency_{cfg['display_name_en'].lower().replace(' ', '_')}"
    _latex_blocks[key] = "\n".join(lines)


def _generate_latex_api_vs_local(best, cfg):
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\footnotesize",
        fr"\caption{{API vs Local comparison — {cfg['display_name']}}}",
        r"\label{tab:api-vs-local-" + cfg['display_name_en'].lower().replace(' ', '-') + "}",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Type} & \textbf{Params} & \textbf{Acc} & \textbf{M-F1} & $\boldsymbol{\kappa}$ & \textbf{Latency} \\",
        r"\midrule",
    ]
    for rec in best:
        lines.append(
            f"{_latex_escape(rec['model_display'])} & {rec['type']} & "
            f"{rec['params']} & {pct_latex(rec['accuracy'])} & "
            f"{fmt(rec['macro_f1'])} & {fmt(rec['kappa'])} & "
            f"{ms(rec['latency']['mean'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    key = f"api_vs_local_{cfg['display_name_en'].lower().replace(' ', '_')}"
    _latex_blocks[key] = "\n".join(lines)


def _generate_latex_cross_task(rows, tasks):
    task_headers = [TASK_CONFIG[t]["display_name_en"] for t in tasks]
    n_cols = 2 + len(tasks) + 1
    col_spec = "ll" + "c" * (len(tasks) + 1)
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\footnotesize",
        r"\caption{Cross-task comparison — Best Macro-F1 per model}",
        r"\label{tab:cross-task}",
        fr"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Type} & " +
        " & ".join(fr"\textbf{{{_latex_escape(h)}}}" for h in task_headers) +
        r" & \textbf{Avg} \\",
        r"\midrule",
    ]
    for row in rows:
        cells = [_latex_escape(str(c)) for c in row]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _latex_blocks["cross_task"] = "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════
# ██  DEBUG
# ════════════════════════════════════════════════════════════════════════════

def debug_experiments(experiments):
    section("DEBUG — First experiment structure")
    if not experiments:
        rprint("  No experiments loaded.")
        return
    exp = experiments[0]
    rprint(f"  File: {exp['path'].name}")
    rprint(f"  Model: {exp['model']}, Lang: {exp['lang']}, Version: {exp['version']}")
    rprint(f"  Predictions: {len(exp['predictions'])}")
    if exp["predictions"]:
        first = exp["predictions"][0]
        rprint(f"\n  Keys in first prediction: {list(first.keys())}")
        rprint(f"\n  First prediction (truncated):")
        rprint(json.dumps(first, ensure_ascii=False, indent=2)[:2000])


# ════════════════════════════════════════════════════════════════════════════
# ██  MAIN RUNNERS
# ════════════════════════════════════════════════════════════════════════════

def run_single_task(task, filter_model=None, filter_lang=None,
                    filter_version=None, debug=False, save_latex=False):
    """Run full evaluation for one task."""
    global _report_lines
    _report_lines = []

    cfg = TASK_CONFIG[task]
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    section(f"EVALUATION: {cfg['display_name']} ({cfg['display_name_en']})")
    rprint(f"  Task: {task}")
    rprint(f"  Output dir: {cfg['output_dir']}")
    rprint(f"  Report: {cfg['report_path']}")
    rprint(f"  Timestamp: {now}")
    if filter_model:
        rprint(f"  Filter model: {filter_model}")
    if filter_lang:
        rprint(f"  Filter lang: {filter_lang}")
    if filter_version:
        rprint(f"  Filter version: {filter_version}")
    rprint()

    labels = load_labels(cfg)
    rprint(f"  Labels ({len(labels)}): {labels}\n", style="bold")

    experiments = load_all_experiments(cfg, filter_model, filter_lang, filter_version)

    if debug:
        debug_experiments(experiments)

    by_key = build_by_experiment(experiments, cfg)

    # ── Per-task tables ──
    print_t1_overview(by_key, cfg)

    records = collect_multiclass_records(by_key, labels, task)

    print_t2_all_experiments(records, cfg)
    best = print_t3_best_per_model(records, cfg, task)

    if best:
        print_t4_per_class(best, labels, cfg)
        print_t5_error_analysis(best, cfg)

    print_t6_version_evolution(by_key, labels, cfg, task)
    print_t7_language_impact(records, cfg)
    print_t8_latency(records, cfg)
    print_t9_reliability(by_key, cfg)

    # Incongruity-specific binary tables
    if task == "incongruities":
        print_binary_tables(by_key, cfg, task)

    # ── Comparative analyses ──
    print_c1_api_vs_local(records, cfg)
    print_c2_param_efficiency(records, cfg)
    print_c3_latency_accuracy_tradeoff(records, cfg)
    print_c5_kappa_interpretation(records, cfg)

    # ── Wrapup ──
    rprint(f"\n{'═' * 100}", style="cyan")
    rprint(f"  EVALUATION COMPLETE: {cfg['display_name']}", style="bold green")
    rprint(f"{'═' * 100}", style="cyan")

    save_report(cfg["report_path"])

    if save_latex:
        save_latex_files(cfg["latex_dir"])

    return records


def run_all_tasks(filter_model=None, filter_lang=None,
                  filter_version=None, debug=False, save_latex=False):
    """Run evaluation for all 3 tasks + cross-task analysis."""
    global _report_lines, _latex_blocks

    all_task_records = {}

    for task in ["intent", "final_status", "incongruities"]:
        try:
            records = run_single_task(
                task, filter_model, filter_lang, filter_version,
                debug=debug, save_latex=False,  # We save at the end
            )
            all_task_records[task] = records
        except (FileNotFoundError, ValueError) as e:
            rprint(f"\n  ⚠ Skipping task '{task}': {e}", style="yellow")

    if len(all_task_records) >= 2:
        # Reset report for cross-task
        _report_lines = []
        section("CROSS-TASK ANALYSIS")

        print_c4_cross_task(all_task_records)
        print_cross_task_summary(all_task_records)

        # Save cross-task report
        cross_report_path = BASE_DIR / "evaluation_reports" / "eval_cross_task.txt"
        save_report(cross_report_path)

    if save_latex:
        latex_dir = BASE_DIR / "evaluation_reports" / "latex"
        save_latex_files(latex_dir)

    rprint(f"\n{'═' * 100}", style="bold green")
    rprint(f"  ALL EVALUATIONS COMPLETE", style="bold green")
    rprint(f"  Tasks processed: {', '.join(all_task_records.keys())}", style="green")
    rprint(f"{'═' * 100}", style="bold green")


# ════════════════════════════════════════════════════════════════════════════
# ██  CLI
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate LLMs — Chapter 5 Thesis Evaluation Suite",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--task", type=str, required=True,
        choices=["intent", "final_status", "incongruities", "all"],
        help="Which task to evaluate. 'all' runs all 3 + cross-task analysis.",
    )
    parser.add_argument("--model", type=str, default=None,
                        help="Filter by model name (e.g. openai_o3)")
    parser.add_argument("--lang", type=str, default=None,
                        help="Filter by language (ro / en)")
    parser.add_argument("--version", type=str, default=None,
                        help="Filter by prompt version (v1, v2, v3, v4)")
    parser.add_argument("--debug", action="store_true",
                        help="Show first experiment structure for debugging")
    parser.add_argument("--save-latex", action="store_true",
                        help="Save LaTeX table files to evaluation_reports/latex/")

    args = parser.parse_args()

    if args.task == "all":
        run_all_tasks(
            filter_model=args.model,
            filter_lang=args.lang,
            filter_version=args.version,
            debug=args.debug,
            save_latex=args.save_latex,
        )
    else:
        run_single_task(
            task=args.task,
            filter_model=args.model,
            filter_lang=args.lang,
            filter_version=args.version,
            debug=args.debug,
            save_latex=args.save_latex,
        )