# src/prompting/evaluate_best_all_tasks.py

"""
Evaluare unified pentru toate cele 3 taskuri:
    1. intent
    2. final_status
    3. incongruities

Ce face:
    - Încarcă toate experimentele disponibile, nu doar ultima versiune.
    - Grupează pe (model, lang, version).
    - Calculează metrici pentru toate experimentele.
    - Alege best prompt per model după Macro-F1, apoi Accuracy.
    - Calculează Cohen's Kappa corect, penalizând predicțiile invalide.
    - Produce tabele:
        T1           overview toate experimentele
        T-ALL        toate experimentele cu metrici
        T-BEST       best prompt per model
        T-PER-CLASS  metrici per clasă pentru best prompt
        T-ERR-BEST   analiză erori pentru best prompt
        LaTeX        tabel pentru lucrare

Utilizare:
    python evaluate_best_all_tasks.py --task intent
    python evaluate_best_all_tasks.py --task final_status
    python evaluate_best_all_tasks.py --task incongruities

Opțional:
    python evaluate_best_all_tasks.py --task intent --debug
    python evaluate_best_all_tasks.py --task final_status --model openai_o3
    python evaluate_best_all_tasks.py --task incongruities --lang ro
    python evaluate_best_all_tasks.py --task intent --version v4
"""

import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    cohen_kappa_score,
    precision_recall_fscore_support,
)

from tabulate import tabulate

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box

    HAS_RICH = True
    console = Console()
except ImportError:
    HAS_RICH = False
    console = None


# ──────────────────────────────────────────────
# Base config
# ──────────────────────────────────────────────

BASE_DIR = Path(
    r"C:\Users\Matebook 14s\Documents"
    r"\Sistem-de-monitorizare-a-interac-iunilor-voicebotilor-folosind-modele-lingvistice-mari-LLM-"
)

TASK_CONFIG = {
    "intent": {
        "display_name": "Extragerea intenției",
        "output_dir": BASE_DIR / "outputs" / "intent",
        "report_path": BASE_DIR / "evaluation_report_intent_best.txt",
        "labels_path": BASE_DIR / "configs" / "intent_definitions.json",
        "labels_default": None,
        "file_pattern": "*.json",
        "pred_key_candidates": ["predictions", "results"],
        "pred_candidates": [
            "predicted_intent",
            "predicted_label",
            "prediction",
            "pred_label",
        ],
        "true_candidates": [
            "dataset_label",
            "dataset_intent",
            "ground_truth",
            "true_label",
            "label",
        ],
        "prefixes_to_remove": ["exp_"],
    },
    "final_status": {
        "display_name": "Clasificarea statusului final",
        "output_dir": BASE_DIR / "outputs_final_status",
        "report_path": BASE_DIR / "evaluation_report_final_status_best.txt",
        "labels_path": BASE_DIR / "configs" / "final_status_definitions.json",
        "labels_default": [
            "rezolvata",
            "partial_rezolvata",
            "nerezolvata",
            "intrerupta",
            "redirectionata",
        ],
        "file_pattern": "*.json",
        "pred_key_candidates": ["results", "predictions"],
        "pred_candidates": [
            "predicted_status",
            "predicted_final_status",
            "predicted_label",
            "prediction",
            "pred_label",
        ],
        "true_candidates": [
            "dataset_status",
            "dataset_label",
            "ground_truth",
            "true_label",
            "label",
        ],
        "prefixes_to_remove": ["exp_fst_", "fst_", "exp_"],
    },
    "incongruities": {
        "display_name": "Detecția neconcordanțelor",
        "output_dir": BASE_DIR / "outputs_incongruities",
        "report_path": BASE_DIR / "evaluation_report_incongruities_best.txt",
        "labels_path": BASE_DIR / "configs" / "incongruities_definitions.json",
        "labels_default": [
            "incomplet",
            "irelevant",
            "contradictoriu",
            "nealiniat_context",
            "halucinatie",
        ],
        "file_pattern": "*.json",
        "pred_key_candidates": ["predictions", "results"],
        "pred_candidates": [
            "predicted_type",
            "predicted_incongruity_type",
            "predicted_label",
            "prediction",
            "pred_label",
        ],
        "true_candidates": [
            "dataset_label",
            "dataset_type",
            "dataset_incongruity_type",
            "ground_truth",
            "true_label",
            "label",
        ],
        "prefixes_to_remove": ["exp_inc_", "inc_", "exp_"],
        "binary": {
            "pred_candidates": [
                "predicted_has_incongruity",
                "has_incongruity",
                "predicted_binary",
                "prediction_binary",
            ],
            "true_candidates": [
                "dataset_has_incongruity",
                "ground_truth_has_incongruity",
                "true_has_incongruity",
                "label_binary",
            ],
        },
    },
}


MODEL_TYPE = {
    "openai_o3": "API",
    "gemini_2.5_flash": "API",
    "aya_expanse_8b": "Local",
    "rollama2_7b": "Local",
    "roberta_encoder": "Local",
    "robert_encoder": "Local",
}

MODEL_DISPLAY = {
    "openai_o3": "OpenAI o3",
    "gemini_2.5_flash": "Gemini 2.5 Flash",
    "aya_expanse_8b": "Aya Expanse 8B",
    "rollama2_7b": "RoLLaMA 2 7B",
    "roberta_encoder": "XLM-RoBERTa",
    "robert_encoder": "RoBERT-base",
}

MODEL_PARAMS = {
    "openai_o3": "~200B+",
    "gemini_2.5_flash": "~unknown",
    "aya_expanse_8b": "8B",
    "rollama2_7b": "7B",
    "roberta_encoder": "~560M",
    "robert_encoder": "~125M",
}


# ──────────────────────────────────────────────
# Output helpers
# ──────────────────────────────────────────────

_report_lines = []


def rprint(text="", style=None):
    _report_lines.append(str(text) + "\n")

    if HAS_RICH and console:
        console.print(text, style=style) if style else console.print(text)
    else:
        print(text)


def rprint_table(rows, headers, title=None):
    plain = tabulate(rows, headers=headers, tablefmt="rounded_outline")
    _report_lines.append(plain + "\n\n")

    if HAS_RICH and console:
        table = Table(
            title=title,
            box=box.ROUNDED,
            header_style="bold cyan",
            border_style="blue",
            show_lines=True,
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
    _report_lines.append(f"\n{'═' * 90}\n  {title}\n{'═' * 90}\n")

    if HAS_RICH and console:
        console.print()
        console.print(
            Panel(
                f"[bold white]{title}[/bold white]",
                border_style="cyan",
                expand=True,
            )
        )
    else:
        print(f"\n{'═' * 90}\n  {title}\n{'═' * 90}")


def save_report(path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(_report_lines)

    rprint(f"\n✓ Raport salvat: {path}", style="bold green")


# ──────────────────────────────────────────────
# Format helpers
# ──────────────────────────────────────────────

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
    return f"{v:.0f}ms" if v is not None else "—"


def get_first_existing(d, candidates):
    for key in candidates:
        if key in d and d.get(key) is not None:
            return d.get(key)
    return None


# ──────────────────────────────────────────────
# Normalization
# ──────────────────────────────────────────────

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


def normalize_label(value):
    if value is None:
        return None

    value = str(value).strip().lower()

    # Curățări simple
    value = value.strip("'").strip('"').strip()
    value = value.replace(" ", "_")

    replacements = {
        # Incongruities
        "halucinație": "halucinatie",
        "halucinatie": "halucinatie",
        "hallucination": "halucinatie",

        "incomplet": "incomplet",
        "incomplete": "incomplet",

        "irelevant": "irelevant",
        "irrelevant": "irelevant",

        "contradictoriu": "contradictoriu",
        "contradiction": "contradictoriu",
        "contradictory": "contradictoriu",

        "nealiniat_context": "nealiniat_context",
        "context_misalignment": "nealiniat_context",
        "misaligned_context": "nealiniat_context",

        "none": "none",
        "no_incongruity": "none",
        "fara_neconcordanta": "none",
        "fără_neconcordanță": "none",
        "nu_exista": "none",
        "absent": "none",

        # Final status variants
        "rezolvată": "rezolvata",
        "rezolvata": "rezolvata",
        "resolved": "rezolvata",

        "parțial_rezolvată": "partial_rezolvata",
        "partial_rezolvata": "partial_rezolvata",
        "partially_resolved": "partial_rezolvata",

        "nerezolvată": "nerezolvata",
        "nerezolvata": "nerezolvata",
        "unresolved": "nerezolvata",

        "întreruptă": "intrerupta",
        "intrerupta": "intrerupta",
        "interrupted": "intrerupta",

        "redirecționată": "redirectionata",
        "redirectionata": "redirectionata",
        "redirected": "redirectionata",

        # Some Romanian intent variants seen in local models
        "blocare_card": "block_card",
        "deblocare_card": "unblock_card",
        "inchidere_cont": "close_account",
        "închidere_cont": "close_account",
        "deschidere_cont": "open_account",
        "sold_cont": "check_balance",
        "extras_cont": "get_account_statement",
        "programare_consultant": "schedule_advisor_meeting",
    }

    return replacements.get(value, value)


def normalize_version(version):
    version = str(version).lower().strip()

    # v3_ro_labels should remain distinct, because it is a real special config
    return version


# ──────────────────────────────────────────────
# Labels
# ──────────────────────────────────────────────

def load_labels(cfg):
    labels_path = cfg["labels_path"]

    if not labels_path.exists():
        if cfg.get("labels_default"):
            rprint(f"⚠ Nu am găsit {labels_path}. Folosesc labels default.")
            return [normalize_label(x) for x in cfg["labels_default"]]

        raise FileNotFoundError(f"Nu există labels file: {labels_path}")

    with open(labels_path, encoding="utf-8") as f:
        defs = json.load(f)

    if "labels" in defs:
        labels = [l["name"] for l in defs["labels"]]
    elif isinstance(defs, list):
        labels = defs
    else:
        labels = cfg.get("labels_default")

    if not labels:
        raise ValueError(f"Nu pot extrage labels din {labels_path}")

    return [normalize_label(x) for x in labels]


# ──────────────────────────────────────────────
# Metadata extraction
# ──────────────────────────────────────────────

def remove_prefixes(name, prefixes):
    out = name

    for p in prefixes:
        if out.startswith(p):
            out = out[len(p):]

    return out


def extract_metadata(path, data, cfg):
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

    exp_name = data.get("experiment_name", meta.get("name", path.stem))
    parts = exp_name.split("__")

    if len(parts) >= 3:
        model_from_name = remove_prefixes(parts[0], cfg["prefixes_to_remove"])
        lang_from_name = parts[1]
        version_from_name = parts[2].replace(".json", "")

        if model == "unknown":
            model = model_from_name
        if lang == "unknown":
            lang = lang_from_name
        if version == "unknown":
            version = version_from_name

    model = str(model).lower().strip()
    lang = str(lang).lower().strip()
    version = normalize_version(version)

    return model, lang, version, ts


def get_predictions_from_data(data, cfg):
    for key in cfg["pred_key_candidates"]:
        if key in data and isinstance(data[key], list):
            return data[key]

    return []


# ──────────────────────────────────────────────
# Loading
# ──────────────────────────────────────────────

def load_all_experiments(cfg, filter_model=None, filter_lang=None, filter_version=None):
    output_dir = cfg["output_dir"]

    # rglob ca să găsească și fișiere din subfoldere, dacă există.
    exp_files = sorted(output_dir.rglob(cfg["file_pattern"]))

    if not exp_files:
        raise FileNotFoundError(f"Nu s-au găsit fișiere JSON în {output_dir}")

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
            "model": model,
            "lang": lang,
            "version": version,
            "timestamp": ts,
            "path": path,
            "data": data,
            "predictions": preds,
        })

    if not experiments:
        raise ValueError("Niciun experiment nu corespunde filtrelor.")

    rprint(f"\nExperimente încărcate ({len(experiments)}):", style="bold")

    for exp in sorted(experiments, key=lambda e: (e["model"], e["lang"], e["version"], e["path"].name)):
        rprint(
            f"  {MODEL_DISPLAY.get(exp['model'], exp['model']):<25} "
            f"[{exp['lang'].upper()}] {exp['version']:<14} ← {exp['path'].name}"
        )

    return experiments


# ──────────────────────────────────────────────
# Prediction normalization
# ──────────────────────────────────────────────

def normalize_prediction_row(row, exp, cfg):
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

    if "binary" in cfg:
        pred_binary_raw = get_first_existing(row, cfg["binary"]["pred_candidates"])
        true_binary_raw = get_first_existing(row, cfg["binary"]["true_candidates"])

        pred_binary = normalize_bool(pred_binary_raw)
        true_binary = normalize_bool(true_binary_raw)

        if true_binary is None and true_norm is not None:
            true_binary = true_norm != "none"

        if pred_binary is None and pred_norm is not None:
            pred_binary = pred_norm != "none"

        normalized["predicted_binary_norm"] = pred_binary
        normalized["true_binary_norm"] = true_binary

    return normalized


def build_by_experiment(experiments, cfg):
    by_key = defaultdict(list)

    for exp in experiments:
        key = (exp["model"], exp["lang"], exp["version"])

        for row in exp["predictions"]:
            norm = normalize_prediction_row(row, exp, cfg)
            by_key[key].append(norm)

    total_predictions = sum(len(v) for v in by_key.values())
    rprint(f"\nTotal predicții: {total_predictions}\n", style="bold")

    return by_key


# ──────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────

def compute_latency(rows):
    lats = [r.get("latency_ms", 0) for r in rows if r.get("latency_ms", 0) > 0]

    if not lats:
        return {
            "mean": None,
            "median": None,
            "p95": None,
            "max": None,
        }

    return {
        "mean": float(np.mean(lats)),
        "median": float(np.median(lats)),
        "p95": float(np.percentile(lats, 95)),
        "max": float(np.max(lats)),
    }


def safe_kappa(y_true, y_pred, labels):
    """
    Cohen's Kappa corect:
    - include labels oficiale
    - include __invalid__ dacă există în predicții
    Astfel, predicțiile invalide sunt penalizate, nu ignorate.
    """

    try:
        kappa_labels = list(labels)

        if "__invalid__" in y_pred and "__invalid__" not in kappa_labels:
            kappa_labels.append("__invalid__")

        return cohen_kappa_score(y_true, y_pred, labels=kappa_labels)
    except Exception:
        return None


def get_multiclass_yt_yp(rows, labels, task):
    y_true = []
    y_pred = []
    valid_rows = []

    for r in rows:
        true_label = r.get("true_label_norm")
        pred_label = r.get("predicted_label_norm")

        if true_label is None:
            continue

        # Pentru incongruities, clasificarea multi-clasă se face doar pe exemplele
        # care au într-adevăr o neconcordanță.
        if task == "incongruities" and true_label not in labels:
            continue

        # Pentru intent/final_status păstrăm doar true labels valide.
        if task != "incongruities" and true_label not in labels:
            continue

        y_true.append(true_label)

        if pred_label in labels:
            y_pred.append(pred_label)
        else:
            y_pred.append("__invalid__")

        valid_rows.append(r)

    return y_true, y_pred, valid_rows


def compute_multiclass_metrics(rows, labels, task):
    y_true, y_pred, valid_rows = get_multiclass_yt_yp(rows, labels, task)

    if not y_true:
        return None

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            labels=labels,
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            y_true,
            y_pred,
            labels=labels,
            average="weighted",
            zero_division=0,
        ),
        "kappa": safe_kappa(y_true, y_pred, labels),
        "n": len(y_true),
        "y_true": y_true,
        "y_pred": y_pred,
        "valid_rows": valid_rows,
    }


def get_binary_yt_yp(rows):
    y_true = []
    y_pred = []

    for r in rows:
        true = r.get("true_binary_norm")
        pred = r.get("predicted_binary_norm")

        if true is not None and pred is not None:
            y_true.append(true)
            y_pred.append(pred)

    return y_true, y_pred


def compute_binary_metrics(rows):
    y_true, y_pred = get_binary_yt_yp(rows)

    if not y_true:
        return None

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(
            y_true,
            y_pred,
            average="binary",
            pos_label=True,
            zero_division=0,
        ),
        "kappa": cohen_kappa_score(y_true, y_pred),
        "n": len(y_true),
        "y_true": y_true,
        "y_pred": y_pred,
    }


# ──────────────────────────────────────────────
# Record collection
# ──────────────────────────────────────────────

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
            "lang": lang,
            "version": version,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "kappa": metrics["kappa"],
            "latency_ms": lat["mean"],
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
            "latency_ms": lat["mean"],
            "n": metrics["n"],
            "y_true": metrics["y_true"],
            "y_pred": metrics["y_pred"],
        })

    return records


def choose_best_per_model(records, metric_name):
    best_by_model = {}

    for rec in records:
        model = rec["model"]

        if model not in best_by_model:
            best_by_model[model] = rec
            continue

        current = best_by_model[model]

        if (
            rec[metric_name] > current[metric_name]
            or (
                rec[metric_name] == current[metric_name]
                and rec["accuracy"] > current["accuracy"]
            )
        ):
            best_by_model[model] = rec

    return sorted(best_by_model.values(), key=lambda r: r[metric_name], reverse=True)


# ──────────────────────────────────────────────
# Tables
# ──────────────────────────────────────────────

def print_t1_overview(by_key, cfg):
    section(f"T1 — Overview experimente: {cfg['display_name']}")

    rows = []

    for key in sorted(by_key):
        model, lang, version = key
        preds = by_key[key]
        total = len(preds)
        fails = sum(1 for r in preds if r.get("parse_failed", False))

        rows.append([
            MODEL_DISPLAY.get(model, model),
            MODEL_TYPE.get(model, "?"),
            MODEL_PARAMS.get(model, "?"),
            lang.upper(),
            version,
            total,
            fails,
            pct(fails / total) if total else "—",
        ])

    rprint_table(
        rows,
        headers=["Model", "Tip", "Param", "Lang", "Ver", "Total", "Fails", "Fail%"],
        title="T1 — Toate experimentele încărcate",
    )


def print_all_multiclass(records, cfg):
    section(f"T-ALL — Toate experimentele: {cfg['display_name']}")

    rows = []

    for rec in records:
        rows.append([
            rec["model_display"],
            rec["type"],
            rec["lang"].upper(),
            rec["version"],
            pct(rec["accuracy"]),
            fmt(rec["macro_f1"]),
            fmt(rec["weighted_f1"]),
            fmt(rec["kappa"]),
            ms(rec["latency_ms"]),
            rec["n"],
        ])

    rows.sort(key=lambda r: float(r[5]) if r[5] != "—" else -1, reverse=True)

    rprint_table(
        rows,
        headers=["Model", "Tip", "Lang", "Ver", "Acc", "M-F1", "W-F1", "κ", "Lat", "N"],
        title="T-ALL — toate versiunile",
    )


def print_best_multiclass(records, cfg):
    section(f"T-BEST — Best prompt per model: {cfg['display_name']}")

    best = choose_best_per_model(records, metric_name="macro_f1")

    rows = []

    for rec in best:
        rows.append([
            rec["model_display"],
            rec["type"],
            rec["lang"].upper(),
            rec["version"],
            pct(rec["accuracy"]),
            fmt(rec["macro_f1"]),
            fmt(rec["weighted_f1"]),
            fmt(rec["kappa"]),
            ms(rec["latency_ms"]),
            rec["n"],
        ])

    rprint_table(
        rows,
        headers=["Model", "Tip", "Lang", "Ver", "Acc", "M-F1", "W-F1", "κ", "Lat", "N"],
        title="T-BEST — best prompt per model",
    )

    return best


def print_binary_tables(by_key, cfg):
    if "binary" not in cfg:
        return None

    section(f"T-BIN-ALL — Detecție binară: {cfg['display_name']}")

    records = collect_binary_records(by_key)

    rows = []

    for rec in records:
        rows.append([
            rec["model_display"],
            rec["type"],
            rec["lang"].upper(),
            rec["version"],
            pct(rec["accuracy"]),
            fmt(rec["f1"]),
            fmt(rec["kappa"]),
            ms(rec["latency_ms"]),
            rec["n"],
        ])

    rows.sort(key=lambda r: float(r[5]) if r[5] != "—" else -1, reverse=True)

    rprint_table(
        rows,
        headers=["Model", "Tip", "Lang", "Ver", "B-Acc", "B-F1", "κ", "Lat", "N"],
        title="T-BIN-ALL — toate versiunile",
    )

    section(f"T-BIN-BEST — Best prompt per model: {cfg['display_name']}")

    best = choose_best_per_model(records, metric_name="f1")

    rows = []

    for rec in best:
        rows.append([
            rec["model_display"],
            rec["type"],
            rec["lang"].upper(),
            rec["version"],
            pct(rec["accuracy"]),
            fmt(rec["f1"]),
            fmt(rec["kappa"]),
            ms(rec["latency_ms"]),
            rec["n"],
        ])

    rprint_table(
        rows,
        headers=["Model", "Tip", "Lang", "Ver", "B-Acc", "B-F1", "κ", "Lat", "N"],
        title="T-BIN-BEST — best prompt per model",
    )

    return best


def print_per_class_for_best(best_records, labels, cfg):
    section(f"T-PER-CLASS — Metrici per clasă pentru best prompt: {cfg['display_name']}")

    rows = []

    for rec in best_records:
        precision, recall, f1, support = precision_recall_fscore_support(
            rec["y_true"],
            rec["y_pred"],
            labels=labels,
            zero_division=0,
        )

        for label, p, r, f, s in zip(labels, precision, recall, f1, support):
            rows.append([
                rec["model_display"],
                f"{rec['lang'].upper()} {rec['version']}",
                label,
                fmt(p),
                fmt(r),
                fmt(f),
                int(s),
            ])

    rprint_table(
        rows,
        headers=["Model", "Cfg", "Clasă", "Precision", "Recall", "F1", "Support"],
        title="T-PER-CLASS — best prompt per model",
    )


def print_error_analysis_for_best(best_records, cfg):
    section(f"T-ERR-BEST — Analiza erorilor pentru best prompt: {cfg['display_name']}")

    for rec in best_records:
        errors = defaultdict(lambda: defaultdict(int))

        for true, pred in zip(rec["y_true"], rec["y_pred"]):
            if true != pred:
                errors[true][pred] += 1

        title = f"{rec['model_display']} [{rec['lang'].upper()} {rec['version']}]"

        if not errors:
            rprint(f"\n▶ {title} — 0 erori", style="green")
            continue

        rprint(f"\n▶ {title}", style="bold")

        rows = []

        for true_label, preds in errors.items():
            for pred_label, cnt in preds.items():
                rows.append([true_label, pred_label, cnt])

        rows.sort(key=lambda x: -x[2])

        rprint_table(
            rows[:15],
            headers=["Etichetă reală", "Prezis ca", "# erori"],
            title=title,
        )


def print_latex_table_for_best(best_records, cfg, task):
    section(f"LaTeX — Tabel pentru lucrare: {cfg['display_name']}")

    label = {
        "intent": "tab:intent_best",
        "final_status": "tab:final_status_best",
        "incongruities": "tab:inc_multiclass_best",
    }[task]

    caption = {
        "intent": "Extragerea intenției (best prompt per model)",
        "final_status": "Clasificarea statusului final (best prompt per model)",
        "incongruities": "Clasificarea tipului de neconcordanță (best prompt per model)",
    }[task]

    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(fr"\caption{{{caption}}}")
    lines.append(fr"\label{{{label}}}")
    lines.append(r"\footnotesize")
    lines.append(r"\begin{tabular}{llcccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Model} & \textbf{Cfg} & \textbf{Acc} & \textbf{M-F1} & $\boldsymbol{\kappa}$ & \textbf{Lat} \\")
    lines.append(r"\midrule")

    for rec in best_records:
        model = rec["model_display"]
        cfg_txt = f"{rec['lang'].upper()} {rec['version']}"
        acc = pct_latex(rec["accuracy"])
        mf1 = fmt(rec["macro_f1"])
        kappa = fmt(rec["kappa"])
        lat = ms(rec["latency_ms"])

        lines.append(f"{model} & {cfg_txt} & {acc} & {mf1} & {kappa} & {lat} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    rprint("\n".join(lines))


def debug_first_prediction(experiments):
    section("DEBUG — Structura primului experiment")

    exp = experiments[0]
    rprint(f"Fișier: {exp['path'].name}")
    rprint(f"Model: {exp['model']}, Lang: {exp['lang']}, Version: {exp['version']}")
    rprint(f"Număr predicții: {len(exp['predictions'])}")

    if exp["predictions"]:
        first = exp["predictions"][0]
        rprint("\nChei disponibile în prima predicție:")
        rprint(list(first.keys()))

        rprint("\nPrima predicție, parțial:")
        rprint(json.dumps(first, ensure_ascii=False, indent=2)[:2500])


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def run(task, filter_model=None, filter_lang=None, filter_version=None, debug=False):
    global _report_lines
    _report_lines = []

    cfg = TASK_CONFIG[task]

    rprint(f"\nTask: {cfg['display_name']}", style="bold cyan")
    rprint(f"Director: {cfg['output_dir']}")
    rprint(f"Raport: {cfg['report_path']}\n")

    labels = load_labels(cfg)
    rprint(f"Labels: {labels}\n", style="bold")

    experiments = load_all_experiments(
        cfg,
        filter_model=filter_model,
        filter_lang=filter_lang,
        filter_version=filter_version,
    )

    if debug:
        debug_first_prediction(experiments)

    by_key = build_by_experiment(experiments, cfg)

    print_t1_overview(by_key, cfg)

    if task == "incongruities":
        print_binary_tables(by_key, cfg)

    records = collect_multiclass_records(by_key, labels, task)

    print_all_multiclass(records, cfg)
    best_records = print_best_multiclass(records, cfg)

    if best_records:
        print_per_class_for_best(best_records, labels, cfg)
        print_error_analysis_for_best(best_records, cfg)
        print_latex_table_for_best(best_records, cfg, task)

    rprint(f"\n{'═' * 90}", style="cyan")
    rprint(f"Evaluare completă: {cfg['display_name']}", style="bold green")
    rprint(f"{'═' * 90}", style="cyan")

    save_report(cfg["report_path"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluare best prompt pentru toate taskurile")

    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["intent", "final_status", "incongruities"],
    )

    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--lang", type=str, default=None)
    parser.add_argument("--version", type=str, default=None)
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    run(
        task=args.task,
        filter_model=args.model,
        filter_lang=args.lang,
        filter_version=args.version,
        debug=args.debug,
    )