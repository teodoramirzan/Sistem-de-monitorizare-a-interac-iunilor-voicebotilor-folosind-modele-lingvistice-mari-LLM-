# src/prompting/evaluate_all_tasks.py

"""
Evaluare detaliată LLM-uri pe toate cele 3 taskuri.

Utilizare:
    python evaluate_all_tasks.py --task intent
    python evaluate_all_tasks.py --task final_status
    python evaluate_all_tasks.py --task incongruities
    python evaluate_all_tasks.py --task final_status --version v4 --save-latex
    python evaluate_all_tasks.py --task incongruities --model openai_o3

Parametrizat pe baza evaluate_models.py original (intent).
"""

import json
import argparse
import numpy as np
import sys
from pathlib import Path
from collections import defaultdict
from itertools import combinations

from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
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
# Config per task
# ──────────────────────────────────────────────

BASE_DIR = Path(
    r"C:\Users\Matebook 14s\Documents"
    r"\Sistem-de-monitorizare-a-interac-iunilor-voicebotilor-folosind-modele-lingvistice-mari-LLM-"
)

TASK_CONFIG = {
    "intent": {
        "output_dir": BASE_DIR / "outputs" / "intent",
        "report_path": BASE_DIR / "evaluation_report_intent.txt",
        "labels_file": "configs/intent_definitions.json",
        "labels_key": lambda defs: [l["name"] for l in defs["labels"]],
        "pred_field": "predicted_intent",
        "label_field": "dataset_label",
        "pred_key": "predictions",
        "file_pattern": "exp_*.json",
        "display_name": "Extragerea intenției",
    },
    "final_status": {
        "output_dir": BASE_DIR / "outputs_final_status",
        "report_path": BASE_DIR / "evaluation_report_final_status.txt",
        "labels_file": "configs/final_status_definitions.json",
        "labels_key": lambda defs: [l["name"] for l in defs["labels"]],
        "pred_field": "predicted_status",
        "label_field": "dataset_status",
        "pred_key": "results",
        "file_pattern": "exp_*.json",
        "display_name": "Clasificarea statusului final",
    },
    "incongruities": {
        "output_dir": BASE_DIR / "outputs_incongruities",
        "report_path": BASE_DIR / "evaluation_report_incongruities.txt",
        "labels_file": "configs/incongruities_definitions.json",
        "labels_key": lambda defs: [l["name"] for l in defs["labels"]],
        "pred_field": "predicted_type",        # ajustează dacă diferă
        "label_field": "dataset_label",        # ajustează dacă diferă
        "pred_key": "predictions",
        "file_pattern": "*.json",
        "display_name": "Detecția neconcordanțelor",
        # Metrici suplimentare binare
        "binary_fields": {
            "pred_binary": "predicted_has_incongruity",
            "label_binary": "dataset_has_incongruity",  # ajustează
        },
    },
}

LATEX_DIR = BASE_DIR / "results" / "latex_tables"

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

MODEL_PARAMS = {
    "openai_o3":        "~200B+",
    "gemini_2.5_flash": "~unknown",
    "aya_expanse_8b":   "8B",
    "rollama2_7b":      "7B",
    "roberta_encoder":  "~560M",
    "robert_encoder":   "~125M",
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
        table = Table(title=title, box=box.ROUNDED, header_style="bold cyan",
                      border_style="blue", show_lines=True)
        for h in headers:
            table.add_column(h, no_wrap=False)
        for row in rows:
            table.add_row(*[str(c) for c in row])
        console.print(table)
    else:
        if title: print(f"\n  {title}")
        print(plain)

def section(title):
    _report_lines.append(f"\n{'═'*90}\n  {title}\n{'═'*90}\n")
    if HAS_RICH and console:
        console.print()
        console.print(Panel(f"[bold white]{title}[/bold white]",
                            border_style="cyan", expand=True))
    else:
        print(f"\n{'═'*90}\n  {title}\n{'═'*90}")

def save_report(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(_report_lines)
    rprint(f"\n✓ Raport salvat: {path}", style="bold green")

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def pct(v): return f"{v*100:.1f}%" if v is not None else "—"
def fmt(v, d=3): return f"{v:.{d}f}" if v is not None else "—"
def ms(v): return f"{v:.0f}ms" if v is not None else "—"

def delta_str(a, b, as_pct=True):
    if a is None or b is None: return "—"
    d = a - b
    sign = "+" if d >= 0 else ""
    return f"{sign}{d*100:.1f}%" if as_pct else f"{sign}{d:.0f}ms"

# ──────────────────────────────────────────────
# Load & organize (parametrizat per task)
# ──────────────────────────────────────────────

def load_experiments(cfg, filter_model=None, filter_lang=None, filter_version=None):
    output_dir = cfg["output_dir"]
    pred_key = cfg["pred_key"]
    pred_field = cfg["pred_field"]
    label_field = cfg["label_field"]

    exp_files = sorted(output_dir.glob(cfg["file_pattern"]))
    if not exp_files:
        raise FileNotFoundError(f"Nu s-au găsit fișiere în {output_dir}")

    grouped = defaultdict(dict)
    for path in exp_files:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        # Detectează metadata — suportă ambele structuri
        meta = data.get("experiment", {})
        if not meta:
            model = data.get("model", "unknown")
            lang = data.get("language", "unknown")
            version = data.get("prompt_version", "v1")
            ts = data.get("timestamp", "00000000")
        else:
            model = meta.get("model", "unknown")
            lang = meta.get("lang", meta.get("language", "unknown"))
            version = meta.get("prompt_version", "v1")
            ts = meta.get("timestamp", "00000000")

        # Extrage model din experiment_name dacă model=unknown
        if model == "unknown":
            exp_name = data.get("experiment_name", path.stem)
            # pattern: prefix_model__lang__version
            parts = exp_name.split("__")
            if len(parts) >= 3:
                model = parts[0].replace("exp_", "").replace("fst_", "").replace("inc_", "")
                lang = parts[1]
                version = parts[2]

        if filter_model and model != filter_model: continue
        if filter_lang and lang != filter_lang: continue
        if filter_version and version != filter_version: continue

        grouped[(model, lang)][version] = (ts, path, data)

    if not grouped:
        raise ValueError("Niciun experiment nu corespunde filtrelor.")

    selected = {}
    for (model, lang), versions in grouped.items():
        chosen = filter_version if filter_version else max(versions, key=lambda v: versions[v][0])
        ts, path, data = versions[chosen]
        selected[(model, lang)] = (chosen, path, data)

    rprint(f"\n  Experimente încărcate ({len(selected)}):", style="bold")
    for (model, lang), (version, path, _) in sorted(selected.items()):
        rprint(f"    {MODEL_DISPLAY.get(model, model):<25} [{lang.upper()}]  {version}  ← {path.name}")

    all_predictions = []
    for (model, lang), (version, path, data) in selected.items():
        for pred in data.get(pred_key, []):
            pred.setdefault("model_name", model)
            pred.setdefault("prompt_lang", lang)
            pred.setdefault("prompt_version", version)
            # Normalizează field names
            if pred_field not in pred and "predicted_intent" in pred:
                pred[pred_field] = pred["predicted_intent"]
            if label_field not in pred and "dataset_label" in pred:
                pred[label_field] = pred["dataset_label"]
            all_predictions.append(pred)

    rprint(f"\n  Total predicții: {len(all_predictions)}\n")
    return all_predictions


def organize(predictions, cfg):
    pred_field = cfg["pred_field"]
    label_field = cfg["label_field"]
    by_key = defaultdict(list)
    by_conv = defaultdict(dict)
    for p in predictions:
        key = (p["model_name"], p["prompt_lang"])
        by_key[key].append(p)
        by_conv[p["conversation_id"]][key] = p.get(pred_field)
    return by_key, by_conv


def get_yt_yp(by_key, key, cfg):
    pred_field = cfg["pred_field"]
    label_field = cfg["label_field"]
    rows = by_key.get(key, [])
    y_true = [r[label_field] for r in rows if r.get(pred_field) is not None]
    y_pred = [r[pred_field]  for r in rows if r.get(pred_field) is not None]
    return y_true, y_pred


def compute_metrics(y_true, y_pred, labels=None):
    if not y_true: return None
    return {
        "accuracy":    accuracy_score(y_true, y_pred),
        "macro_f1":    f1_score(y_true, y_pred, average="macro",    labels=labels, zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0),
        "kappa":       cohen_kappa_score(y_true, y_pred),
        "n":           len(y_true),
    }


def compute_latency(rows):
    lats = [r.get("latency_ms", 0) for r in rows if r.get("latency_ms", 0) > 0]
    if not lats: return {}
    return {
        "mean": np.mean(lats), "median": np.median(lats),
        "p95": np.percentile(lats, 95), "p99": np.percentile(lats, 99),
        "min": np.min(lats), "max": np.max(lats),
    }

# ──────────────────────────────────────────────
# Tabele (adaptate per task)
# ──────────────────────────────────────────────

def print_t1(by_key, cfg):
    section(f"T1 — Overview modele: {cfg['display_name']}")
    pred_field = cfg["pred_field"]
    models = sorted({k[0] for k in by_key})
    rows = []
    for m in models:
        langs = sorted({k[1] for k in by_key if k[0] == m})
        for l in langs:
            preds = by_key[(m, l)]
            total = len(preds)
            fails = sum(1 for r in preds if r.get("parse_failed", False))
            version = preds[0].get("prompt_version", "?") if preds else "?"
            rows.append([
                MODEL_DISPLAY.get(m, m), MODEL_TYPE.get(m, "?"),
                MODEL_PARAMS.get(m, "?"), l.upper(), version,
                total, fails, pct(fails/total) if total else "—"
            ])
    rprint_table(rows,
                 headers=["Model","Tip","Param","Lang","Ver","Total","Fails","Fail%"],
                 title=f"T1 — {cfg['display_name']}")


def print_t2(by_key, cfg, labels):
    section(f"T2 — Acuratețe per model: {cfg['display_name']}")
    models = sorted({k[0] for k in by_key})
    rows = []
    for m in models:
        m_ro = compute_metrics(*get_yt_yp(by_key, (m,"ro"), cfg), labels=labels)
        m_en = compute_metrics(*get_yt_yp(by_key, (m,"en"), cfg), labels=labels)
        rows.append([
            MODEL_DISPLAY.get(m, m), MODEL_TYPE.get(m, "?"),
            pct(m_ro["accuracy"] if m_ro else None),
            pct(m_en["accuracy"] if m_en else None),
            delta_str(m_ro["accuracy"] if m_ro else None, m_en["accuracy"] if m_en else None),
            pct(m_ro["macro_f1"] if m_ro else None),
            pct(m_en["macro_f1"] if m_en else None),
            fmt(m_ro["kappa"] if m_ro else None),
            fmt(m_en["kappa"] if m_en else None),
        ])
    rows.sort(key=lambda r: float(r[2].replace("%","")) if r[2]!="—" else 0, reverse=True)
    rprint_table(rows,
                 headers=["Model","Tip","Acc RO","Acc EN","ΔAcc","F1 RO","F1 EN","κ RO","κ EN"],
                 title=f"T2 — Acuratețe: {cfg['display_name']}")


def print_t2b(cfg, filter_model=None):
    section(f"T2b — Evoluție v1→v4: {cfg['display_name']}")
    output_dir = cfg["output_dir"]
    pred_field = cfg["pred_field"]
    label_field = cfg["label_field"]
    pred_key = cfg["pred_key"]

    exp_files = sorted(output_dir.glob(cfg["file_pattern"]))
    data_versions = defaultdict(dict)

    for path in exp_files:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        # Extrage meta
        meta = data.get("experiment", {})
        exp_name = data.get("experiment_name", meta.get("name", path.stem))
        parts = exp_name.split("__")
        if len(parts) >= 3:
            model = parts[0].replace("exp_","").replace("fst_","").replace("inc_","")
            lang = parts[1]
            version = parts[2]
        else:
            model = meta.get("model", "unknown")
            lang = meta.get("lang", "unknown")
            version = meta.get("prompt_version", "v1")

        if filter_model and model != filter_model: continue

        preds = data.get(pred_key, [])
        y_true = [r[label_field] for r in preds if r.get(pred_field) is not None]
        y_pred = [r[pred_field]  for r in preds if r.get(pred_field) is not None]

        if y_true:
            data_versions[(model, lang)][version] = accuracy_score(y_true, y_pred)

    versions = ["v1", "v2", "v3", "v4"]
    rows = []
    for (model, lang) in sorted(data_versions.keys()):
        vdata = data_versions[(model, lang)]
        row = [MODEL_DISPLAY.get(model, model), lang.upper()]
        for v in versions:
            row.append(pct(vdata.get(v)))
        rows.append(row)

    rprint_table(rows,
                 headers=["Model","Lang"] + [f"Acc {v}" for v in versions],
                 title=f"T2b — Evoluție: {cfg['display_name']}")


def print_t3(by_key, cfg):
    section(f"T3 — Latență: {cfg['display_name']}")
    rows = []
    for key in sorted(by_key):
        model, lang = key
        lat = compute_latency(by_key[key])
        if not lat: continue
        rows.append([
            MODEL_DISPLAY.get(model, model), lang.upper(),
            MODEL_TYPE.get(model, "?"),
            ms(lat["mean"]), ms(lat["median"]),
            ms(lat["p95"]), ms(lat["max"]),
        ])
    rows.sort(key=lambda r: float(r[3].replace("ms","")))
    rprint_table(rows,
                 headers=["Model","Lang","Tip","Mean","Median","p95","Max"],
                 title=f"T3 — Latență: {cfg['display_name']}")


def print_t6(by_key, cfg):
    section(f"T6 — Analiza erorilor: {cfg['display_name']}")
    pred_field = cfg["pred_field"]
    label_field = cfg["label_field"]

    for key in sorted(by_key):
        model, lang = key
        errors = defaultdict(lambda: defaultdict(int))
        for r in by_key[key]:
            pred = r.get(pred_field)
            true = r.get(label_field)
            if pred and pred != true:
                errors[true][pred] += 1

        if not errors:
            rprint(f"\n  ▶  {MODEL_DISPLAY.get(model, model)} [{lang.upper()}] — 0 erori", style="green")
            continue

        rprint(f"\n  ▶  {MODEL_DISPLAY.get(model, model)} [{lang.upper()}]", style="bold")
        rows = [
            [true, pred, cnt]
            for true, preds in errors.items()
            for pred, cnt in sorted(preds.items(), key=lambda x: -x[1])
        ]
        rows.sort(key=lambda r: -r[2])
        rprint_table(rows[:15], headers=["Etichetă reală", "Prezis ca", "# greșeli"])


def print_t7(by_key, cfg, labels):
    section(f"T7 — Impact limbă: {cfg['display_name']}")
    models = sorted({k[0] for k in by_key})
    rows = []
    for model in models:
        m_ro = compute_metrics(*get_yt_yp(by_key, (model,"ro"), cfg), labels=labels)
        m_en = compute_metrics(*get_yt_yp(by_key, (model,"en"), cfg), labels=labels)
        if not m_ro or not m_en: continue
        d = m_ro["accuracy"] - m_en["accuracy"]
        w = "RO" if d > 0.01 else "EN" if d < -0.01 else "="
        lat_ro = compute_latency(by_key.get((model,"ro"),[]))
        lat_en = compute_latency(by_key.get((model,"en"),[]))
        rows.append([
            MODEL_DISPLAY.get(model, model),
            pct(m_ro["accuracy"]), pct(m_en["accuracy"]),
            delta_str(m_ro["accuracy"], m_en["accuracy"]),
            pct(m_ro["macro_f1"]), pct(m_en["macro_f1"]),
            delta_str(lat_ro.get("mean"), lat_en.get("mean"), as_pct=False) if lat_ro and lat_en else "—",
            w,
        ])
    rprint_table(rows,
                 headers=["Model","Acc RO","Acc EN","ΔAcc","F1 RO","F1 EN","ΔLat","Winner"],
                 title=f"T7 — Impact limbă: {cfg['display_name']}")


def print_t8(by_key, cfg):
    section(f"T8 — Fiabilitate output: {cfg['display_name']}")
    rows = []
    for key in sorted(by_key):
        model, lang = key
        preds = by_key[key]
        total = len(preds)
        fails = sum(1 for r in preds if r.get("parse_failed", False))
        high = sum(1 for r in preds if r.get("confidence") == "high")
        med = sum(1 for r in preds if r.get("confidence") == "medium")
        low = sum(1 for r in preds if r.get("confidence") == "low")
        rows.append([
            MODEL_DISPLAY.get(model, model), lang.upper(),
            total, fails, pct(fails/total),
            high, pct(high/total), med, pct(med/total), low, pct(low/total),
        ])
    rprint_table(rows,
                 headers=["Model","Lang","Total","Fails","Fail%",
                          "High","High%","Med","Med%","Low","Low%"],
                 title=f"T8 — Fiabilitate: {cfg['display_name']}")

# ──────────────────────────────────────────────
# Metrici binare (doar pentru incongruities)
# ──────────────────────────────────────────────

def print_binary_metrics(by_key, cfg):
    bf = cfg.get("binary_fields")
    if not bf: return

    section(f"T-BIN — Metrici binare detecție: {cfg['display_name']}")
    pred_b = bf["pred_binary"]
    label_b = bf["label_binary"]

    rows = []
    for key in sorted(by_key):
        model, lang = key
        preds = by_key[key]
        y_true = [r.get(label_b) for r in preds if r.get(pred_b) is not None]
        y_pred = [r.get(pred_b)  for r in preds if r.get(pred_b) is not None]

        if not y_true: continue

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="binary", pos_label=True, zero_division=0)

        rows.append([
            MODEL_DISPLAY.get(model, model), lang.upper(),
            pct(acc), fmt(f1),
        ])

    rows.sort(key=lambda r: float(r[3]) if r[3]!="—" else 0, reverse=True)
    rprint_table(rows,
                 headers=["Model","Lang","B-Acc","B-F1"],
                 title=f"Detecție binară: {cfg['display_name']}")

# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def run(task, filter_model=None, filter_lang=None, filter_version=None, save_latex=False):
    global _report_lines
    _report_lines = []

    cfg = TASK_CONFIG[task]
    rprint(f"\n  Task: {cfg['display_name']}", style="bold cyan")
    rprint(f"  Director: {cfg['output_dir']}\n")

    # Încarcă labels dacă fișierul există
    labels = None
    labels_path = Path(cfg["labels_file"])
    if labels_path.exists():
        with open(labels_path, encoding="utf-8") as f:
            defs = json.load(f)
        labels = cfg["labels_key"](defs)
        rprint(f"  Labels: {len(labels)} clase din {labels_path.name}")
    else:
        rprint(f"  ⚠ {labels_path} nu există — labels=None (sklearn va deduce din date)")

    predictions = load_experiments(cfg, filter_model, filter_lang, filter_version)
    by_key, by_conv = organize(predictions, cfg)

    print_t1(by_key, cfg)
    print_t2(by_key, cfg, labels)
    print_t2b(cfg, filter_model)
    print_t3(by_key, cfg)
    print_t6(by_key, cfg)
    print_t7(by_key, cfg, labels)
    print_t8(by_key, cfg)

    # Metrici binare (doar incongruities)
    if task == "incongruities":
        print_binary_metrics(by_key, cfg)

    rprint(f"\n{'═'*90}", style="cyan")
    rprint(f"  Evaluare completă: {cfg['display_name']}", style="bold green")
    rprint(f"{'═'*90}", style="cyan")

    save_report(cfg["report_path"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluare LLM-uri pe 3 taskuri")
    parser.add_argument("--task", type=str, required=True,
                        choices=["intent", "final_status", "incongruities"])
    parser.add_argument("--model",   type=str, default=None)
    parser.add_argument("--lang",    type=str, default=None)
    parser.add_argument("--version", type=str, default=None)
    parser.add_argument("--save-latex", action="store_true")
    args = parser.parse_args()

    run(task=args.task,
        filter_model=args.model,
        filter_lang=args.lang,
        filter_version=args.version,
        save_latex=args.save_latex)
