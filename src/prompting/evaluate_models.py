# src/prompting/evaluate_models.py

"""
Evaluare detaliată LLM-uri: Acuratețe, Latență, Diferențe, API vs. Local.

Citește toate fișierele exp_*.json din outputs/ generate de intent_experiments.ipynb.
Dacă există mai multe versiuni de prompt pentru același model+limbă (ex: v1, v2),
folosește implicit cea mai recentă. Poți filtra explicit cu --version.

Utilizare:
    python evaluate_models.py                        # toate experimentele
    python evaluate_models.py --version v1           # doar versiunea v1
    python evaluate_models.py --model openai_o3      # doar un model
    python evaluate_models.py --model openai_o3 --version v2 --lang ro
    python evaluate_models.py --save-latex           # salvează și tabele LaTeX

Output:
    - Terminal: tabele colorate cu rich
    - evaluation_report_intent.txt: raport complet plain text
    - results/latex_tables/*.tex: tabele în format LaTeX (cu --save-latex)
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

# Rich pentru terminal colorat
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich import box
    from rich.columns import Columns
    from rich.style import Style
    HAS_RICH = True
    console = Console()
except ImportError:
    HAS_RICH = False
    console = None

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

OUTPUT_DIR = Path(
    r"C:\Users\Matebook 14s\Documents"
    r"\Sistem-de-monitorizare-a-interac-iunilor-voicebotilor-folosind-modele-lingvistice-mari-LLM-"
    r"\outputs"
)

LATEX_DIR = Path(
    r"C:\Users\Matebook 14s\Documents"
    r"\Sistem-de-monitorizare-a-interac-iunilor-voicebotilor-folosind-modele-lingvistice-mari-LLM-"
    r"\results\latex_tables"
)

REPORT_PATH = Path(
    r"C:\Users\Matebook 14s\Documents"
    r"\Sistem-de-monitorizare-a-interac-iunilor-voicebotilor-folosind-modele-lingvistice-mari-LLM-"
    r"\evaluation_report_intent.txt"
)

with open("configs/intent_definitions.json", encoding="utf-8") as f:
    _defs = json.load(f)
LABELS = [l["name"] for l in _defs["labels"]]

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

# Culori rich pentru metrici
def acc_color(v):
    if v is None: return "dim"
    if v >= 0.95: return "bold green"
    if v >= 0.85: return "green"
    if v >= 0.70: return "yellow"
    if v >= 0.50: return "orange1"
    return "red"

def delta_color(v):
    if v is None: return "dim"
    if v > 0.01:  return "green"
    if v < -0.01: return "red"
    return "dim"

# ──────────────────────────────────────────────
# Output helpers — scrie atât în terminal cât și în fișier
# ──────────────────────────────────────────────

_report_lines = []

def rprint(text="", style=None):
    """Printează în terminal (cu rich dacă disponibil) și salvează în buffer."""
    _report_lines.append(str(text) + "\n")
    if HAS_RICH and console:
        if style:
            console.print(text, style=style)
        else:
            console.print(text)
    else:
        print(text)

def rprint_table(rows, headers, title=None, tablefmt="rounded_outline"):
    """Printează tabel în terminal și salvează plain text în buffer."""
    plain = tabulate(rows, headers=headers, tablefmt=tablefmt)
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
            table.add_column(h, no_wrap=False)
        for row in rows:
            table.add_row(*[str(c) for c in row])
        console.print(table)
    else:
        if title:
            print(f"\n  {title}")
        print(plain)

def section(title):
    _report_lines.append(f"\n{'═'*90}\n  {title}\n{'═'*90}\n")
    if HAS_RICH and console:
        console.print()
        console.print(Panel(f"[bold white]{title}[/bold white]",
                            border_style="cyan", expand=True))
    else:
        print(f"\n{'═'*90}")
        print(f"  {title}")
        print('═'*90)

def save_report():
    """Salvează raportul complet în fișier txt."""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.writelines(_report_lines)
    rprint(f"\n✓ Raport salvat în: {REPORT_PATH}", style="bold green")

# ──────────────────────────────────────────────
# Helpers metrici
# ──────────────────────────────────────────────

def pct(v):
    return f"{v*100:.1f}%" if v is not None else "—"

def fmt(v, d=3):
    return f"{v:.{d}f}" if v is not None else "—"

def ms(v):
    return f"{v:.0f}ms" if v is not None else "—"

def delta_str(a, b, as_pct=True):
    if a is None or b is None:
        return "—"
    d = a - b
    sign = "+" if d >= 0 else ""
    return f"{sign}{d*100:.1f}%" if as_pct else f"{sign}{d:.0f}ms"

# ──────────────────────────────────────────────
# Load & organize
# ──────────────────────────────────────────────

def load_experiments(output_dir, filter_model=None, filter_lang=None, filter_version=None):
    exp_files = sorted(output_dir.glob("exp_*.json"))
    if not exp_files:
        raise FileNotFoundError(f"Nu s-au găsit fișiere exp_*.json în {output_dir}")

    grouped = defaultdict(dict)
    for path in exp_files:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        meta = data.get("experiment", {})
        model   = meta.get("model", "unknown")
        lang    = meta.get("lang", "unknown")
        version = meta.get("prompt_version", "v1")
        ts      = meta.get("timestamp", "00000000_000000")

        if filter_model   and model   != filter_model:   continue
        if filter_lang    and lang    != filter_lang:     continue
        if filter_version and version != filter_version:  continue

        grouped[(model, lang)][version] = (ts, path, data)

    if not grouped:
        raise ValueError("Niciun experiment nu corespunde filtrelor specificate.")

    selected = {}
    for (model, lang), versions in grouped.items():
        chosen_version = filter_version if filter_version else max(versions, key=lambda v: versions[v][0])
        ts, path, data = versions[chosen_version]
        selected[(model, lang)] = (chosen_version, path, data)

    rprint(f"\n  Experimente încărcate ({len(selected)}):", style="bold")
    for (model, lang), (version, path, _) in sorted(selected.items()):
        rprint(f"    {MODEL_DISPLAY.get(model, model):<25} [{lang.upper()}]  {version}  ← {path.name}")

    all_predictions = []
    for (model, lang), (version, path, data) in selected.items():
        for pred in data.get("predictions", []):
            pred.setdefault("model_name",     model)
            pred.setdefault("prompt_lang",    lang)
            pred.setdefault("prompt_version", version)
            all_predictions.append(pred)

    rprint(f"\n  Total predicții: {len(all_predictions)}\n")
    return all_predictions


def organize(predictions):
    by_key  = defaultdict(list)
    by_conv = defaultdict(dict)
    for p in predictions:
        key = (p["model_name"], p["prompt_lang"])
        by_key[key].append(p)
        by_conv[p["conversation_id"]][key] = p["predicted_intent"]
    return by_key, by_conv


def get_yt_yp(by_key, key):
    rows   = by_key.get(key, [])
    y_true = [r["dataset_label"]    for r in rows if r["predicted_intent"] is not None]
    y_pred = [r["predicted_intent"] for r in rows if r["predicted_intent"] is not None]
    return y_true, y_pred


def compute_metrics(y_true, y_pred):
    if not y_true:
        return None
    return {
        "accuracy":    accuracy_score(y_true, y_pred),
        "macro_f1":    f1_score(y_true, y_pred, average="macro",    labels=LABELS, zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", labels=LABELS, zero_division=0),
        "kappa":       cohen_kappa_score(y_true, y_pred, labels=LABELS),
        "n":           len(y_true),
    }


def compute_latency(rows):
    lats = [r["latency_ms"] for r in rows if r["latency_ms"] > 0]
    if not lats:
        return {}
    return {
        "mean":   np.mean(lats),
        "median": np.median(lats),
        "p95":    np.percentile(lats, 95),
        "p99":    np.percentile(lats, 99),
        "min":    np.min(lats),
        "max":    np.max(lats),
        "std":    np.std(lats),
    }

# ──────────────────────────────────────────────
# T1. Overview modele
# ──────────────────────────────────────────────

def print_t1_model_overview(by_key):
    section("T1 — Experimente încărcate: API vs. Local")
    models = sorted({k[0] for k in by_key})
    rows = []
    for m in models:
        mtype  = MODEL_TYPE.get(m, "?")
        params = MODEL_PARAMS.get(m, "?")
        langs  = sorted({k[1] for k in by_key if k[0] == m})
        for l in langs:
            preds   = by_key[(m, l)]
            total   = len(preds)
            fails   = sum(1 for r in preds if r["parse_failed"])
            version = preds[0].get("prompt_version", "?") if preds else "?"
            rows.append([
                MODEL_DISPLAY.get(m, m), mtype, params, l.upper(),
                version, total, fails,
                pct(fails / total) if total else "—"
            ])
    rprint_table(rows,
                 headers=["Model", "Tip", "Parametri", "Lang", "Prompt ver.",
                           "Total pred.", "Parse fail", "Fail%"],
                 title="T1 — Overview modele")

# ──────────────────────────────────────────────
# T2. Acuratețe per model și limbă
# ──────────────────────────────────────────────

def print_t2_accuracy(by_key):
    section("T2 — Acuratețe per model și limbă (+ delta RO vs EN)")
    models = sorted({k[0] for k in by_key})
    rows   = []

    for m in models:
        m_ro = compute_metrics(*get_yt_yp(by_key, (m, "ro")))
        m_en = compute_metrics(*get_yt_yp(by_key, (m, "en")))

        acc_ro = m_ro["accuracy"]  if m_ro else None
        acc_en = m_en["accuracy"]  if m_en else None
        f1_ro  = m_ro["macro_f1"]  if m_ro else None
        f1_en  = m_en["macro_f1"]  if m_en else None
        kap_ro = m_ro["kappa"]     if m_ro else None
        kap_en = m_en["kappa"]     if m_en else None
        mtype  = MODEL_TYPE.get(m, "?")

        rows.append([
            MODEL_DISPLAY.get(m, m), mtype,
            pct(acc_ro), pct(acc_en), delta_str(acc_ro, acc_en),
            pct(f1_ro),  pct(f1_en),  delta_str(f1_ro,  f1_en),
            fmt(kap_ro), fmt(kap_en),
        ])

    # Sortează după acc_ro descrescător
    rows.sort(key=lambda r: float(r[2].replace("%","")) if r[2] != "—" else 0, reverse=True)

    rprint_table(rows,
                 headers=["Model", "Tip",
                           "Acc RO", "Acc EN", "ΔAcc (RO-EN)",
                           "F1 RO",  "F1 EN",  "ΔF1 (RO-EN)",
                           "κ RO",   "κ EN"],
                 title="T2 — Acuratețe per model și limbă")
    rprint("  * Delta pozitiv = promptul RO performează mai bine decât EN.")

# ──────────────────────────────────────────────
# T2b. Acuratețe per versiune de prompt
# ──────────────────────────────────────────────

def print_t2b_accuracy_by_version(output_dir, filter_model=None):
    """Tabel suplimentar: evoluția acurateței pe versiuni v1→v4 per model."""
    section("T2b — Evoluția acurateței pe versiuni de prompt (v1→v4)")

    exp_files = sorted(output_dir.glob("exp_*.json"))
    data_by_model_lang_version = defaultdict(dict)

    for path in exp_files:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        meta    = data.get("experiment", {})
        model   = meta.get("model", "unknown")
        lang    = meta.get("lang", "unknown")
        version = meta.get("prompt_version", "v1")
        if filter_model and model != filter_model:
            continue

        preds  = data.get("predictions", [])
        y_true = [r["dataset_label"]    for r in preds if r.get("predicted_intent")]
        y_pred = [r["predicted_intent"] for r in preds if r.get("predicted_intent")]

        if y_true:
            acc = accuracy_score(y_true, y_pred)
            f1  = f1_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0)
            data_by_model_lang_version[(model, lang)][version] = (acc, f1)

    versions = ["v1", "v2", "v3", "v4"]
    rows = []
    for (model, lang) in sorted(data_by_model_lang_version.keys()):
        vdata = data_by_model_lang_version[(model, lang)]
        row = [MODEL_DISPLAY.get(model, model), lang.upper()]
        for v in versions:
            if v in vdata:
                acc, f1 = vdata[v]
                row.append(f"{acc*100:.1f}%")
            else:
                row.append("—")
        rows.append(row)

    headers = ["Model", "Lang"] + [f"Acc {v}" for v in versions]
    rprint_table(rows, headers=headers, title="T2b — Evoluție pe versiuni")
    rprint("  * Arată cum evoluează acuratețea de la v1 (zero-shot simplu) la v4 (few-shot).")

# ──────────────────────────────────────────────
# T3. Latență detaliată
# ──────────────────────────────────────────────

def print_t3_latency(by_key):
    section("T3 — Latență detaliată per model și limbă (ms)")
    rows = []
    for key in sorted(by_key):
        model, lang = key
        mtype = MODEL_TYPE.get(model, "?")
        lat   = compute_latency(by_key[key])
        if not lat:
            continue
        rows.append([
            MODEL_DISPLAY.get(model, model), lang.upper(), mtype,
            ms(lat["mean"]), ms(lat["median"]),
            ms(lat["p95"]),  ms(lat["p99"]),
            ms(lat["min"]),  ms(lat["max"]),
        ])
    rows.sort(key=lambda r: float(r[3].replace("ms", "")))
    rprint_table(rows,
                 headers=["Model", "Lang", "Tip", "Mean", "Median", "p95", "p99", "Min", "Max"],
                 title="T3 — Latență detaliată")

# ──────────────────────────────────────────────
# T4. API vs. Local
# ──────────────────────────────────────────────

def print_t4_api_vs_local(by_key):
    section("T4 — API vs. Local: comparație agregată")
    groups = {"API": [], "Local": []}
    for key in by_key:
        mtype = MODEL_TYPE.get(key[0], "?")
        if mtype in groups:
            groups[mtype].append(key)

    summary = []
    for gname, keys in groups.items():
        all_preds = []
        for k in keys:
            all_preds.extend(by_key[k])
        y_true = [r["dataset_label"]    for r in all_preds if r["predicted_intent"] is not None]
        y_pred = [r["predicted_intent"] for r in all_preds if r["predicted_intent"] is not None]
        lats   = [r["latency_ms"]       for r in all_preds if r["latency_ms"] > 0]
        fails  = sum(1 for r in all_preds if r["parse_failed"])

        acc   = accuracy_score(y_true, y_pred) if y_true else None
        mf1   = f1_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0) if y_true else None
        kappa = cohen_kappa_score(y_true, y_pred, labels=LABELS) if y_true else None

        summary.append([
            gname, len(keys),
            pct(acc), pct(mf1), fmt(kappa),
            ms(np.mean(lats)) if lats else "—",
            ms(np.percentile(lats, 95)) if lats else "—",
            pct(fails / len(all_preds)) if all_preds else "—",
        ])

    rprint_table(summary,
                 headers=["Grup", "Nr. chei", "Accuracy", "Macro F1", "Cohen κ",
                           "Lat mean", "Lat p95", "Parse fail%"],
                 title="T4 — API vs. Local agregat")

# ──────────────────────────────────────────────
# T5. Diferențe între modele
# ──────────────────────────────────────────────

def print_t5_differences(by_key, by_conv):
    section("T5 — Diferențe între modele: acord și dezacord per pereche")
    keys     = sorted(by_key.keys())
    conv_ids = sorted(by_conv.keys())
    rows = []

    for ka, kb in combinations(keys, 2):
        model_a, lang_a = ka
        model_b, lang_b = kb
        both_correct = both_wrong = a_right = b_right = total = 0

        for cid in conv_ids:
            true_label = next(
                (r["dataset_label"] for r in by_key[ka] if r["conversation_id"] == cid), None
            )
            if true_label is None:
                continue
            pa = by_conv[cid].get(ka)
            pb = by_conv[cid].get(kb)
            if pa is None or pb is None:
                continue
            total += 1
            ca = pa == true_label
            cb = pb == true_label
            if ca and cb:           both_correct += 1
            elif not ca and not cb: both_wrong   += 1
            elif ca:                a_right      += 1
            else:                   b_right      += 1

        if total == 0:
            continue
        agree = (both_correct + both_wrong) / total
        rows.append([
            f"{MODEL_DISPLAY.get(model_a, model_a)}/{lang_a.upper()}",
            f"{MODEL_DISPLAY.get(model_b, model_b)}/{lang_b.upper()}",
            pct(agree),
            both_correct, both_wrong, a_right, b_right, total,
        ])

    rows.sort(key=lambda r: float(r[2].replace("%","")), reverse=True)
    rprint_table(rows[:20],
                 headers=["Model A", "Model B", "% Acord",
                           "Ambii corecți", "Ambii greșiți",
                           "Doar A corect", "Doar B corect", "Total"],
                 title="T5 — Acord între modele (top 20 perechi)")
    rprint("  * '% Acord' = conversații unde ambii dau același răspuns (corect SAU greșit).")

# ──────────────────────────────────────────────
# T6. Analiza erorilor
# ──────────────────────────────────────────────

def print_t6_error_analysis(by_key):
    section("T6 — Analiza erorilor: intenții greșite cel mai des per model")
    for key in sorted(by_key):
        model, lang = key
        mtype  = MODEL_TYPE.get(model, "?")
        errors = defaultdict(lambda: defaultdict(int))
        for r in by_key[key]:
            if r["predicted_intent"] and r["predicted_intent"] != r["dataset_label"]:
                errors[r["dataset_label"]][r["predicted_intent"]] += 1

        if not errors:
            rprint(f"\n  ▶  {MODEL_DISPLAY.get(model, model)} [{lang.upper()}] — 0 erori", style="green")
            continue

        rprint(f"\n  ▶  {MODEL_DISPLAY.get(model, model)} [{lang.upper()}] ({mtype})", style="bold")
        rows = [
            [true, pred, cnt]
            for true, preds in errors.items()
            for pred, cnt in sorted(preds.items(), key=lambda x: -x[1])
        ]
        rows.sort(key=lambda r: -r[2])
        rprint_table(rows[:10],
                     headers=["Intenție reală", "Prezis ca", "# greșeli"])

# ──────────────────────────────────────────────
# T7. Impact limbă
# ──────────────────────────────────────────────

def print_t7_language_impact(by_key):
    section("T7 — Impactul limbii promptului: RO vs EN per model")
    models = sorted({k[0] for k in by_key})
    rows   = []

    for model in models:
        ro_key = (model, "ro")
        en_key = (model, "en")
        if ro_key not in by_key or en_key not in by_key:
            continue

        m_ro   = compute_metrics(*get_yt_yp(by_key, ro_key))
        m_en   = compute_metrics(*get_yt_yp(by_key, en_key))
        lat_ro = compute_latency(by_key[ro_key])
        lat_en = compute_latency(by_key[en_key])

        if not m_ro or not m_en:
            continue

        delta_acc = m_ro["accuracy"] - m_en["accuracy"]
        winner    = "RO 🏆" if delta_acc > 0.01 else "EN 🏆" if delta_acc < -0.01 else "="
        delta_lat = lat_ro.get("mean", 0) - lat_en.get("mean", 0) if lat_ro and lat_en else None

        rows.append([
            MODEL_DISPLAY.get(model, model),
            MODEL_TYPE.get(model, "?"),
            pct(m_ro["accuracy"]), pct(m_en["accuracy"]),
            delta_str(m_ro["accuracy"], m_en["accuracy"]),
            pct(m_ro["macro_f1"]),  pct(m_en["macro_f1"]),
            delta_str(m_ro["macro_f1"], m_en["macro_f1"]),
            delta_str(lat_ro.get("mean"), lat_en.get("mean"), as_pct=False) if delta_lat is not None else "—",
            winner,
        ])

    rprint_table(rows,
                 headers=["Model", "Tip",
                           "Acc RO", "Acc EN", "ΔAcc",
                           "F1 RO",  "F1 EN",  "ΔF1",
                           "ΔLat (RO-EN)", "Câștigător"],
                 title="T7 — Impact limbă prompt")
    rprint("  * ΔLat pozitiv = promptul RO e mai lent. '=' = diferență < 1%.")

# ──────────────────────────────────────────────
# T8. Fiabilitate output
# ──────────────────────────────────────────────

def print_t8_reliability(by_key):
    section("T8 — Fiabilitate output: parse failures și distribuție confidence")
    rows = []
    for key in sorted(by_key):
        model, lang = key
        mtype  = MODEL_TYPE.get(model, "?")
        preds  = by_key[key]
        total  = len(preds)
        fails  = sum(1 for r in preds if r["parse_failed"])
        high   = sum(1 for r in preds if r.get("confidence") == "high")
        medium = sum(1 for r in preds if r.get("confidence") == "medium")
        low    = sum(1 for r in preds if r.get("confidence") == "low")
        none_  = sum(1 for r in preds if r.get("confidence") is None)

        rows.append([
            MODEL_DISPLAY.get(model, model), lang.upper(), mtype, total,
            fails,  pct(fails / total),
            high,   pct(high / total),
            medium, pct(medium / total),
            low,    pct(low / total),
            none_,
        ])

    rprint_table(rows,
                 headers=["Model", "Lang", "Tip", "Total",
                           "Fails", "Fail%",
                           "High", "High%", "Med", "Med%",
                           "Low", "Low%", "No conf"],
                 title="T8 — Fiabilitate output")

# ──────────────────────────────────────────────
# Export LaTeX
# ──────────────────────────────────────────────

def save_latex_tables(by_key, output_dir_path, latex_dir):
    """Salvează tabelele principale în format LaTeX booktabs."""
    latex_dir.mkdir(parents=True, exist_ok=True)

    def esc(s):
        return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")

    # ── T2 Acuratețe ──────────────────────────────────────────────────────
    models = sorted({k[0] for k in by_key})
    rows_t2 = []
    for m in models:
        m_ro = compute_metrics(*get_yt_yp(by_key, (m, "ro")))
        m_en = compute_metrics(*get_yt_yp(by_key, (m, "en")))
        rows_t2.append([
            esc(MODEL_DISPLAY.get(m, m)),
            MODEL_TYPE.get(m, "?"),
            pct(m_ro["accuracy"] if m_ro else None),
            pct(m_en["accuracy"] if m_en else None),
            esc(delta_str(m_ro["accuracy"] if m_ro else None, m_en["accuracy"] if m_en else None)),
            pct(m_ro["macro_f1"] if m_ro else None),
            pct(m_en["macro_f1"] if m_en else None),
            fmt(m_ro["kappa"] if m_ro else None),
            fmt(m_en["kappa"] if m_en else None),
        ])
    rows_t2.sort(key=lambda r: float(r[2].replace(r"\%","").replace("%","")) if r[2] != "—" else 0, reverse=True)

    headers_t2 = ["Model", "Tip", "Acc RO", "Acc EN", r"$\Delta$Acc",
                  "F1 RO", "F1 EN", r"$\kappa$ RO", r"$\kappa$ EN"]
    latex_t2 = tabulate(rows_t2, headers=headers_t2, tablefmt="latex_booktabs")
    with open(latex_dir / "t2_accuracy.tex", "w", encoding="utf-8") as f:
        f.write(latex_t2)

    # ── T2b Evoluție pe versiuni ───────────────────────────────────────────
    exp_files = sorted(output_dir_path.glob("exp_*.json"))
    data_versions = defaultdict(dict)
    for path in exp_files:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        meta    = data.get("experiment", {})
        model   = meta.get("model", "unknown")
        lang    = meta.get("lang", "unknown")
        version = meta.get("prompt_version", "v1")
        preds   = data.get("predictions", [])
        y_true  = [r["dataset_label"]    for r in preds if r.get("predicted_intent")]
        y_pred  = [r["predicted_intent"] for r in preds if r.get("predicted_intent")]
        if y_true:
            data_versions[(model, lang)][version] = accuracy_score(y_true, y_pred)

    versions = ["v1", "v2", "v3", "v4"]
    rows_t2b = []
    for (model, lang) in sorted(data_versions.keys()):
        vdata = data_versions[(model, lang)]
        row = [esc(MODEL_DISPLAY.get(model, model)), lang.upper()]
        for v in versions:
            row.append(pct(vdata.get(v)))
        rows_t2b.append(row)

    headers_t2b = ["Model", "Lang"] + [f"Acc {v}" for v in versions]
    latex_t2b = tabulate(rows_t2b, headers=headers_t2b, tablefmt="latex_booktabs")
    with open(latex_dir / "t2b_accuracy_by_version.tex", "w", encoding="utf-8") as f:
        f.write(latex_t2b)

    # ── T3 Latență ────────────────────────────────────────────────────────
    rows_t3 = []
    for key in sorted(by_key):
        model, lang = key
        lat = compute_latency(by_key[key])
        if not lat:
            continue
        rows_t3.append([
            esc(MODEL_DISPLAY.get(model, model)), lang.upper(),
            MODEL_TYPE.get(model, "?"),
            ms(lat["mean"]), ms(lat["median"]),
            ms(lat["p95"]), ms(lat["max"]),
        ])
    rows_t3.sort(key=lambda r: float(r[3].replace("ms", "")))
    latex_t3 = tabulate(rows_t3,
                        headers=["Model", "Lang", "Tip", "Mean", "Median", "p95", "Max"],
                        tablefmt="latex_booktabs")
    with open(latex_dir / "t3_latency.tex", "w", encoding="utf-8") as f:
        f.write(latex_t3)

    # ── T7 Impact limbă ───────────────────────────────────────────────────
    rows_t7 = []
    for model in sorted({k[0] for k in by_key}):
        m_ro = compute_metrics(*get_yt_yp(by_key, (model, "ro")))
        m_en = compute_metrics(*get_yt_yp(by_key, (model, "en")))
        if not m_ro or not m_en:
            continue
        delta_acc = m_ro["accuracy"] - m_en["accuracy"]
        winner = "RO" if delta_acc > 0.01 else "EN" if delta_acc < -0.01 else "="
        rows_t7.append([
            esc(MODEL_DISPLAY.get(model, model)),
            MODEL_TYPE.get(model, "?"),
            pct(m_ro["accuracy"]), pct(m_en["accuracy"]),
            esc(delta_str(m_ro["accuracy"], m_en["accuracy"])),
            pct(m_ro["macro_f1"]),  pct(m_en["macro_f1"]),
            winner,
        ])
    latex_t7 = tabulate(rows_t7,
                        headers=["Model", "Tip", "Acc RO", "Acc EN", r"$\Delta$Acc",
                                  "F1 RO", "F1 EN", "Câștigător"],
                        tablefmt="latex_booktabs")
    with open(latex_dir / "t7_language_impact.tex", "w", encoding="utf-8") as f:
        f.write(latex_t7)

    rprint(f"\n✓ Tabele LaTeX salvate în: {latex_dir}", style="bold green")
    rprint(f"  t2_accuracy.tex          — Acuratețe per model și limbă")
    rprint(f"  t2b_accuracy_by_version.tex — Evoluție pe versiuni v1→v4")
    rprint(f"  t3_latency.tex           — Latență detaliată")
    rprint(f"  t7_language_impact.tex   — Impact limbă prompt")
    rprint()
    rprint("  Folosire în LaTeX:")
    rprint(r"  \input{results/latex_tables/t2_accuracy.tex}")

# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def run(filter_model=None, filter_lang=None, filter_version=None, save_latex=False):
    rprint(f"\nCaut experimente în:\n  {OUTPUT_DIR}", style="bold")

    predictions = load_experiments(
        output_dir=OUTPUT_DIR,
        filter_model=filter_model,
        filter_lang=filter_lang,
        filter_version=filter_version,
    )
    by_key, by_conv = organize(predictions)

    print_t1_model_overview(by_key)
    print_t2_accuracy(by_key)
    print_t2b_accuracy_by_version(OUTPUT_DIR, filter_model=filter_model)
    print_t3_latency(by_key)
    print_t4_api_vs_local(by_key)
    print_t5_differences(by_key, by_conv)
    print_t6_error_analysis(by_key)
    print_t7_language_impact(by_key)
    print_t8_reliability(by_key)

    if save_latex:
        save_latex_tables(by_key, OUTPUT_DIR, LATEX_DIR)

    rprint(f"\n{'═'*90}", style="cyan")
    rprint("  Evaluare completă.", style="bold green")
    rprint(f"{'═'*90}", style="cyan")

    save_report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluare detaliată LLM-uri pe experimentele din outputs/"
    )
    parser.add_argument("--model",       type=str,  default=None)
    parser.add_argument("--lang",        type=str,  default=None)
    parser.add_argument("--version",     type=str,  default=None)
    parser.add_argument("--save-latex",  action="store_true",
                        help="Salvează tabele în format LaTeX în results/latex_tables/")
    args = parser.parse_args()

    run(
        filter_model=args.model,
        filter_lang=args.lang,
        filter_version=args.version,
        save_latex=args.save_latex,
    )