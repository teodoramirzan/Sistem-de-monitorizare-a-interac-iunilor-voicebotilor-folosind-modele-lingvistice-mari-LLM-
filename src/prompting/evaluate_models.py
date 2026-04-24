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

Tabele generate:
─────────────────────────────────────────────────────────────────────────────
T1. Experimente încărcate: ce fișiere s-au găsit și ce versiune s-a folosit
T2. Acuratețe per model și limbă (RO vs EN) — delta între limbi
T3. Latență detaliată: mean, median, p95, p99, min, max
T4. API vs. Local — comparație directă pe toate metricile
T5. Diferențe între modele: unde sunt de acord, unde diferă
T6. Analiza erorilor: ce intenții greșesc cel mai mult fiecare model
T7. Impact limbă prompt: RO vs EN per model
T8. Fiabilitate output: parse fail rate și distribuție confidence
─────────────────────────────────────────────────────────────────────────────
"""

import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from itertools import combinations

from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from tabulate import tabulate

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

OUTPUT_DIR = Path(
    r"C:\Users\Matebook 14s\Documents"
    r"\Sistem-de-monitorizare-a-interac-iunilor-voicebotilor-folosind-modele-lingvistice-mari-LLM-"
    r"\outputs"
)

with open("configs/intent_definitions.json", encoding="utf-8") as f:
    _defs = json.load(f)
LABELS = [l["name"] for l in _defs["labels"]]

# Clasificare fixă — știm dinainte care sunt API și care sunt locale
MODEL_TYPE = {
    "openai_o3":       "API",
    "gemini_2.5_pro":  "API",
    "aya_expanse_8b":  "Local",
    "rollama2_7b":     "Local",
    "robert_encoder":  "Local",
}

MODEL_PARAMS = {
    "openai_o3":       "~200B+",
    "gemini_2.5_pro":  "~1T+",
    "aya_expanse_8b":  "8B",
    "rollama2_7b":     "7B",
    "robert_encoder":  "~125M",
}

SECTION = "═" * 90

def section(title):
    print(f"\n{SECTION}")
    print(f"  {title}")
    print(SECTION)

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

def load_experiments(
    output_dir: Path,
    filter_model: str = None,
    filter_lang: str = None,
    filter_version: str = None,
) -> list[dict]:
    """
    Citește toate fișierele exp_*.json din output_dir.

    Dacă pentru același (model, lang) există mai multe versiuni de prompt,
    folosește implicit cea mai recentă (după timestamp din metadata).
    Poți forța o versiune specifică cu filter_version.

    Returnează lista plată de predicții, ca și cum ar veni dintr-un singur fișier.
    """
    exp_files = sorted(output_dir.glob("exp_*.json"))
    if not exp_files:
        raise FileNotFoundError(
            f"Nu s-au găsit fișiere exp_*.json în {output_dir}\n"
            f"Rulează mai întâi intent_experiments.ipynb."
        )

    # Grupează fișierele pe (model, lang) → {version: (timestamp, path)}
    grouped = defaultdict(dict)
    for path in exp_files:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        meta = data.get("experiment", {})

        model   = meta.get("model", "unknown")
        lang    = meta.get("lang", "unknown")
        version = meta.get("prompt_version", "v1")
        ts      = meta.get("timestamp", "00000000_000000")

        # Filtre opționale
        if filter_model   and model   != filter_model:   continue
        if filter_lang    and lang    != filter_lang:     continue
        if filter_version and version != filter_version:  continue

        grouped[(model, lang)][version] = (ts, path, data)

    if not grouped:
        raise ValueError("Niciun experiment nu corespunde filtrelor specificate.")

    # Pentru fiecare (model, lang) alege versiunea corectă
    selected = {}   # {(model, lang): (version, data)}
    for (model, lang), versions in grouped.items():
        if filter_version:
            chosen_version = filter_version
        else:
            # Cea mai recentă după timestamp
            chosen_version = max(versions, key=lambda v: versions[v][0])

        ts, path, data = versions[chosen_version]
        selected[(model, lang)] = (chosen_version, path, data)

    # Afișează ce s-a încărcat
    print(f"\n  Experimente încărcate ({len(selected)}):")
    for (model, lang), (version, path, _) in sorted(selected.items()):
        print(f"    {model:<22} [{lang}]  version={version}  ← {path.name}")

    # Aplatizează predicțiile
    all_predictions = []
    for (model, lang), (version, path, data) in selected.items():
        for pred in data.get("predictions", []):
            # Asigură câmpurile necesare
            pred.setdefault("model_name",     model)
            pred.setdefault("prompt_lang",    lang)
            pred.setdefault("prompt_version", version)
            all_predictions.append(pred)

    print(f"\n  Total predicții: {len(all_predictions)}\n")
    return all_predictions


def organize(predictions: list[dict]):
    """
    Returnează două structuri:
    - by_key:  {(model, lang): [prediction_dict, ...]}
    - by_conv: {conv_id: {(model, lang): predicted_intent}}
    """
    by_key  = defaultdict(list)
    by_conv = defaultdict(dict)

    for p in predictions:
        key = (p["model_name"], p["prompt_lang"])
        by_key[key].append(p)
        by_conv[p["conversation_id"]][key] = p["predicted_intent"]

    return by_key, by_conv


def get_yt_yp(by_key, key):
    """Returnează (y_true, y_pred) filtrând parse failures."""
    rows = by_key.get(key, [])
    y_true = [r["dataset_label"] for r in rows if r["predicted_intent"] is not None]
    y_pred = [r["predicted_intent"] for r in rows if r["predicted_intent"] is not None]
    return y_true, y_pred


def compute_metrics(y_true, y_pred):
    if not y_true:
        return None
    return {
        "accuracy":     accuracy_score(y_true, y_pred),
        "macro_f1":     f1_score(y_true, y_pred, average="macro",    labels=LABELS, zero_division=0),
        "weighted_f1":  f1_score(y_true, y_pred, average="weighted", labels=LABELS, zero_division=0),
        "kappa":        cohen_kappa_score(y_true, y_pred, labels=LABELS),
        "n":            len(y_true),
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
# T1. Clasificare modele: API vs. Local
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
            rows.append([m, mtype, params, l, version, total, fails,
                         pct(fails / total) if total else "—"])

    print(tabulate(rows,
                   headers=["Model", "Tip", "Parametri", "Lang", "Prompt ver.",
                             "Total pred.", "Parse fail", "Fail%"],
                   tablefmt="rounded_outline"))


# ──────────────────────────────────────────────
# T2. Acuratețe per model și limbă
# ──────────────────────────────────────────────

def print_t2_accuracy(by_key):
    section("T2 — Acuratețe per model și limbă (+ delta RO vs EN)")

    models = sorted({k[0] for k in by_key})
    rows   = []

    for m in models:
        mtype = MODEL_TYPE.get(m, "?")
        m_ro = compute_metrics(*get_yt_yp(by_key, (m, "ro")))
        m_en = compute_metrics(*get_yt_yp(by_key, (m, "en")))

        acc_ro  = m_ro["accuracy"]  if m_ro else None
        acc_en  = m_en["accuracy"]  if m_en else None
        f1_ro   = m_ro["macro_f1"]  if m_ro else None
        f1_en   = m_en["macro_f1"]  if m_en else None
        kap_ro  = m_ro["kappa"]     if m_ro else None
        kap_en  = m_en["kappa"]     if m_en else None

        rows.append([
            m, mtype,
            pct(acc_ro), pct(acc_en), delta_str(acc_ro, acc_en),
            pct(f1_ro),  pct(f1_en),  delta_str(f1_ro,  f1_en),
            fmt(kap_ro), fmt(kap_en),
        ])

    headers = [
        "Model", "Tip",
        "Acc RO", "Acc EN", "ΔAcc (RO-EN)",
        "F1 RO",  "F1 EN",  "ΔF1 (RO-EN)",
        "κ RO",   "κ EN",
    ]
    print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))
    print("\n  * Delta pozitiv = promptul RO performează mai bine decât EN pe acel model.")


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
            model, lang, mtype,
            ms(lat["mean"]), ms(lat["median"]),
            ms(lat["p95"]),  ms(lat["p99"]),
            ms(lat["min"]),  ms(lat["max"]),
            ms(lat["std"]),
        ])

    # Sortează după mean latency
    rows.sort(key=lambda r: float(r[3].replace("ms", "")))

    headers = ["Model", "Lang", "Tip", "Mean", "Median", "p95", "p99", "Min", "Max", "Std"]
    print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))


# ──────────────────────────────────────────────
# T4. API vs. Local — comparație directă
# ──────────────────────────────────────────────

def print_t4_api_vs_local(by_key):
    section("T4 — API vs. Local: comparație agregată")

    groups = {"API": [], "Local": []}
    for key in by_key:
        model = key[0]
        mtype = MODEL_TYPE.get(model, "?")
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

        acc    = accuracy_score(y_true, y_pred)        if y_true else None
        mf1    = f1_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0) if y_true else None
        kappa  = cohen_kappa_score(y_true, y_pred, labels=LABELS) if y_true else None

        summary.append([
            gname,
            len(keys),
            pct(acc), pct(mf1), fmt(kappa),
            ms(np.mean(lats))   if lats else "—",
            ms(np.median(lats)) if lats else "—",
            ms(np.percentile(lats, 95)) if lats else "—",
            pct(fails / len(all_preds)) if all_preds else "—",
            len(y_true),
        ])

    headers = [
        "Grup", "Nr. modele×limbi",
        "Accuracy", "Macro F1", "Cohen κ",
        "Lat mean", "Lat median", "Lat p95",
        "Parse fail%", "N predicții valide"
    ]
    print(tabulate(summary, headers=headers, tablefmt="rounded_outline"))

    # Per-model breakdown în grupuri
    print("\n  ── Detaliu per model ──")
    detail_rows = []
    for gname, keys in groups.items():
        for key in sorted(keys):
            model, lang = key
            y_true, y_pred = get_yt_yp(by_key, key)
            lats = [r["latency_ms"] for r in by_key[key] if r["latency_ms"] > 0]
            fails = sum(1 for r in by_key[key] if r["parse_failed"])
            total = len(by_key[key])

            acc  = accuracy_score(y_true, y_pred) if y_true else None
            mf1  = f1_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0) if y_true else None

            detail_rows.append([
                gname, model, lang,
                pct(acc), pct(mf1),
                ms(np.mean(lats)) if lats else "—",
                ms(np.percentile(lats, 95)) if lats else "—",
                pct(fails / total) if total else "—",
            ])

    detail_rows.sort(key=lambda r: (r[0], float(r[3].replace("%","")) if r[3] != "—" else 0), reverse=True)
    print(tabulate(detail_rows,
                   headers=["Grup", "Model", "Lang", "Accuracy", "Macro F1",
                             "Lat mean", "Lat p95", "Parse fail%"],
                   tablefmt="simple"))


# ──────────────────────────────────────────────
# T5. Diferențe între modele: acord și dezacord
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
                (r["dataset_label"] for r in by_key[ka] if r["conversation_id"] == cid),
                None
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

            if ca and cb:     both_correct += 1
            elif not ca and not cb: both_wrong += 1
            elif ca:          a_right += 1
            else:             b_right += 1

        if total == 0:
            continue

        agree = (both_correct + both_wrong) / total

        rows.append([
            f"{model_a}/{lang_a}",
            f"{model_b}/{lang_b}",
            pct(agree),
            both_correct, both_wrong,
            a_right, b_right,
            total,
        ])

    rows.sort(key=lambda r: float(r[2].replace("%","")), reverse=True)

    headers = [
        "Model A", "Model B",
        "% Acord",
        "Ambii corecți", "Ambii greșiți",
        "Doar A corect", "Doar B corect",
        "Total",
    ]
    print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))
    print("\n  * '% Acord' = conversații unde ambii dau același răspuns (corect SAU greșit).")


# ──────────────────────────────────────────────
# T6. Analiza erorilor: intenții greșite frecvent
# ──────────────────────────────────────────────

def print_t6_error_analysis(by_key):
    section("T6 — Analiza erorilor: intenții greșite cel mai des per model")

    for key in sorted(by_key):
        model, lang = key
        mtype = MODEL_TYPE.get(model, "?")

        errors = defaultdict(lambda: defaultdict(int))  # {true_label: {pred_label: count}}
        for r in by_key[key]:
            if r["predicted_intent"] is None:
                continue
            if r["predicted_intent"] != r["dataset_label"]:
                errors[r["dataset_label"]][r["predicted_intent"]] += 1

        if not errors:
            print(f"\n  ▶  {model} [{lang.upper()}] ({mtype}) — 0 erori")
            continue

        print(f"\n  ▶  {model} [{lang.upper()}] ({mtype})")

        rows = []
        for true_label in sorted(errors):
            for pred_label, count in sorted(errors[true_label].items(), key=lambda x: -x[1]):
                rows.append([true_label, pred_label, count])

        rows.sort(key=lambda r: -r[2])
        print(tabulate(rows[:15],  # top 15 erori
                       headers=["Intenție reală", "Prezis ca", "# greșeli"],
                       tablefmt="simple"))


# ──────────────────────────────────────────────
# T7. Impact limbă prompt: RO vs EN per model
# ──────────────────────────────────────────────

def print_t7_language_impact(by_key):
    section("T7 — Impactul limbii promptului: RO vs EN per model")

    models = sorted({k[0] for k in by_key})
    rows   = []

    for model in models:
        mtype  = MODEL_TYPE.get(model, "?")
        ro_key = (model, "ro")
        en_key = (model, "en")

        if ro_key not in by_key or en_key not in by_key:
            continue

        m_ro = compute_metrics(*get_yt_yp(by_key, ro_key))
        m_en = compute_metrics(*get_yt_yp(by_key, en_key))

        lat_ro = compute_latency(by_key[ro_key])
        lat_en = compute_latency(by_key[en_key])

        if not m_ro or not m_en:
            continue

        delta_acc = m_ro["accuracy"]   - m_en["accuracy"]
        delta_f1  = m_ro["macro_f1"]   - m_en["macro_f1"]
        delta_lat = (lat_ro.get("mean", 0) - lat_en.get("mean", 0)) if lat_ro and lat_en else None

        winner = "RO" if delta_acc > 0.01 else "EN" if delta_acc < -0.01 else "="

        rows.append([
            model, mtype,
            pct(m_ro["accuracy"]), pct(m_en["accuracy"]),
            delta_str(m_ro["accuracy"], m_en["accuracy"]),
            pct(m_ro["macro_f1"]),  pct(m_en["macro_f1"]),
            delta_str(m_ro["macro_f1"], m_en["macro_f1"]),
            delta_str(lat_ro.get("mean"), lat_en.get("mean"), as_pct=False) if delta_lat is not None else "—",
            winner,
        ])

    headers = [
        "Model", "Tip",
        "Acc RO", "Acc EN", "ΔAcc",
        "F1 RO",  "F1 EN",  "ΔF1",
        "ΔLat (RO-EN)", "Câștigător",
    ]
    print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))
    print("\n  * ΔLat pozitiv = promptul RO e mai lent. '=' = diferență < 1%.")


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
            model, lang, mtype, total,
            fails,  pct(fails / total),
            high,   pct(high / total),
            medium, pct(medium / total),
            low,    pct(low / total),
            none_,
        ])

    headers = [
        "Model", "Lang", "Tip", "Total",
        "Fails", "Fail%",
        "High", "High%",
        "Med",  "Med%",
        "Low",  "Low%",
        "No conf",
    ]
    print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))
    print("\n  * 'No conf' = parse failed sau modelul nu a returnat câmpul confidence.")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def run(filter_model=None, filter_lang=None, filter_version=None):
    print(f"\nCaut experimente în:\n  {OUTPUT_DIR}")

    predictions = load_experiments(
        output_dir=OUTPUT_DIR,
        filter_model=filter_model,
        filter_lang=filter_lang,
        filter_version=filter_version,
    )
    by_key, by_conv = organize(predictions)

    print_t1_model_overview(by_key)
    print_t2_accuracy(by_key)
    print_t3_latency(by_key)
    print_t4_api_vs_local(by_key)
    print_t5_differences(by_key, by_conv)
    print_t6_error_analysis(by_key)
    print_t7_language_impact(by_key)
    print_t8_reliability(by_key)

    print(f"\n{SECTION}")
    print("  Evaluare completă.")
    print(SECTION)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluare detaliată LLM-uri pe experimentele din outputs/"
    )
    parser.add_argument("--model",   type=str, default=None,
                        help="Filtrează după model (ex: openai_o3)")
    parser.add_argument("--lang",    type=str, default=None,
                        help="Filtrează după limbă: ro | en")
    parser.add_argument("--version", type=str, default=None,
                        help="Filtrează după versiunea promptului (ex: v1). "
                             "Implicit: cea mai recentă per (model, lang).")
    args = parser.parse_args()

    run(
        filter_model=args.model,
        filter_lang=args.lang,
        filter_version=args.version,
    )
