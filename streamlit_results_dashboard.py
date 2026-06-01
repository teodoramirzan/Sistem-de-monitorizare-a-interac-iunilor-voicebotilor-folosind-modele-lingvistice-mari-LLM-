from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "evaluation_reports"
OUTPUT_DIRS = {
    "final_status": ROOT / "outputs_final_status",
    "incongruities": ROOT / "outputs_incongruities",
}

TASK_LABELS = {
    "intent": "Intent extraction",
    "final_status": "Status final",
    "incongruities": "Neconcordanțe",
}

BEST_HINTS = {
    "intent": "OpenAI o3 · EN · v4 · Accuracy/F1 98.3%",
    "final_status": "OpenAI o3 · RO · v4 · Accuracy 90.0%, Macro-F1 0.860",
    "incongruities": "Gemini 2.5 Flash · RO · v4 · Binary F1 0.7606 în fișierul local, raportul agregat indică varianta recomandată.",
}


def repair_text(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return value.encode("latin1").decode("utf-8")
        except UnicodeError:
            return value
    return value


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def parse_json_experiment(path: Path, task: str) -> dict[str, Any]:
    data = read_json(path)
    experiment = data.get("experiment") or {}
    metrics = data.get("metrics") or {}
    row = {
        "task": task,
        "model": repair_text(experiment.get("model") or data.get("model") or "necunoscut"),
        "language": repair_text(experiment.get("lang") or data.get("language") or ""),
        "prompt_version": repair_text(experiment.get("prompt_version") or data.get("prompt_version") or ""),
        "dataset_size": experiment.get("n_conversations") or data.get("dataset_size"),
        "timestamp": experiment.get("timestamp") or data.get("timestamp"),
        "source_file": path.name,
    }
    for key, value in metrics.items():
        if isinstance(value, (int, float, str)) or value is None:
            row[key] = value
    predictions = data.get("predictions") or data.get("results") or []
    row["n_predictions"] = len(predictions) if isinstance(predictions, list) else 0
    return row


def load_json_metrics() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for task, folder in OUTPUT_DIRS.items():
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*.json")):
            try:
                rows.append(parse_json_experiment(path, task))
            except Exception as exc:
                rows.append({"task": task, "source_file": path.name, "error": str(exc)})
    return pd.DataFrame(rows)


def parse_intent_report() -> pd.DataFrame:
    report = REPORTS / "eval_intent.txt"
    if not report.exists():
        return pd.DataFrame()
    rows = []
    pattern = re.compile(
        r"│\s*(?P<model>[^│]+?)\s*│\s*(?P<kind>API|Local)\s*│\s*(?P<lang>EN|RO)\s*│\s*(?P<prompt>v[\w_]+)\s*│\s*(?P<acc>[\d.]+)%\s*│\s*(?P<precision>[\d.]+)\s*│\s*(?P<recall>[\d.]+)\s*│\s*(?P<f1>[\d.]+)\s*│\s*(?P<kappa_label>[^│]+?)\s*│\s*(?P<latency>[\d]+)\s*ms\s*│\s*(?P<n>\d+)\s*│"
    )
    for line in report.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.search(line)
        if not match:
            continue
        item = match.groupdict()
        rows.append(
            {
                "task": "intent",
                "model": item["model"].replace("★", "").strip(),
                "language": item["lang"].lower(),
                "prompt_version": item["prompt"].strip(),
                "accuracy": float(item["acc"]) / 100,
                "macro_f1": float(item["f1"]),
                "precision": float(item["precision"]),
                "recall": float(item["recall"]),
                "latency_ms": int(item["latency"]),
                "dataset_size": int(item["n"]),
                "source_file": "evaluation_reports/eval_intent.txt",
            }
        )
    return pd.DataFrame(rows)


def metric_column(task: str, df: pd.DataFrame) -> str:
    candidates = {
        "intent": ["macro_f1", "accuracy"],
        "final_status": ["macro_f1", "accuracy", "weighted_f1"],
        "incongruities": ["binary_f1", "type_macro_f1", "binary_accuracy"],
    }[task]
    for column in candidates:
        if column in df.columns:
            return column
    return "dataset_size"


def format_percent(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number * 100:.1f}%" if number <= 1.5 else f"{number:.1f}"


def load_predictions(task: str, source_file: str) -> pd.DataFrame:
    folder = OUTPUT_DIRS.get(task)
    if not folder:
        return pd.DataFrame()
    path = folder / source_file
    if not path.exists():
        return pd.DataFrame()
    data = read_json(path)
    items = data.get("predictions") or data.get("results") or []
    if not isinstance(items, list):
        return pd.DataFrame()
    rows = []
    for item in items:
        if isinstance(item, dict):
            rows.append({key: repair_text(value) for key, value in item.items() if key != "raw_response"})
    return pd.DataFrame(rows)


st.set_page_config(page_title="Rezultate evaluare LLM", layout="wide")
st.title("Rezultate evaluare LLM")
st.caption("Explorer pentru rapoartele și fișierele JSON existente în repository.")

json_metrics = load_json_metrics()
intent_metrics = parse_intent_report()
metrics = pd.concat([intent_metrics, json_metrics], ignore_index=True, sort=False)

if metrics.empty:
    st.warning("Nu am găsit rezultate de evaluare în repository.")
    st.stop()

with st.sidebar:
    st.header("Filtre")
    task = st.selectbox("Task", ["intent", "final_status", "incongruities"], format_func=lambda value: TASK_LABELS[value])
    task_df = metrics[metrics["task"] == task].copy()
    models = sorted(task_df["model"].dropna().unique().tolist())
    selected_models = st.multiselect("Modele", models, default=models)
    languages = sorted(task_df["language"].dropna().unique().tolist())
    selected_languages = st.multiselect("Limbă", languages, default=languages)
    versions = sorted(task_df["prompt_version"].dropna().unique().tolist())
    selected_versions = st.multiselect("Prompt", versions, default=versions)

filtered = task_df[
    task_df["model"].isin(selected_models)
    & task_df["language"].isin(selected_languages)
    & task_df["prompt_version"].isin(selected_versions)
].copy()

main_metric = metric_column(task, filtered)
best = filtered.sort_values(main_metric, ascending=False).head(1) if main_metric in filtered.columns else pd.DataFrame()

st.subheader(TASK_LABELS[task])
st.info(f"Variantă recomandată pentru demo: {BEST_HINTS[task]}")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Experimente", len(filtered))
col2.metric("Metrică principală", main_metric)
if not best.empty:
    winner = best.iloc[0]
    col3.metric("Cel mai bun scor", format_percent(winner.get(main_metric)))
    col4.metric("Model", f"{winner.get('model')} · {winner.get('language')} {winner.get('prompt_version')}")
else:
    col3.metric("Cel mai bun scor", "n/a")
    col4.metric("Model", "n/a")

if not filtered.empty and main_metric in filtered.columns:
    chart_df = filtered[["model", "language", "prompt_version", main_metric]].dropna().copy()
    chart_df["config"] = chart_df["model"].astype(str) + " · " + chart_df["language"].astype(str) + " · " + chart_df["prompt_version"].astype(str)
    chart_df = chart_df.sort_values(main_metric, ascending=False).head(20)
    st.bar_chart(chart_df.set_index("config")[[main_metric]])

st.dataframe(
    filtered.sort_values(main_metric, ascending=False) if main_metric in filtered.columns else filtered,
    use_container_width=True,
    hide_index=True,
)

if task in OUTPUT_DIRS and not filtered.empty:
    st.divider()
    st.subheader("Predicții dintr-un experiment")
    source_options = filtered["source_file"].dropna().unique().tolist()
    selected_source = st.selectbox("Fișier JSON", source_options)
    predictions = load_predictions(task, selected_source)
    if predictions.empty:
        st.caption("Fișierul selectat nu conține predicții afișabile.")
    else:
        only_errors = st.checkbox("Arată doar predicțiile greșite / parse failed", value=False)
        view = predictions.copy()
        if only_errors:
            mismatch_columns = [
                ("dataset_status", "predicted_status"),
                ("dataset_has_incongruity", "predicted_has_incongruity"),
                ("dataset_incongruity_type", "predicted_incongruity_type"),
            ]
            mask = pd.Series(False, index=view.index)
            if "parse_failed" in view:
                mask = mask | view["parse_failed"].fillna(False).astype(bool)
            for left, right in mismatch_columns:
                if left in view and right in view:
                    mask = mask | (view[left].astype(str) != view[right].astype(str))
            view = view[mask]
        st.dataframe(view, use_container_width=True, hide_index=True)

st.divider()
st.subheader("Rapoarte text")
report_map = {
    "intent": "eval_intent.txt",
    "final_status": "eval_final_status.txt",
    "incongruities": "eval_incongruities.txt",
}
report_path = REPORTS / report_map[task]
if report_path.exists():
    with st.expander(f"Vezi {report_map[task]}", expanded=False):
        st.text(report_path.read_text(encoding="utf-8", errors="replace")[:20000])
else:
    st.caption("Nu există raport text pentru taskul selectat.")
