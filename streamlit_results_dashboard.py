from __future__ import annotations

import json
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
    "intent": "Extragerea intenției",
    "final_status": "Clasificarea statusului final",
    "incongruities": "Detecția neconcordanțelor",
}

TASK_ORDER = ["intent", "final_status", "incongruities"]
PROMPT_ORDER = {"v1": 1, "v2": 2, "v3": 3, "v3_ro_labels": 3.5, "v4": 4}

MODEL_INFO = {
    "openai_o3": ("OpenAI o3", "API"),
    "OpenAI o3": ("OpenAI o3", "API"),
    "gemini_2.5_flash": ("Gemini 2.5 Flash", "API"),
    "Gemini 2.5 Flash": ("Gemini 2.5 Flash", "API"),
    "aya_expanse_8b": ("Aya Expanse 8B", "Local"),
    "Aya Expanse 8B": ("Aya Expanse 8B", "Local"),
    "rollama2_7b": ("RoLLaMA 2 7B", "Local"),
    "RoLLaMA 2 7B": ("RoLLaMA 2 7B", "Local"),
    "mistral_7b": ("Mistral 7B", "Local"),
    "Mistral 7B": ("Mistral 7B", "Local"),
    "qwen2_5_3b": ("Qwen2.5 3B", "Local"),
    "qwen2.5_3b": ("Qwen2.5 3B", "Local"),
    "Qwen2.5 3B": ("Qwen2.5 3B", "Local"),
    "roberta_encoder": ("XLM-RoBERTa", "Encoder"),
    "XLM-RoBERTa": ("XLM-RoBERTa", "Encoder"),
}

THESIS_RECOMMENDATIONS = {
    "intent": "OpenAI o3, EN v4: Accuracy/F1 98.3%. Gemini 2.5 Flash RO v4 este foarte aproape, la 97.8%.",
    "final_status": "OpenAI o3, RO v4: Accuracy 90.0%, Macro-F1 0.860. Statusul final este singurul task unde promptul în română câștigă clar pentru o3.",
    "incongruities": "Gemini 2.5 Flash, EN v4: Macro-F1 pe tip 0.734 și acord substanțial. Taskul rămâne cel mai dificil din cele trei.",
}


def repair_text(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return value.encode("latin1").decode("utf-8")
        except UnicodeError:
            return value
    return value


def model_display(model: Any) -> str:
    return MODEL_INFO.get(str(model), (repair_text(model), "Local"))[0]


def model_kind(model: Any) -> str:
    return MODEL_INFO.get(str(model), (repair_text(model), "Local"))[1]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def prediction_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        return {"n_predictions": 0, "n_failed": None, "parse_failure_rate": None, "latency_ms": None, "p95_latency_ms": None}
    frame = pd.DataFrame(items)
    failed = frame["parse_failed"].fillna(False).astype(bool).sum() if "parse_failed" in frame else None
    latency = pd.to_numeric(frame["latency_ms"], errors="coerce") if "latency_ms" in frame else pd.Series(dtype=float)
    return {
        "n_predictions": len(items),
        "n_failed": int(failed) if failed is not None else None,
        "parse_failure_rate": float(failed / len(items)) if failed is not None and len(items) else None,
        "latency_ms": float(latency.mean()) if not latency.dropna().empty else None,
        "p95_latency_ms": float(latency.quantile(0.95)) if not latency.dropna().empty else None,
    }


def parse_json_experiment(path: Path, task: str) -> dict[str, Any]:
    data = read_json(path)
    experiment = data.get("experiment") or {}
    metrics = data.get("metrics") or {}
    raw_model = experiment.get("model") or data.get("model") or "necunoscut"
    items = data.get("predictions") or data.get("results") or []
    items = items if isinstance(items, list) else []
    row = {
        "task": task,
        "model_key": repair_text(raw_model),
        "model": model_display(raw_model),
        "model_type": model_kind(raw_model),
        "language": str(repair_text(experiment.get("lang") or data.get("language") or "")).lower(),
        "prompt_version": repair_text(experiment.get("prompt_version") or data.get("prompt_version") or ""),
        "dataset_size": experiment.get("n_conversations") or data.get("dataset_size"),
        "timestamp": experiment.get("timestamp") or data.get("timestamp"),
        "source_file": path.name,
    }
    row.update(prediction_summary(items))
    for key, value in metrics.items():
        if isinstance(value, (int, float, str)) or value is None:
            row[key] = value
    if row.get("n_failed") is None and row.get("n_valid") is not None and row.get("dataset_size") is not None:
        row["n_failed"] = int(row["dataset_size"]) - int(row["n_valid"])
        row["parse_failure_rate"] = row["n_failed"] / int(row["dataset_size"])
    return row


@st.cache_data(show_spinner=False)
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


@st.cache_data(show_spinner=False)
def parse_intent_report() -> pd.DataFrame:
    report = REPORTS / "eval_intent.txt"
    if not report.exists():
        return pd.DataFrame()
    rows = []
    for line in report.read_text(encoding="utf-8", errors="replace").splitlines():
        if "│" not in line or "ms" not in line or "%" not in line:
            continue
        parts = [part.strip() for part in line.split("│") if part.strip()]
        if len(parts) < 11:
            continue
        model, kind, lang, prompt = parts[0], parts[1], parts[2], parts[3]
        if kind not in {"API", "Local"} or lang not in {"EN", "RO"} or not prompt.startswith("v"):
            continue
        try:
            rows.append(
                {
                    "task": "intent",
                    "model_key": model.replace("★", "").strip(),
                    "model": model.replace("★", "").strip(),
                    "model_type": kind,
                    "language": lang.lower(),
                    "prompt_version": prompt,
                    "accuracy": float(parts[4].replace("%", "")) / 100,
                    "precision": float(parts[5]),
                    "recall": float(parts[6]),
                    "macro_f1": float(parts[7]),
                    "kappa_label": parts[8],
                    "latency_ms": float(parts[9].replace("ms", "").strip()),
                    "dataset_size": int(parts[10]),
                    "n_failed": 0,
                    "parse_failure_rate": 0.0,
                    "source_file": "evaluation_reports/eval_intent.txt",
                }
            )
        except ValueError:
            continue
    return pd.DataFrame(rows).drop_duplicates()


@st.cache_data(show_spinner=False)
def load_all_metrics() -> pd.DataFrame:
    metrics = pd.concat([parse_intent_report(), load_json_metrics()], ignore_index=True, sort=False)
    if metrics.empty:
        return metrics
    for column in ["accuracy", "macro_f1", "weighted_f1", "binary_accuracy", "binary_f1", "type_accuracy", "type_macro_f1", "kappa"]:
        if column in metrics:
            metrics[column] = pd.to_numeric(metrics[column], errors="coerce")
    metrics["prompt_order"] = metrics["prompt_version"].map(PROMPT_ORDER).fillna(99)
    metrics["config"] = metrics["model"].astype(str) + " · " + metrics["language"].str.upper() + " " + metrics["prompt_version"].astype(str)
    return metrics


def metric_column(task: str, df: pd.DataFrame) -> str:
    candidates = {
        "intent": ["macro_f1", "accuracy"],
        "final_status": ["macro_f1", "accuracy", "weighted_f1"],
        "incongruities": ["type_macro_f1", "binary_f1", "binary_accuracy"],
    }[task]
    for column in candidates:
        if column in df.columns and df[column].notna().any():
            return column
    return "dataset_size"


def format_percent(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    number = float(value)
    return f"{number * 100:.1f}%" if number <= 1.5 else f"{number:.1f}"


def best_rows(df: pd.DataFrame, by: str, metric: str) -> pd.DataFrame:
    if df.empty or metric not in df:
        return pd.DataFrame()
    return df.dropna(subset=[metric]).sort_values(metric, ascending=False).groupby(by, as_index=False).head(1)


def filtered_task_df(metrics: pd.DataFrame, task: str) -> pd.DataFrame:
    return metrics[metrics["task"] == task].copy()


def render_best_cards(df: pd.DataFrame, task: str) -> None:
    metric = metric_column(task, df)
    best = df.dropna(subset=[metric]).sort_values(metric, ascending=False).head(1)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Experimente", len(df))
    col2.metric("Metrică principală", metric)
    if best.empty:
        col3.metric("Cel mai bun scor", "n/a")
        col4.metric("Configurație", "n/a")
        return
    row = best.iloc[0]
    col3.metric("Cel mai bun scor", format_percent(row[metric]))
    col4.metric("Configurație", row["config"])


def chart_top_configs(df: pd.DataFrame, task: str, limit: int = 12) -> None:
    metric = metric_column(task, df)
    chart_df = df.dropna(subset=[metric]).sort_values(metric, ascending=False).head(limit)
    if chart_df.empty:
        st.caption("Nu există date suficiente pentru grafic.")
        return
    st.bar_chart(chart_df.set_index("config")[[metric]])


def render_prompt_evolution(df: pd.DataFrame, task: str) -> None:
    metric = metric_column(task, df)
    evo = df.dropna(subset=[metric]).copy()
    if evo.empty:
        st.caption("Nu există date pentru evoluția prompturilor.")
        return
    evo = evo[evo["prompt_version"].astype(str).str.startswith("v")]
    selected = best_rows(evo, ["model", "language", "prompt_version"], metric)
    pivot = selected.pivot_table(index="prompt_order", columns="model", values=metric, aggfunc="max").sort_index()
    if pivot.empty:
        st.caption("Nu există serii comparabile v1-v4.")
        return
    st.line_chart(pivot)
    st.caption("Graficul urmărește progresia v1-v4, ca în subsecțiunile de evoluție a prompturilor din capitolul 6.")


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


def render_methodology(metrics: pd.DataFrame) -> None:
    st.header("6.1 Metodologia de evaluare")
    st.write(
        "Dashboard-ul urmează logica din capitolul de evaluare: fiecare experiment este definit de task, model, "
        "limba promptului și versiunea promptului, apoi este comparat prin metrici agregate, latență și fiabilitatea outputului."
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Taskuri", metrics["task"].nunique())
    c2.metric("Experimente încărcate", len(metrics))
    c3.metric("Modele", metrics["model"].nunique())
    c4.metric("Conversații / rulare", int(metrics["dataset_size"].dropna().max()))
    st.subheader("Spațiul experimental")
    experiment_space = metrics.groupby(["task", "model_type"]).size().reset_index(name="experimente")
    experiment_space["task"] = experiment_space["task"].map(TASK_LABELS)
    st.dataframe(experiment_space, use_container_width=True, hide_index=True)
    st.subheader("Metrici folosite")
    st.markdown(
        "- **Accuracy**: proporția predicțiilor corecte.\n"
        "- **Macro-F1**: scor echilibrat pe clase, util când distribuția este dezechilibrată.\n"
        "- **Weighted-F1**: scor ponderat cu frecvența claselor, apropiat de perspectiva operațională.\n"
        "- **Cohen's κ**: acordul peste nivelul așteptat întâmplător.\n"
        "- **Parse failure rate**: rata răspunsurilor invalide sau neparsabile ca JSON."
    )


def render_task_page(metrics: pd.DataFrame) -> None:
    st.header("6.3-6.5 Rezultate pe task")
    task = st.selectbox("Task", TASK_ORDER, format_func=lambda value: TASK_LABELS[value])
    task_df = filtered_task_df(metrics, task)
    st.info(THESIS_RECOMMENDATIONS[task])

    with st.expander("Filtre", expanded=True):
        c1, c2, c3 = st.columns(3)
        models = c1.multiselect("Modele", sorted(task_df["model"].dropna().unique()), default=sorted(task_df["model"].dropna().unique()))
        languages = c2.multiselect("Limbă", sorted(task_df["language"].dropna().unique()), default=sorted(task_df["language"].dropna().unique()))
        versions = c3.multiselect("Prompt", sorted(task_df["prompt_version"].dropna().unique()), default=sorted(task_df["prompt_version"].dropna().unique()))
    view = task_df[task_df["model"].isin(models) & task_df["language"].isin(languages) & task_df["prompt_version"].isin(versions)]

    render_best_cards(view, task)
    st.subheader("Cele mai bune configurații")
    chart_top_configs(view, task)
    st.dataframe(view.sort_values(metric_column(task, view), ascending=False), use_container_width=True, hide_index=True)

    st.subheader("Cele mai bune rezultate per model")
    metric = metric_column(task, view)
    per_model = best_rows(view, "model", metric)
    st.dataframe(per_model.sort_values(metric, ascending=False), use_container_width=True, hide_index=True)

    st.subheader("Evoluția prompturilor")
    render_prompt_evolution(view, task)


def render_transversal(metrics: pd.DataFrame) -> None:
    st.header("6.6 Analize comparative transversale")

    st.subheader("Comparație între taskuri")
    rows = []
    for task in TASK_ORDER:
        task_df = filtered_task_df(metrics, task)
        metric = metric_column(task, task_df)
        best = task_df.dropna(subset=[metric]).sort_values(metric, ascending=False).head(1)
        if not best.empty:
            row = best.iloc[0].to_dict()
            rows.append({"task": TASK_LABELS[task], "metric": metric, "best_score": row[metric], "best_config": row["config"]})
    best_task_df = pd.DataFrame(rows)
    st.dataframe(best_task_df, use_container_width=True, hide_index=True)
    if not best_task_df.empty:
        st.bar_chart(best_task_df.set_index("task")[["best_score"]])

    st.subheader("Modele API vs modele locale")
    comparable = []
    for task in TASK_ORDER:
        task_df = filtered_task_df(metrics, task)
        metric = metric_column(task, task_df)
        best_type = best_rows(task_df, ["model_type"], metric)
        if not best_type.empty:
            best_type = best_type.assign(task=TASK_LABELS[task], metric=metric, score=best_type[metric])
            comparable.append(best_type[["task", "model_type", "model", "config", "score", "metric"]])
    if comparable:
        api_local = pd.concat(comparable, ignore_index=True)
        st.dataframe(api_local.sort_values(["task", "score"], ascending=[True, False]), use_container_width=True, hide_index=True)
        st.bar_chart(api_local.pivot_table(index="task", columns="model_type", values="score", aggfunc="max"))

    st.subheader("Impactul limbii promptului")
    language_rows = []
    for task in TASK_ORDER:
        task_df = filtered_task_df(metrics, task)
        metric = metric_column(task, task_df)
        per_lang = best_rows(task_df, ["model", "language"], metric)
        for model, group in per_lang.groupby("model"):
            values = group.set_index("language")[metric]
            if "ro" in values and "en" in values:
                language_rows.append(
                    {
                        "task": TASK_LABELS[task],
                        "model": model,
                        "M-F1 RO": values.get("ro"),
                        "M-F1 EN": values.get("en"),
                        "diferență RO-EN": values.get("ro") - values.get("en"),
                        "câștigător": "RO" if values.get("ro") > values.get("en") else "EN" if values.get("en") > values.get("ro") else "≈",
                    }
                )
    language_df = pd.DataFrame(language_rows)
    st.dataframe(language_df, use_container_width=True, hide_index=True)

    st.subheader("Latență")
    latency = metrics.dropna(subset=["latency_ms"]).copy()
    if not latency.empty:
        best_latency = best_rows(latency, ["task", "model"], "latency_ms")
        st.bar_chart(best_latency.sort_values("latency_ms").set_index("config")[["latency_ms"]].head(20))

    st.subheader("Fiabilitatea outputului")
    reliability = metrics.dropna(subset=["parse_failure_rate"]).copy()
    if not reliability.empty:
        reliability["parse_failure_%"] = reliability["parse_failure_rate"] * 100
        st.dataframe(
            reliability.sort_values("parse_failure_%", ascending=False)[
                ["task", "model", "language", "prompt_version", "n_failed", "parse_failure_%", "source_file"]
            ],
            use_container_width=True,
            hide_index=True,
        )


def render_predictions(metrics: pd.DataFrame) -> None:
    st.header("Inspectare predicții")
    task = st.selectbox("Task cu predicții JSON", ["final_status", "incongruities"], format_func=lambda value: TASK_LABELS[value])
    task_df = filtered_task_df(metrics, task)
    source = st.selectbox("Fișier", sorted(task_df["source_file"].dropna().unique()))
    predictions = load_predictions(task, source)
    if predictions.empty:
        st.warning("Fișierul selectat nu conține predicții afișabile.")
        return
    only_errors = st.checkbox("Arată doar greșelile / parse failed", value=True)
    view = predictions.copy()
    if only_errors:
        mask = pd.Series(False, index=view.index)
        if "parse_failed" in view:
            mask = mask | view["parse_failed"].fillna(False).astype(bool)
        for left, right in [
            ("dataset_status", "predicted_status"),
            ("dataset_has_incongruity", "predicted_has_incongruity"),
            ("dataset_incongruity_type", "predicted_incongruity_type"),
        ]:
            if left in view and right in view:
                mask = mask | (view[left].astype(str) != view[right].astype(str))
        view = view[mask]
    st.dataframe(view, use_container_width=True, hide_index=True)


def render_reports() -> None:
    st.header("Rapoarte text din repository")
    report_options = {
        "Intent": REPORTS / "eval_intent.txt",
        "Status final": REPORTS / "eval_final_status.txt",
        "Neconcordanțe": REPORTS / "eval_incongruities.txt",
        "Comparație transversală": REPORTS / "eval_cross_task.txt",
    }
    selected = st.selectbox("Raport", list(report_options))
    path = report_options[selected]
    if path.exists():
        st.text(path.read_text(encoding="utf-8", errors="replace")[:30000])
    else:
        st.warning("Raportul nu există în repository.")


st.set_page_config(page_title="Capitolul 6 · Dashboard evaluare", layout="wide")
st.title("Capitolul 6 · Evaluarea modelelor")
st.caption("Versiune interactivă a capitolului de evaluare: metodologie, rezultate pe task și analize transversale.")

metrics = load_all_metrics()
if metrics.empty:
    st.warning("Nu am găsit rezultate de evaluare în repository.")
    st.stop()

page = st.sidebar.radio(
    "Secțiune",
    [
        "6.1 Metodologie",
        "6.3-6.5 Rezultate pe task",
        "6.6 Analize transversale",
        "Predicții",
        "Rapoarte text",
    ],
)

if page == "6.1 Metodologie":
    render_methodology(metrics)
elif page == "6.3-6.5 Rezultate pe task":
    render_task_page(metrics)
elif page == "6.6 Analize transversale":
    render_transversal(metrics)
elif page == "Predicții":
    render_predictions(metrics)
else:
    render_reports()
