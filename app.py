from __future__ import annotations

from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from modeling import (
    ID_COLUMN,
    MODEL_FILES,
    TARGET_COLUMN,
    calculate_metrics,
    split_features_target,
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
DEFAULT_TEST_PATH = ROOT / "test_data.csv"
SOURCE_DATA_PATH = ROOT / "mental_health_diagnosis_treatment.csv"

st.set_page_config(
    page_title="Care outcomes | Model lab",
    page_icon=":material/monitoring:",
    layout="wide",
)


@st.cache_resource
def load_models() -> dict[str, object]:
    return {
        name: joblib.load(MODEL_DIR / filename)
        for name, filename in MODEL_FILES.items()
    }


@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def evaluate_all(
    models: dict[str, object], data: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    features, actual = split_features_target(data)
    records = []
    details = {}
    for name, model in models.items():
        predicted = model.predict(features)
        probabilities = model.predict_proba(features)
        records.append(
            {"Model": name, **calculate_metrics(actual, predicted, probabilities, model.classes_)}
        )
        details[name] = {"actual": actual, "predicted": predicted, "classes": model.classes_}
    return pd.DataFrame(records).set_index("Model"), details


def render_confusion_matrix(actual: pd.Series, predicted: pd.Series, labels: list[str]) -> None:
    matrix = confusion_matrix(actual, predicted, labels=labels)
    rows = [
        {"Actual": labels[row], "Predicted": labels[column], "Patients": int(matrix[row, column])}
        for row in range(len(labels))
        for column in range(len(labels))
    ]
    chart = (
        alt.Chart(pd.DataFrame(rows))
        .mark_rect(cornerRadius=2)
        .encode(
            x=alt.X("Predicted:N", title="Predicted outcome", sort=labels),
            y=alt.Y("Actual:N", title="Actual outcome", sort=labels),
            color=alt.Color("Patients:Q", scale=alt.Scale(scheme="tealblues"), title="Patients"),
            tooltip=["Actual:N", "Predicted:N", "Patients:Q"],
        )
        .properties(height=260)
    )
    st.altair_chart(chart, alt="Confusion matrix comparing actual and predicted treatment outcomes")


try:
    models = load_models()
except FileNotFoundError:
    st.error("Saved models are missing. Run `python train_models.py` once to create them.")
    st.stop()

with st.sidebar:
    st.markdown("### Care outcomes")
    st.caption("Treatment outcome classification lab")
    uploaded_file = st.file_uploader("Upload a labeled test CSV", type=["csv"])
    selected_model = st.selectbox("Inspect model", list(MODEL_FILES))
    st.caption("Uploads are scored by saved models. The app never retrains on uploaded data.")

try:
    if uploaded_file is not None:
        evaluation_data = pd.read_csv(uploaded_file)
        data_source = uploaded_file.name
    else:
        evaluation_data = load_csv(str(DEFAULT_TEST_PATH))
        data_source = "Built-in stratified holdout"
    comparison, details = evaluate_all(models, evaluation_data)
except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as error:
    st.error(str(error))
    st.stop()

unknown_labels = set(evaluation_data[TARGET_COLUMN].dropna().astype(str)) - set(
    next(iter(models.values())).classes_
)
if unknown_labels:
    st.error(f"Unknown outcome labels: {', '.join(sorted(unknown_labels))}")
    st.stop()

selected = details[selected_model]
selected_scores = comparison.loc[selected_model]
st.title("Care outcomes, in the model lab", icon=":material/monitoring:")
st.caption(
    "Compare five classifiers on treatment outcomes. Results below are calculated on the selected test CSV."
)

with st.container(horizontal=True):
    st.metric("Test records", f"{len(evaluation_data):,}", border=True)
    st.metric("Outcome classes", f"{evaluation_data[TARGET_COLUMN].nunique()}", border=True)
    st.metric("Selected model", selected_model, border=True)
    st.metric("Accuracy", f"{selected_scores['Accuracy']:.3f}", border=True)

evaluation_tab, dataset_tab, method_tab = st.tabs(
    ["Model evaluation", "Dataset explorer", "Method notes"]
)

with evaluation_tab:
    st.subheader("Five-model comparison")
    st.caption("Precision, recall, and F1 use macro averaging; AUC is macro one-vs-rest.")
    st.dataframe(
        comparison.reset_index(),
        column_config={
            "Model": st.column_config.TextColumn("Model", width="medium"),
            **{
                metric: st.column_config.NumberColumn(
                    metric, format="%.3f", width="small"
                )
                for metric in comparison.columns
            },
        },
        hide_index=True,
        alt="Evaluation metrics for all five saved classifiers",
    )
    metric_data = comparison.reset_index().melt(
        id_vars="Model", var_name="Metric", value_name="Score"
    )
    metric_chart = (
        alt.Chart(metric_data)
        .mark_bar()
        .encode(
            x=alt.X("Model:N", sort=list(comparison.index), axis=alt.Axis(labelAngle=-25)),
            xOffset=alt.XOffset("Metric:N"),
            y=alt.Y("Score:Q", title="Score", scale=alt.Scale(domain=[-1, 1])),
            color=alt.Color("Metric:N", title="Metric"),
            tooltip=["Model:N", "Metric:N", alt.Tooltip("Score:Q", format=".3f")],
        )
        .properties(height=320)
    )
    st.altair_chart(
        metric_chart,
        alt="Six evaluation metrics compared across the five classifiers",
    )

    st.subheader(f"{selected_model} diagnostics")
    matrix_column, report_column = st.columns(2)
    labels = list(selected["classes"])
    with matrix_column:
        st.markdown("**Confusion matrix**")
        render_confusion_matrix(selected["actual"], selected["predicted"], labels)
    with report_column:
        st.markdown("**Classification report**")
        report = classification_report(
            selected["actual"],
            selected["predicted"],
            labels=labels,
            target_names=labels,
            output_dict=True,
            zero_division=0,
        )
        report_frame = pd.DataFrame(report).transpose().rename_axis("Class")
        st.dataframe(
            report_frame.style.format("{:.3f}", na_rep=""),
            alt=f"Classification report for {selected_model}",
        )

    winner = comparison["F1"].idxmax()
    st.caption(f"Highest macro F1 on this test file: {winner} ({comparison.loc[winner, 'F1']:.3f}).")

with dataset_tab:
    st.subheader("Test set composition")
    outcome_counts = (
        evaluation_data[TARGET_COLUMN]
        .astype(str)
        .value_counts()
        .rename_axis("Outcome")
        .reset_index(name="Patients")
    )
    chart_column, summary_column = st.columns([1.2, 1])
    with chart_column:
        st.bar_chart(
            outcome_counts,
            x="Outcome",
            y="Patients",
            height=300,
            alt="Number of test records in each treatment outcome class",
        )
    with summary_column:
        st.markdown("**Evaluation file**")
        st.write(data_source)
        st.markdown("**Rows with a recorded outcome**")
        st.write(f"{evaluation_data[TARGET_COLUMN].notna().sum():,}")
        st.markdown("**Input features**")
        st.write(f"{len([column for column in evaluation_data.columns if column not in {TARGET_COLUMN, ID_COLUMN}])}")
    st.subheader("Preview")
    st.dataframe(evaluation_data.head(20), hide_index=True, alt="First 20 rows of the evaluation dataset")

with method_tab:
    st.subheader("Training and evaluation")
    st.markdown(
        "The five pipelines were fitted once on a stratified 80% training split. "
        "The bundled test file is the remaining 20%; uploading another CSV only runs `predict` and `predict_proba`."
    )
    st.markdown(
        "Preprocessing imputes missing numeric/categorical values, scales numeric inputs, "
        "one-hot encodes categories, and derives calendar fields from the treatment start date. "
        "Patient ID is excluded from model inputs."
    )
    st.markdown(
        "**Scope:** this coursework dataset and these models are for learning and evaluation only. "
        "They are not a clinical diagnostic or treatment tool."
    )

st.caption(f"Evaluation source: {data_source} · Models: {len(models)} · Target: {TARGET_COLUMN}")