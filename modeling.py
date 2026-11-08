from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    f1_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

TARGET_COLUMN = "Outcome"
ID_COLUMN = "Patient ID"
DATE_COLUMN = "Treatment Start Date"

RAW_FEATURE_COLUMNS = [
    "Age",
    "Gender",
    "Diagnosis",
    "Symptom Severity (1-10)",
    "Mood Score (1-10)",
    "Sleep Quality (1-10)",
    "Physical Activity (hrs/week)",
    "Medication",
    "Therapy Type",
    DATE_COLUMN,
    "Treatment Duration (weeks)",
    "Stress Level (1-10)",
    "Treatment Progress (1-10)",
    "AI-Detected Emotional State",
    "Adherence to Treatment (%)",
]

NUMERIC_FEATURES = [
    "Age",
    "Symptom Severity (1-10)",
    "Mood Score (1-10)",
    "Sleep Quality (1-10)",
    "Physical Activity (hrs/week)",
    "Treatment Duration (weeks)",
    "Stress Level (1-10)",
    "Treatment Progress (1-10)",
    "Adherence to Treatment (%)",
    "Treatment Start Year",
    "Treatment Start Month",
    "Treatment Start Day",
]

CATEGORICAL_FEATURES = [
    "Gender",
    "Diagnosis",
    "Medication",
    "Therapy Type",
    "AI-Detected Emotional State",
]

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knn.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}


def prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and transform the raw assignment columns for model input."""
    missing = [column for column in RAW_FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {', '.join(missing)}")

    features = frame[RAW_FEATURE_COLUMNS].copy()
    treatment_date = pd.to_datetime(features.pop(DATE_COLUMN), errors="coerce")
    features["Treatment Start Year"] = treatment_date.dt.year
    features["Treatment Start Month"] = treatment_date.dt.month
    features["Treatment Start Day"] = treatment_date.dt.day

    for column in NUMERIC_FEATURES:
        features[column] = pd.to_numeric(features[column], errors="coerce")
    for column in CATEGORICAL_FEATURES:
        features[column] = features[column].astype("string").replace("<NA>", np.nan)

    return features[NUMERIC_FEATURES + CATEGORICAL_FEATURES]


def split_features_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if TARGET_COLUMN not in frame.columns:
        raise ValueError(f"The CSV must include the '{TARGET_COLUMN}' target column.")
    labeled = frame.loc[frame[TARGET_COLUMN].notna()].copy()
    if labeled.empty:
        raise ValueError("The uploaded CSV has no labeled rows to evaluate.")
    return prepare_features(labeled), labeled[TARGET_COLUMN].astype(str)


def make_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        sparse_threshold=0,
    )


def make_models() -> dict[str, Pipeline]:
    classifiers = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6, class_weight="balanced", random_state=42
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7, weights="distance"),
        "Gaussian Naive Bayes": GaussianNB(),
        "Support Vector Machine": CalibratedClassifierCV(
            estimator=SVC(kernel="rbf", class_weight="balanced", random_state=42),
            method="sigmoid",
            cv=5,
        ),
    }
    return {
        name: Pipeline(
            [("preprocessor", make_preprocessor()), ("classifier", classifier)]
        )
        for name, classifier in classifiers.items()
    }


def calculate_metrics(
    actual: Sequence[str],
    predicted: Sequence[str],
    probabilities: np.ndarray,
    classes: Sequence[str],
) -> dict[str, float]:
    """Return the six rubric metrics; AUC is macro one-vs-rest."""
    actual_array = np.asarray(actual)
    predicted_array = np.asarray(predicted)
    auc_scores = []
    for index, class_name in enumerate(classes):
        binary_actual = (actual_array == class_name).astype(int)
        if np.unique(binary_actual).size == 2:
            auc_scores.append(roc_auc_score(binary_actual, probabilities[:, index]))

    return {
        "Accuracy": accuracy_score(actual_array, predicted_array),
        "AUC": float(np.mean(auc_scores)) if auc_scores else float("nan"),
        "Precision": precision_score(
            actual_array, predicted_array, average="macro", zero_division=0
        ),
        "Recall": recall_score(
            actual_array, predicted_array, average="macro", zero_division=0
        ),
        "F1": f1_score(actual_array, predicted_array, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(actual_array, predicted_array),
    }