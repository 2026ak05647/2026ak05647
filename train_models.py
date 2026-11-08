from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from modeling import (
    ID_COLUMN,
    MODEL_FILES,
    TARGET_COLUMN,
    calculate_metrics,
    make_models,
    split_features_target,
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "mental_health_diagnosis_treatment.csv"
MODEL_DIR = ROOT / "models"
TEST_DATA_PATH = ROOT / "test_data.csv"


def main() -> None:
    data = pd.read_csv(DATA_PATH)
    if len(data) < 500:
        raise ValueError(f"The assignment requires at least 500 rows; found {len(data)}.")

    training_data, test_data = train_test_split(
        data,
        test_size=0.2,
        random_state=42,
        stratify=data[TARGET_COLUMN],
    )
    train_features, train_target = split_features_target(training_data)
    test_features, test_target = split_features_target(test_data)

    MODEL_DIR.mkdir(exist_ok=True)
    results = []
    models = make_models()
    for name, model in models.items():
        model.fit(train_features, train_target)
        joblib.dump(model, MODEL_DIR / MODEL_FILES[name])
        predicted = model.predict(test_features)
        probabilities = model.predict_proba(test_features)
        results.append(
            {"Model": name, **calculate_metrics(test_target, predicted, probabilities, model.classes_)}
        )

    test_data.to_csv(TEST_DATA_PATH, index=False)
    pd.DataFrame(results).to_csv(ROOT / "baseline_metrics.csv", index=False)
    manifest = {
        "target": TARGET_COLUMN,
        "features": [column for column in data.columns if column not in {TARGET_COLUMN, ID_COLUMN}],
        "classes": sorted(data[TARGET_COLUMN].dropna().astype(str).unique().tolist()),
        "train_rows": len(training_data),
        "test_rows": len(test_data),
        "random_state": 42,
    }
    (MODEL_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Saved {len(models)} fitted model pipelines to {MODEL_DIR}")
    print(f"Saved stratified holdout ({len(test_data)} rows) to {TEST_DATA_PATH}")
    print(pd.DataFrame(results).to_string(index=False, float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()