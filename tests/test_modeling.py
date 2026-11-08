import numpy as np
import pandas as pd
import pytest

from modeling import RAW_FEATURE_COLUMNS, calculate_metrics, prepare_features


def sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            **{column: [1, 2] for column in RAW_FEATURE_COLUMNS},
            "Treatment Start Date": ["2024-01-02", "2024-02-03"],
            "Patient ID": [101, 102],
            "Outcome": ["Improved", "Deteriorated"],
        }
    )


def test_prepare_features_expands_date_and_excludes_identifier_and_target():
    features = prepare_features(sample_frame())

    assert "Treatment Start Year" in features.columns
    assert "Treatment Start Month" in features.columns
    assert "Treatment Start Day" in features.columns
    assert "Treatment Start Date" not in features.columns
    assert "Patient ID" not in features.columns
    assert "Outcome" not in features.columns
    assert features.loc[0, "Treatment Start Month"] == 1


def test_prepare_features_reports_missing_required_fields():
    with pytest.raises(ValueError, match="Missing required feature columns"):
        prepare_features(pd.DataFrame({"Age": [30]}))


def test_calculate_metrics_returns_all_six_rubric_metrics():
    actual = np.array(["Improved", "Deteriorated", "Improved", "Deteriorated"])
    predicted = np.array(["Improved", "Deteriorated", "Deteriorated", "Deteriorated"])
    probabilities = np.array([[0.9, 0.1], [0.2, 0.8], [0.4, 0.6], [0.1, 0.9]])

    scores = calculate_metrics(
        actual,
        predicted,
        probabilities,
        ["Improved", "Deteriorated"],
    )

    assert set(scores) == {"Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"}
    assert all(0 <= score <= 1 for score in scores.values())