# Mental health treatment outcome classifier

## Problem statement

Compare five supervised classification models for the three-class `Outcome` target and evaluate each model on the same stratified test split. The Streamlit app also accepts a labeled CSV and evaluates it using the saved pipelines without retraining.

## Dataset description

The supplied dataset has 500 patient records, 15 raw predictors, and the `Outcome` target (`Deteriorated`, `No Change`, or `Improved`). Patient ID is retained in the CSV for traceability and excluded from model inputs. The treatment start date is converted to year, month, and day features. The holdout is a stratified 20% split with `random_state=42`.

## Repository and app links

- GitHub repository: https://github.com/2026ak05647/2026ak05647
- Streamlit Community Cloud app: deployment pending

## Run locally

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
py train_models.py
streamlit run app.py
```

The training script saves five fitted pipelines in `models/`, the stratified test split in `test_data.csv`, and the baseline comparison in `baseline_metrics.csv`.

## Models and evaluation metrics

The classifiers are Logistic Regression, Decision Tree, K-Nearest Neighbors, Gaussian Naive Bayes, and Support Vector Machine. Precision, recall, and F1 use macro averaging; AUC is macro one-vs-rest.

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.380 | 0.510 | 0.383 | 0.379 | 0.380 | 0.071 |
| Decision Tree | 0.310 | 0.501 | 0.285 | 0.311 | 0.286 | -0.037 |
| K-Nearest Neighbors | 0.340 | 0.497 | 0.359 | 0.341 | 0.347 | 0.008 |
| Gaussian Naive Bayes | 0.280 | 0.467 | 0.281 | 0.280 | 0.280 | -0.081 |
| Support Vector Machine | 0.430 | 0.578 | 0.614 | 0.423 | 0.369 | 0.154 |

## Performance observations

On this fixed holdout, Support Vector Machine has the highest accuracy, AUC, and MCC, while Logistic Regression has the highest macro F1. No model is consistently strongest across all metrics. Accuracy ranges from 0.280 to 0.430 and MCC is close to zero for every model, so these results indicate weak predictive performance and should not be treated as evidence of clinical usefulness. Re-run the experiment and discuss the metric tradeoffs when using a different split or test file.

## App features

- Upload a labeled test CSV and select one of the saved models.
- Compare all five models on accuracy, AUC, precision, recall, F1, and MCC.
- Inspect the selected model's confusion matrix and classification report.
- Explore outcome distribution and preview the evaluation data.

## Test CSV format

Include the required predictor columns and an `Outcome` column. `Patient ID` is optional. Labels must match the training classes. Uploaded data is evaluated by the saved pipelines and is never used to fit or update a model.

## Submission evidence

Capture the required assignment-execution screenshot from BITS Virtual Lab yourself and include it in the final submission PDF. Include this README content in that PDF as required by the assignment.

## Scope and limitations

This is a coursework demonstration, not a medical diagnostic or treatment recommendation system. Results depend on this dataset and split and should not guide real care.