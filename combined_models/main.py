import json
import os
import sys
import time

import numpy as np
import pandas as pd


DEMO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if DEMO_DIR not in sys.path:
    # Allow this script to import sibling model modules when run directly.
    sys.path.insert(0, DEMO_DIR)

from single_models.decision_tree_model import DecisionTreeClassifierFromScratch
from single_models.knn_model import KNNClassifier
from single_models.logistic_regression_model import LogisticRegressionModel
from single_models.random_forest_model import RandomForestModel


DATA_DIR = os.path.join(DEMO_DIR, "data", "preprocessed_data")
REPORT_DIR = os.path.join(DEMO_DIR, "reports", "generated")
TRAIN_PATH = os.path.join(DATA_DIR, "train_processed.csv")
TEST_PATH = os.path.join(DATA_DIR, "test_processed.csv")
REPORT_MD_PATH = os.path.join(REPORT_DIR, "best_combined_model_report.md")
REPORT_JSON_PATH = os.path.join(REPORT_DIR, "best_combined_model_metrics.json")
TARGET_COL = "loan_status"


BEST_ENSEMBLE = [
    # Tuned model weights used for probability averaging.
    {
        "name": "Random Forest",
        "weight": 0.80,
        "model": RandomForestModel(
            params={
                "n_estimators": 100,
                "max_depth": 15,
                "min_samples_split": 10,
                "min_samples_leaf": 2,
                "max_features": "sqrt",
                "bootstrap": True,
                "random_state": 42,
            }
        ),
    },
    {
        "name": "Decision Tree",
        "weight": 0.05,
        "model": DecisionTreeClassifierFromScratch(
            max_depth=7,
            min_samples_split=10,
            min_samples_leaf=10,
            random_state=42,
        ),
    },
    {
        "name": "KNN",
        "weight": 0.05,
        "model": KNNClassifier(
            k=11,
            distance_metric="manhattan",
            weights="distance",
        ),
    },
    {
        "name": "Logistic Regression",
        "weight": 0.10,
        "model": LogisticRegressionModel(
            params={
                "learning_rate": 0.05,
                "max_iter": 4000,
                "l2_lambda": 0.001,
                "tol": 1e-7,
                "fit_intercept": True,
                "random_state": 42,
            }
        ),
    },
]

CLASSIFICATION_THRESHOLD = 0.50


def validate_ensemble_config():
    """Validate the fixed probability-averaging configuration before training."""
    names = [item["name"] for item in BEST_ENSEMBLE]
    weights = np.asarray([item["weight"] for item in BEST_ENSEMBLE], dtype=float)

    if len(names) != len(set(names)):
        raise ValueError("Each base model must have a unique name.")
    if np.any(weights < 0):
        raise ValueError("Ensemble weights must be non-negative.")
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError(
            f"Ensemble weights must sum to 1.0; received {weights.sum():.6f}."
        )


def load_data():
    if not os.path.exists(TRAIN_PATH) or not os.path.exists(TEST_PATH):
        raise FileNotFoundError(
            "Preprocessed data not found. Run `python data/preprocess.py` first."
        )

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    X_train = train_df.drop(columns=[TARGET_COL]).values
    y_train = train_df[TARGET_COL].values.astype(int)
    X_test = test_df.drop(columns=[TARGET_COL]).values
    y_test = test_df[TARGET_COL].values.astype(int)
    return X_train, y_train, X_test, y_test


def train_model(model, X_train, y_train):
    if hasattr(model, "train"):
        model.train(X_train, y_train)
    else:
        model.fit(X_train, y_train)
    return model


def confusion_matrix_binary(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    return tn, fp, fn, tp


def auc_score(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    positives = y_true == 1
    n_pos = int(np.sum(positives))
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0

    order = np.argsort(y_score, kind="mergesort")
    sorted_scores = y_score[order]
    ranks = np.empty(len(y_score), dtype=np.float64)
    start = 0
    while start < len(y_score):
        end = start + 1
        while end < len(y_score) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end

    rank_sum_pos = float(np.sum(ranks[positives]))
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def evaluate(y_true, y_proba, threshold):
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix_binary(y_true, y_pred)
    accuracy = float(np.mean(np.asarray(y_true) == y_pred))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "accuracy": accuracy,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": auc_score(y_true, y_proba),
        "confusion_matrix": np.array([[tn, fp], [fn, tp]]),
    }


def print_metrics(metrics):
    print("\nBest Combined Model: Probability Averaging")
    print("Confusion Matrix:")
    print(metrics["confusion_matrix"])
    print(f"Accuracy : {metrics['accuracy']:.6f}")
    print(f"Precision: {metrics['precision']:.6f}")
    print(f"Recall   : {metrics['recall']:.6f}")
    print(f"F1-score : {metrics['f1']:.6f}")
    print(f"AUC      : {metrics['auc']:.6f}")


def write_report(metrics, total_time):
    os.makedirs(REPORT_DIR, exist_ok=True)
    cm = metrics["confusion_matrix"].tolist()
    model_rows = "\n".join(
        f"| {item['name']} | {item['weight']:.2f} |" for item in BEST_ENSEMBLE
    )

    report = f"""# Best Combined Model Report

## Model

Strategy: Probability Averaging

Classification threshold: {CLASSIFICATION_THRESHOLD:.2f}

| Base model | Weight |
| --- | ---: |
{model_rows}

## Metrics

| Metric | Value |
| --- | ---: |
| Accuracy | {metrics['accuracy']:.6f} |
| Precision | {metrics['precision']:.6f} |
| Recall | {metrics['recall']:.6f} |
| F1-score | {metrics['f1']:.6f} |
| AUC | {metrics['auc']:.6f} |

## Confusion Matrix

```text
[[TN, FP],
 [FN, TP]] = {cm}
```

Total time: {total_time:.1f}s
"""

    payload = {
        "strategy": "probability_averaging",
        "threshold": CLASSIFICATION_THRESHOLD,
        "models": [
            {"name": item["name"], "weight": item["weight"]}
            for item in BEST_ENSEMBLE
        ],
        "metrics": {
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "auc": metrics["auc"],
            "confusion_matrix": cm,
        },
        "total_time_seconds": total_time,
    }

    with open(REPORT_MD_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    with open(REPORT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Report written to: {REPORT_MD_PATH}")
    print(f"Metrics JSON written to: {REPORT_JSON_PATH}")


def main():
    validate_ensemble_config()
    X_train, y_train, X_test, y_test = load_data()
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    weighted_proba = np.zeros(len(y_test), dtype=np.float64)
    total_start = time.perf_counter()

    for item in BEST_ENSEMBLE:
        start = time.perf_counter()
        print(f"Training {item['name']} (weight={item['weight']:.2f}) ...", flush=True)
        model = train_model(item["model"], X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        weighted_proba += item["weight"] * proba
        print(f"  done in {time.perf_counter() - start:.1f}s", flush=True)

    metrics = evaluate(y_test, weighted_proba, CLASSIFICATION_THRESHOLD)
    total_time = time.perf_counter() - total_start
    print_metrics(metrics)
    print(f"Total time: {total_time:.1f}s")
    write_report(metrics, total_time)
    return metrics


if __name__ == "__main__":
    main()
