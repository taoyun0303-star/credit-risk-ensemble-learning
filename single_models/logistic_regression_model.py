import os
import time

import numpy as np
import pandas as pd


class LogisticRegressionModel:
    """Binary logistic regression trained with batch gradient descent."""

    def __init__(self, params=None):
        if params is None:
            params = {
                "learning_rate": 0.05,
                "max_iter": 2000,
                "l2_lambda": 1e-3,
                "tol": 1e-7,
                "fit_intercept": True,
                "random_state": 42,
            }

        self.params = params
        self.learning_rate = float(params.get("learning_rate", 0.05))
        self.max_iter = int(params.get("max_iter", 2000))
        self.l2_lambda = float(params.get("l2_lambda", 1e-3))
        self.tol = float(params.get("tol", 1e-7))
        self.fit_intercept = bool(params.get("fit_intercept", True))
        self.random_state = int(params.get("random_state", 42))

        self.weights_ = None
        self.bias_ = 0.0
        self.loss_history_ = []

    @staticmethod
    def _sigmoid(z):
        z = np.asarray(z, dtype=np.float64)
        z = np.clip(z, -500.0, 500.0)
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def _validate_xy(X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")

        unique_values = np.unique(y)
        if not np.all(np.isin(unique_values, [0.0, 1.0])):
            raise ValueError("y must be binary labels in {0, 1}.")

        return X, y

    def _compute_loss(self, y_true, y_prob):
        eps = 1e-12
        y_prob = np.clip(y_prob, eps, 1.0 - eps)

        ce = -np.mean(y_true * np.log(y_prob) + (1.0 - y_true) * np.log(1.0 - y_prob))
        reg = 0.5 * self.l2_lambda * np.sum(self.weights_ ** 2)
        return ce + reg

    def train(self, X_train, y_train, X_val=None, y_val=None):
        X, y = self._validate_xy(X_train, y_train)

        rng = np.random.RandomState(self.random_state)
        n_samples, n_features = X.shape

        self.weights_ = rng.normal(loc=0.0, scale=0.01, size=n_features)
        self.bias_ = 0.0
        self.loss_history_ = []

        prev_loss = np.inf

        for _ in range(self.max_iter):
            linear = X @ self.weights_
            if self.fit_intercept:
                linear = linear + self.bias_

            y_prob = self._sigmoid(linear)
            loss = self._compute_loss(y, y_prob)
            self.loss_history_.append(loss)

            # L2 regularization is applied only to weights, not the bias term.
            error = y_prob - y
            grad_w = (X.T @ error) / n_samples + self.l2_lambda * self.weights_
            grad_b = float(np.mean(error))

            self.weights_ -= self.learning_rate * grad_w
            if self.fit_intercept:
                self.bias_ -= self.learning_rate * grad_b

            if abs(prev_loss - loss) < self.tol:
                break
            prev_loss = loss

        return self

    def _check_trained(self):
        if self.weights_ is None:
            raise ValueError("Model not trained yet.")

    def predict_proba(self, X):
        self._check_trained()
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")

        linear = X @ self.weights_
        if self.fit_intercept:
            linear = linear + self.bias_

        prob_1 = self._sigmoid(linear)
        prob_0 = 1.0 - prob_1
        return np.column_stack([prob_0, prob_1])

    def predict(self, X, threshold=0.5):
        prob_1 = self.predict_proba(X)[:, 1]
        return (prob_1 >= threshold).astype(int)


def confusion_matrix_binary(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    return tn, fp, fn, tp


def accuracy_score_manual(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean(y_true == y_pred)


def precision_score_manual(y_true, y_pred):
    _, fp, _, tp = confusion_matrix_binary(y_true, y_pred)
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def recall_score_manual(y_true, y_pred):
    _, _, fn, tp = confusion_matrix_binary(y_true, y_pred)
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def f1_score_manual(y_true, y_pred):
    precision = precision_score_manual(y_true, y_pred)
    recall = recall_score_manual(y_true, y_pred)
    return 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


def classification_report_binary_manual(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    report = {}
    for cls in [0, 1]:
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        support = np.sum(y_true == cls)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        report[cls] = {
            "precision": precision,
            "recall": recall,
            "f1-score": f1,
            "support": int(support),
        }

    macro_precision = (report[0]["precision"] + report[1]["precision"]) / 2.0
    macro_recall = (report[0]["recall"] + report[1]["recall"]) / 2.0
    macro_f1 = (report[0]["f1-score"] + report[1]["f1-score"]) / 2.0
    total_support = report[0]["support"] + report[1]["support"]

    weighted_precision = (
        report[0]["precision"] * report[0]["support"] + report[1]["precision"] * report[1]["support"]
    ) / total_support
    weighted_recall = (
        report[0]["recall"] * report[0]["support"] + report[1]["recall"] * report[1]["support"]
    ) / total_support
    weighted_f1 = (
        report[0]["f1-score"] * report[0]["support"] + report[1]["f1-score"] * report[1]["support"]
    ) / total_support
    accuracy = accuracy_score_manual(y_true, y_pred)

    print("              precision    recall  f1-score   support")
    print()
    print(
        f"           0       {report[0]['precision']:.2f}      "
        f"{report[0]['recall']:.2f}      {report[0]['f1-score']:.2f}      "
        f"{report[0]['support']}"
    )
    print(
        f"           1       {report[1]['precision']:.2f}      "
        f"{report[1]['recall']:.2f}      {report[1]['f1-score']:.2f}      "
        f"{report[1]['support']}"
    )
    print()
    print(f"    accuracy                           {accuracy:.2f}      {total_support}")
    print(
        f"   macro avg       {macro_precision:.2f}      {macro_recall:.2f}      "
        f"{macro_f1:.2f}      {total_support}"
    )
    print(
        f"weighted avg       {weighted_precision:.2f}      {weighted_recall:.2f}      "
        f"{weighted_f1:.2f}      {total_support}"
    )


def roc_curve_manual(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    thresholds = np.unique(y_score)[::-1]
    tpr_list = [0.0]
    fpr_list = [0.0]

    P = np.sum(y_true == 1)
    N = np.sum(y_true == 0)

    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix_binary(y_true, y_pred)
        tpr = tp / P if P > 0 else 0.0
        fpr = fp / N if N > 0 else 0.0
        tpr_list.append(tpr)
        fpr_list.append(fpr)

    tpr_list.append(1.0)
    fpr_list.append(1.0)
    return np.asarray(fpr_list), np.asarray(tpr_list)


def auc_manual(fpr, tpr):
    order = np.argsort(fpr)
    fpr = fpr[order]
    tpr = tpr[order]
    area = 0.0
    for i in range(1, len(fpr)):
        area += (fpr[i] - fpr[i - 1]) * (tpr[i] + tpr[i - 1]) / 2.0
    return float(area)


def _resolve_data_path(path_candidates):
    for path in path_candidates:
        if os.path.exists(path):
            return path
    return None


def load_processed_data(
    train_path=None,
    test_path=None,
    target_column="loan_status",
):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if train_path is None:
        train_path = _resolve_data_path(
            [
                os.path.join(project_root, "data", "preprocessed_data", "train_processed.csv"),
                os.path.join(project_root, "dataset", "train_processed.csv"),
                os.path.join(project_root, "data", "train_processed.csv"),
                "data/preprocessed_data/train_processed.csv",
                "dataset/train_processed.csv",
                "data/train_processed.csv",
            ]
        )
    if test_path is None:
        test_path = _resolve_data_path(
            [
                os.path.join(project_root, "data", "preprocessed_data", "test_processed.csv"),
                os.path.join(project_root, "dataset", "test_processed.csv"),
                os.path.join(project_root, "data", "test_processed.csv"),
                "data/preprocessed_data/test_processed.csv",
                "dataset/test_processed.csv",
                "data/test_processed.csv",
            ]
        )

    if train_path is None or not os.path.exists(train_path):
        raise FileNotFoundError(
            "Train file not found. Tried: data/preprocessed_data/train_processed.csv, "
            "dataset/train_processed.csv, and data/train_processed.csv"
        )
    if test_path is None or not os.path.exists(test_path):
        raise FileNotFoundError(
            "Test file not found. Tried: data/preprocessed_data/test_processed.csv, "
            "dataset/test_processed.csv, and data/test_processed.csv"
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    if target_column not in train_df.columns or target_column not in test_df.columns:
        raise ValueError(f"Target column '{target_column}' not found in data.")

    X_train = train_df.drop(columns=[target_column]).values
    y_train = train_df[target_column].values
    X_test = test_df.drop(columns=[target_column]).values
    y_test = test_df[target_column].values
    return X_train, y_train, X_test, y_test


def run_logistic_regression_experiment(
    train_path=None,
    test_path=None,
    target_column="loan_status",
    learning_rate=0.05,
    max_iter=2000,
    l2_lambda=1e-3,
    threshold=0.5,
):
    print("Logistic Regression Experiment (From Scratch)")
    print(f"learning_rate: {learning_rate}")
    print(f"max_iter     : {max_iter}")
    print(f"l2_lambda    : {l2_lambda}")

    X_train, y_train, X_test, y_test = load_processed_data(
        train_path=train_path,
        test_path=test_path,
        target_column=target_column,
    )

    model = LogisticRegressionModel(
        params={
            "learning_rate": learning_rate,
            "max_iter": max_iter,
            "l2_lambda": l2_lambda,
            "tol": 1e-7,
            "fit_intercept": True,
            "random_state": 42,
        }
    )

    start = time.time()
    model.train(X_train, y_train)
    y_pred = model.predict(X_test, threshold=threshold)
    y_proba = model.predict_proba(X_test)[:, 1]
    elapsed = time.time() - start

    tn, fp, fn, tp = confusion_matrix_binary(y_test, y_pred)
    acc = accuracy_score_manual(y_test, y_pred)
    precision = precision_score_manual(y_test, y_pred)
    recall = recall_score_manual(y_test, y_pred)
    f1 = f1_score_manual(y_test, y_pred)
    fpr, tpr = roc_curve_manual(y_test, y_proba)
    auc = auc_manual(fpr, tpr)

    print()
    print("Classification Report:")
    classification_report_binary_manual(y_test, y_pred)

    print()
    print("Confusion Matrix:")
    print(np.array([[tn, fp], [fn, tp]]))

    print()
    print(f"Accuracy : {acc:.6f}")
    print(f"Precision: {precision:.6f}")
    print(f"Recall   : {recall:.6f}")
    print(f"F1-score : {f1:.6f}")
    print(f"AUC      : {auc:.6f}")
    print(f"Time Cost: {elapsed:.6f} seconds")

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "auc": auc,
        "time_cost": elapsed,
    }


def get_best_logistic_regression_predictions(
    train_path=None,
    test_path=None,
    target_column="loan_status",
):
    X_train, y_train, X_test, y_test = load_processed_data(
        train_path=train_path,
        test_path=test_path,
        target_column=target_column,
    )

    model = LogisticRegressionModel(
        params={
            "learning_rate": 0.05,
            "max_iter": 2000,
            "l2_lambda": 1e-3,
            "tol": 1e-7,
            "fit_intercept": True,
            "random_state": 42,
        }
    )
    model.train(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return y_test, y_pred, y_proba


if __name__ == "__main__":
    run_logistic_regression_experiment()
