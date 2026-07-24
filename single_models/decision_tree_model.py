import os
import time

import numpy as np
import pandas as pd


class DecisionTreeNode:
    def __init__(self):
        self.is_leaf = False
        self.prediction = 0
        self.proba = 0.0
        self.feature_index = None
        self.threshold = None
        self.left = None
        self.right = None


class DecisionTreeClassifierFromScratch:
    """Greedy binary decision tree classifier using Gini impurity."""

    def __init__(self, max_depth=10, min_samples_split=10, min_samples_leaf=5, random_state=42):
        self.max_depth = int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.random_state = int(random_state)
        self.root_ = None
        self.n_features_ = None

    @staticmethod
    def _gini(y):
        if len(y) == 0:
            return 0.0
        p1 = np.mean(y)
        p0 = 1.0 - p1
        return 1.0 - p0 * p0 - p1 * p1

    def _best_split(self, X, y):
        n_samples, n_features = X.shape
        if n_samples < self.min_samples_split:
            return None, None

        parent_gini = self._gini(y)
        best_gain = -np.inf
        best_feature = None
        best_threshold = None

        for feature_idx in range(n_features):
            values = X[:, feature_idx]
            unique_vals = np.unique(values)
            if len(unique_vals) <= 1:
                continue

            # Test midpoints between sorted unique values as split candidates.
            thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0

            for threshold in thresholds:
                left_mask = values <= threshold
                right_mask = ~left_mask

                left_count = np.sum(left_mask)
                right_count = np.sum(right_mask)

                if left_count < self.min_samples_leaf or right_count < self.min_samples_leaf:
                    continue

                y_left = y[left_mask]
                y_right = y[right_mask]

                weighted_gini = (left_count / n_samples) * self._gini(y_left) + (
                    right_count / n_samples
                ) * self._gini(y_right)
                gain = parent_gini - weighted_gini

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold

    def _build_tree(self, X, y, depth):
        node = DecisionTreeNode()
        positive_ratio = float(np.mean(y))
        node.proba = positive_ratio
        node.prediction = int(positive_ratio >= 0.5)

        if (
            depth >= self.max_depth
            or len(y) < self.min_samples_split
            or np.all(y == y[0])
        ):
            node.is_leaf = True
            return node

        feature_idx, threshold = self._best_split(X, y)
        if feature_idx is None:
            node.is_leaf = True
            return node

        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask

        if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
            node.is_leaf = True
            return node

        node.feature_index = feature_idx
        node.threshold = threshold
        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        return node

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same number of samples.")
        if not np.all(np.isin(np.unique(y), [0, 1])):
            raise ValueError("y must be binary labels in {0, 1}.")

        self.n_features_ = X.shape[1]
        self.root_ = self._build_tree(X, y, depth=0)
        return self

    def _predict_one(self, x):
        node = self.root_
        while not node.is_leaf:
            if x[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.prediction, node.proba

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if self.root_ is None:
            raise ValueError("Model not trained yet.")

        preds = [self._predict_one(x)[0] for x in X]
        return np.asarray(preds, dtype=np.int64)

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if self.root_ is None:
            raise ValueError("Model not trained yet.")

        prob1 = [self._predict_one(x)[1] for x in X]
        prob1 = np.asarray(prob1, dtype=np.float64)
        prob0 = 1.0 - prob1
        return np.column_stack([prob0, prob1])


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
        report[cls] = {"precision": precision, "recall": recall, "f1-score": f1, "support": int(support)}

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
    auc = 0.0
    for i in range(1, len(fpr)):
        auc += (fpr[i] - fpr[i - 1]) * (tpr[i] + tpr[i - 1]) / 2.0
    return float(auc)


def _resolve_data_path(path_candidates):
    for path in path_candidates:
        if os.path.exists(path):
            return path
    return None


def load_processed_data(train_path=None, test_path=None, target_column="loan_status"):
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


def run_decision_tree_experiment(
    train_path=None,
    test_path=None,
    target_column="loan_status",
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
):
    print("Decision Tree Experiment (From Scratch)")
    print(f"max_depth        : {max_depth}")
    print(f"min_samples_split: {min_samples_split}")
    print(f"min_samples_leaf : {min_samples_leaf}")

    X_train, y_train, X_test, y_test = load_processed_data(
        train_path=train_path,
        test_path=test_path,
        target_column=target_column,
    )

    model = DecisionTreeClassifierFromScratch(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
    )

    start_time = time.time()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    elapsed_time = time.time() - start_time

    tn, fp, fn, tp = confusion_matrix_binary(y_test, y_pred)
    accuracy = accuracy_score_manual(y_test, y_pred)
    precision = precision_score_manual(y_test, y_pred)
    recall = recall_score_manual(y_test, y_pred)
    f1 = f1_score_manual(y_test, y_pred)
    fpr, tpr = roc_curve_manual(y_test, y_proba)
    auc_value = auc_manual(fpr, tpr)

    print()
    print("Classification Report:")
    classification_report_binary_manual(y_test, y_pred)

    print()
    print("Confusion Matrix:")
    print(np.array([[tn, fp], [fn, tp]]))

    print()
    print(f"Accuracy : {accuracy:.6f}")
    print(f"Precision: {precision:.6f}")
    print(f"Recall   : {recall:.6f}")
    print(f"F1-score : {f1:.6f}")
    print(f"AUC      : {auc_value:.6f}")
    print(f"Time Cost: {elapsed_time:.6f} seconds")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "auc": auc_value,
        "time_cost": elapsed_time,
        "confusion_matrix": np.array([[tn, fp], [fn, tp]]),
    }


def get_best_decision_tree_predictions(train_path=None, test_path=None, target_column="loan_status"):
    X_train, y_train, X_test, y_test = load_processed_data(
        train_path=train_path,
        test_path=test_path,
        target_column=target_column,
    )
    model = DecisionTreeClassifierFromScratch(max_depth=10, min_samples_split=10, min_samples_leaf=5, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return y_test, y_pred, y_proba


if __name__ == "__main__":
    run_decision_tree_experiment()
