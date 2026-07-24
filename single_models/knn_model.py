# knn_model.py
import os
import time
import numpy as np
import pandas as pd


class KNNClassifier:
    # K-nearest neighbors classifier implemented from scratch.
    # Default values match the current tuned configuration.

    def __init__(self, k=9, distance_metric="manhattan", weights="distance"):
        if not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer.")

        if distance_metric not in ["euclidean", "manhattan"]:
            raise ValueError("distance_metric must be 'euclidean' or 'manhattan'.")

        if weights not in ["uniform", "distance"]:
            raise ValueError("weights must be 'uniform' or 'distance'.")

        self.k = k
        self.distance_metric = distance_metric
        self.weights = weights

        self.X_train = None
        self.y_train = None
        self.classes_ = None

    def fit(self, X, y):
        # KNN stores the training data instead of learning parameters.
        X_array = np.asarray(X, dtype=np.float64)
        y_array = np.asarray(y)

        if X_array.ndim != 2:
            raise ValueError("X must be a 2D array.")

        if y_array.ndim != 1:
            raise ValueError("y must be a 1D array.")

        if len(X_array) != len(y_array):
            raise ValueError("X and y must have the same number of samples.")

        self.X_train = X_array
        self.y_train = y_array
        self.classes_ = np.unique(y_array)

        return self

    def _compute_distances_one_sample(self, x):
        # Compute distances from one sample to all stored training samples.
        if self.distance_metric == "euclidean":
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
        else:
            distances = np.sum(np.abs(self.X_train - x), axis=1)

        return distances

    def _predict_one_sample(self, x):
        # Return both the predicted label and positive-class probability.
        distances = self._compute_distances_one_sample(x)

        k_effective = min(self.k, len(distances))
        neighbor_indices = np.argsort(distances)[:k_effective]

        neighbor_labels = self.y_train[neighbor_indices]
        neighbor_distances = distances[neighbor_indices]

        if self.weights == "uniform":
            class_scores = {}
            for label in neighbor_labels:
                class_scores[label] = class_scores.get(label, 0.0) + 1.0
        else:
            class_scores = {}
            for label, dist in zip(neighbor_labels, neighbor_distances):
                if dist == 0:
                    weight = 1e12
                else:
                    weight = 1.0 / (dist + 1e-12)
                class_scores[label] = class_scores.get(label, 0.0) + weight

        predicted_label = max(class_scores.items(), key=lambda item: item[1])[0]

        total_score = sum(class_scores.values())

        if 1 in class_scores:
            prob_class_1 = class_scores[1] / total_score
        else:
            prob_class_1 = 0.0

        return predicted_label, prob_class_1

    def predict(self, X):
        X_array = np.asarray(X, dtype=np.float64)

        if X_array.ndim != 2:
            raise ValueError("X must be a 2D array.")

        predictions = []
        for i in range(X_array.shape[0]):
            pred_label, _ = self._predict_one_sample(X_array[i])
            predictions.append(pred_label)

        return np.asarray(predictions)

    def predict_proba(self, X):
        # Return class probabilities with shape (n_samples, 2).
        X_array = np.asarray(X, dtype=np.float64)

        if X_array.ndim != 2:
            raise ValueError("X must be a 2D array.")

        probas = []
        for i in range(X_array.shape[0]):
            _, prob_1 = self._predict_one_sample(X_array[i])
            prob_0 = 1.0 - prob_1
            probas.append([prob_0, prob_1])

        return np.asarray(probas)


def confusion_matrix_binary(y_true, y_pred):
    # Binary confusion matrix values returned as TN, FP, FN, TP.
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
    tn, fp, fn, tp = confusion_matrix_binary(y_true, y_pred)
    denominator = tp + fp
    if denominator == 0:
        return 0.0
    return tp / denominator


def recall_score_manual(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix_binary(y_true, y_pred)
    denominator = tp + fn
    if denominator == 0:
        return 0.0
    return tp / denominator


def f1_score_manual(y_true, y_pred):
    precision = precision_score_manual(y_true, y_pred)
    recall = recall_score_manual(y_true, y_pred)

    denominator = precision + recall
    if denominator == 0:
        return 0.0

    return 2.0 * precision * recall / denominator


def classification_report_binary_manual(y_true, y_pred):
    # Print a compact binary classification report without sklearn.
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
            "support": int(support)
        }

    macro_precision = (report[0]["precision"] + report[1]["precision"]) / 2.0
    macro_recall = (report[0]["recall"] + report[1]["recall"]) / 2.0
    macro_f1 = (report[0]["f1-score"] + report[1]["f1-score"]) / 2.0
    total_support = report[0]["support"] + report[1]["support"]

    weighted_precision = (
        report[0]["precision"] * report[0]["support"] +
        report[1]["precision"] * report[1]["support"]
    ) / total_support

    weighted_recall = (
        report[0]["recall"] * report[0]["support"] +
        report[1]["recall"] * report[1]["support"]
    ) / total_support

    weighted_f1 = (
        report[0]["f1-score"] * report[0]["support"] +
        report[1]["f1-score"] * report[1]["support"]
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
    # Build ROC points by sweeping unique score thresholds.
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    sorted_indices = np.argsort(-y_score)
    y_true_sorted = y_true[sorted_indices]
    y_score_sorted = y_score[sorted_indices]

    thresholds = np.unique(y_score_sorted)[::-1]

    tpr_list = [0.0]
    fpr_list = [0.0]
    threshold_list = [np.inf]

    P = np.sum(y_true == 1)
    N = np.sum(y_true == 0)

    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix_binary(y_true, y_pred)

        tpr = tp / P if P > 0 else 0.0
        fpr = fp / N if N > 0 else 0.0

        tpr_list.append(tpr)
        fpr_list.append(fpr)
        threshold_list.append(threshold)

    tpr_list.append(1.0)
    fpr_list.append(1.0)
    threshold_list.append(-np.inf)

    return np.asarray(fpr_list), np.asarray(tpr_list), np.asarray(threshold_list)


def auc_manual(fpr, tpr):
    # Integrate the ROC curve with the trapezoidal rule.
    sorted_indices = np.argsort(fpr)
    fpr_sorted = fpr[sorted_indices]
    tpr_sorted = tpr[sorted_indices]

    auc_value = 0.0
    for i in range(1, len(fpr_sorted)):
        x1 = fpr_sorted[i - 1]
        x2 = fpr_sorted[i]
        y1 = tpr_sorted[i - 1]
        y2 = tpr_sorted[i]

        auc_value += (x2 - x1) * (y1 + y2) / 2.0

    return auc_value


def plot_roc_curve_manual(y_true, y_score, title="ROC Curve"):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "ROC plotting is optional. Install matplotlib to use plot_roc_curve_manual()."
        ) from exc

    fpr, tpr, thresholds = roc_curve_manual(y_true, y_score)
    auc_value = auc_manual(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc_value:.6f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

    return fpr, tpr, thresholds, auc_value


def _resolve_data_path(path_candidates):
    for path in path_candidates:
        if os.path.exists(path):
            return path
    return None


def load_processed_data(
    train_path=None,
    test_path=None,
    target_column="loan_status"
):
    # Load the preprocessed train and test splits used by all model demos.
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

    if target_column not in train_df.columns:
        raise ValueError(f"Target column '{target_column}' not found in train data.")

    if target_column not in test_df.columns:
        raise ValueError(f"Target column '{target_column}' not found in test data.")

    X_train = train_df.drop(columns=[target_column]).values
    y_train = train_df[target_column].values

    X_test = test_df.drop(columns=[target_column]).values
    y_test = test_df[target_column].values

    return X_train, y_train, X_test, y_test


def run_knn_experiment(
    k=9,
    distance_metric="manhattan",
    weights="distance",
    train_path=None,
    test_path=None,
    target_column="loan_status",
    plot_roc=True
):
    print("KNN Experiment")
    print(f"k               : {k}")
    print(f"distance_metric : {distance_metric}")
    print(f"weights         : {weights}")

    X_train, y_train, X_test, y_test = load_processed_data(
        train_path=train_path,
        test_path=test_path,
        target_column=target_column
    )

    print("Train shape:", X_train.shape)
    print("Test shape :", X_test.shape)

    model = KNNClassifier(
        k=k,
        distance_metric=distance_metric,
        weights=weights
    )

    start_time = time.time()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    end_time = time.time()
    elapsed_time = end_time - start_time

    accuracy = accuracy_score_manual(y_test, y_pred)
    precision = precision_score_manual(y_test, y_pred)
    recall = recall_score_manual(y_test, y_pred)
    f1 = f1_score_manual(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix_binary(y_test, y_pred)

    fpr, tpr, thresholds = roc_curve_manual(y_test, y_proba)
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

    if plot_roc:
        plot_roc_curve_manual(
            y_true=y_test,
            y_score=y_proba,
            title=f"ROC Curve - KNN (k={k}, metric={distance_metric}, weights={weights})"
        )

    return {
        "k": k,
        "distance_metric": distance_metric,
        "weights": weights,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "auc": auc_value,
        "time_cost": elapsed_time,
        "confusion_matrix": np.array([[tn, fp], [fn, tp]])
    }


def get_best_knn_model(train_path=None,
                       test_path=None,
                       target_column="loan_status"):
    # Train the tuned KNN model for use by the ensemble script.
    X_train, y_train, X_test, y_test = load_processed_data(
        train_path=train_path,
        test_path=test_path,
        target_column=target_column
    )

    model = KNNClassifier(
        k=9,
        distance_metric="manhattan",
        weights="distance"
    )
    model.fit(X_train, y_train)

    return model


def get_best_knn_predictions(train_path=None,
                             test_path=None,
                             target_column="loan_status"):
    # Return test labels, predictions, and probabilities for the tuned KNN.
    X_train, y_train, X_test, y_test = load_processed_data(
        train_path=train_path,
        test_path=test_path,
        target_column=target_column
    )

    model = KNNClassifier(
        k=9,
        distance_metric="manhattan",
        weights="distance"
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return y_test, y_pred, y_proba


if __name__ == "__main__":
    run_knn_experiment(
        k=9,
        distance_metric="manhattan",
        weights="distance",
        train_path=None,
        test_path=None,
        target_column="loan_status",
        plot_roc=False
    )
