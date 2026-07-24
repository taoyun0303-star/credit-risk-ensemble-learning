"""Small dependency-free regression tests for the public teaching project."""

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from combined_models.main import auc_score, confusion_matrix_binary, validate_ensemble_config
from data.preprocess import TARGET_COL, align_feature_columns, one_hot_encode


class EnsembleMetricTests(unittest.TestCase):
    def test_perfect_auc_and_confusion_matrix(self):
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.2, 0.8, 0.9])
        y_pred = (y_score >= 0.5).astype(int)

        self.assertAlmostEqual(auc_score(y_true, y_score), 1.0)
        self.assertEqual(confusion_matrix_binary(y_true, y_pred), (2, 0, 0, 2))

    def test_tied_scores_have_neutral_auc(self):
        self.assertAlmostEqual(auc_score([0, 1], [0.5, 0.5]), 0.5)

    def test_fixed_ensemble_configuration_is_valid(self):
        self.assertIsNone(validate_ensemble_config())


class PreprocessingTests(unittest.TestCase):
    def test_independent_one_hot_encoding_is_aligned(self):
        train = pd.DataFrame(
            {
                "person_home_ownership": ["RENT", "OWN"],
                "loan_intent": ["MEDICAL", "VENTURE"],
                TARGET_COL: [0, 1],
            }
        )
        test = pd.DataFrame(
            {
                "person_home_ownership": ["RENT", "MORTGAGE"],
                "loan_intent": ["MEDICAL", "EDUCATION"],
                TARGET_COL: [1, 0],
            }
        )

        train_encoded = one_hot_encode(train)
        test_encoded = one_hot_encode(test)
        train_aligned, test_aligned = align_feature_columns(train_encoded, test_encoded)

        self.assertEqual(list(train_aligned.columns), list(test_aligned.columns))
        self.assertEqual(test_aligned[TARGET_COL].tolist(), [1, 0])
        self.assertEqual(test_aligned.loc[1, "home_ownership_OWN"], 0)


if __name__ == "__main__":
    unittest.main()
