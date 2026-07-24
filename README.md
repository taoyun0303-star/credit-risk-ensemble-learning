# Credit Risk Ensemble Learning From Scratch

[Quick start](#quick-start) · [Recorded result](docs/RECORDED_RESULTS.md)

An educational Python project that builds a loan-default classifier with a
weighted probability-averaging ensemble. The final model combines four
handwritten implementations: Random Forest, Decision Tree, KNN, and Logistic
Regression. It is intended for learning reproducible tabular preprocessing and
ensemble learning, not for real lending decisions or for claiming a new
research result.

## What this project demonstrates

- Stratified splitting, outlier filtering, missing-value handling, one-hot
  encoding, IQR clipping, and training-only Min-Max scaling.
- Four transparent base learners implemented with NumPy and Pandas rather than
  packaged machine-learning estimators.
- Weighted probability averaging, binary classification metrics, and a
  reproducible report artifact.
- Small built-in regression tests for metric functions, ensemble configuration,
  and feature-schema alignment.

## Project layout

```text
credit-risk-ensemble-learning/
├── combined_models/main.py     # Train and evaluate the final four-model ensemble
├── data/preprocess.py          # Leakage-aware preprocessing pipeline
├── data/README.md              # Data download and redistribution policy
├── single_models/              # From-scratch base learners
├── tests/test_core.py          # Dependency-free regression tests
├── docs/RECORDED_RESULTS.md    # Course-project reference metrics
└── reports/generated/          # Local output; ignored by Git
```

## Quick start

Python 3.10 or later is recommended.

```bash
python -m pip install -r requirements.txt
```

1. Obtain the source CSV yourself and put it at
   `data/original_data/credit_risk_dataset.csv`. The data is excluded from this
   repository; read `data/README.md` first.
2. Generate the processed training and test sets:

   ```bash
   python data/preprocess.py
   ```

3. Run the final ensemble:

   ```bash
   python combined_models/main.py
   ```

4. Run the checks:

   ```bash
   python -m unittest discover -s tests -v
   ```

The model writes its Markdown and JSON outputs to `reports/generated/`.
Install `matplotlib` separately only when you want to call the optional KNN ROC
plotting helper.

## Final ensemble

The probability-averaging weights are fixed and checked at runtime:

| Base model | Weight |
| --- | ---: |
| Random Forest | 0.80 |
| Decision Tree | 0.05 |
| KNN | 0.05 |
| Logistic Regression | 0.10 |

The default classification threshold is `0.50`. See
[`docs/RECORDED_RESULTS.md`](docs/RECORDED_RESULTS.md) for the original
course-project result and its limitations.

## Scope and limitations

- The code is educational. It is not a production credit-scoring system.
- The project uses one public dataset and one fixed split, so its reported
  metrics do not establish general real-world performance.
- The output is a risk score for learning purposes only. Production lending
  requires calibration, cost analysis, fairness assessment, monitoring,
  governance, and domain validation.
