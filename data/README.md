# Data policy

The original CSV and regenerated processed CSV files are intentionally excluded
from version control. Download `credit_risk_dataset.csv` from the original
Kaggle source and place it here:

```text
data/original_data/credit_risk_dataset.csv
```

Then generate the derived files locally:

```bash
python data/preprocess.py
```

Please review the data provider's current terms and license before downloading,
redistributing, or using the dataset outside this educational project.
