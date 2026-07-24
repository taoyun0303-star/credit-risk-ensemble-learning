# Recorded baseline result

This repository records the final course-project run for reference, rather than
claiming a new research benchmark. The values below were produced by the fixed
four-model probability-averaging configuration on the project split.

| Metric | Recorded value |
| --- | ---: |
| Accuracy | 0.9331 |
| Precision | 0.9852 |
| Recall | 0.7039 |
| F1-score | 0.8212 |
| AUC | 0.9297 |

The classification threshold is `0.50`; the confusion matrix is
`[[5080, 15], [421, 1001]]` in `[[TN, FP], [FN, TP]]` order.

These values are a reproducibility target, not an assertion of general
credit-risk performance. Results can vary when data versions, split policies,
or model settings change.
