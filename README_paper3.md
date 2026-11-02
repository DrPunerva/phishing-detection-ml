# Paper 3: LFP-Stack, a Low-False-Positive Stacked Ensemble for Phishing Website Detection

LFP-Stack = stacked ensemble (Logistic Regression, SVM, Random Forest, Extra Trees, Gradient Boosting; Logistic Regression meta-learner on out-of-fold scores) + a decision threshold chosen on training data so that the false positive rate is at most 1%.

## Datasets (not included)

| ID | Dataset | Source |
|----|---------|--------|
| D1 | Hannousse & Yahiouche, 11,430 URLs, 87 features | https://data.mendeley.com/datasets/c2gw7fy2j4 (`dataset_B_05_2020.csv`) |
| D2 | Tan, 10,000 web pages, 48 features | https://data.mendeley.com/datasets/h3cgnj8hft/1 (`Phishing_Legitimate_full.arff`) |

Convert D2 to CSV first:

```bash
python arff_to_csv.py Phishing_Legitimate_full.arff tan_phishing_legitimate.csv
```

## Run

```bash
pip install scikit-learn pandas numpy scipy
python proposed_model_experiment.py dataset_B_05_2020.csv hannousse results
python proposed_model_experiment.py tan_phishing_legitimate.csv tan results
```

Each run uses five stratified 80/20 splits (seeds 0-4) and takes about 2-3 minutes.

## Main results (mean over 5 splits, FPR-constrained threshold)

| Dataset | Model | F1 | Recall | FPR | Precision at 1% phishing |
|---------|-------|----|--------|-----|--------------------------|
| D1 | LFP-Stack (proposed) | 0.9548 | 0.9211 | 0.0082 | 0.551 |
| D1 | Gradient Boosting | 0.9501 | 0.9132 | 0.0091 | 0.524 |
| D2 | LFP-Stack (proposed) | 0.9872 | 0.9824 | 0.0078 | 0.569 |
| D2 | Gradient Boosting | 0.9869 | 0.9826 | 0.0086 | 0.549 |

Full results: `results_hannousse.csv`, `results_tan.csv`. McNemar tests: `mcnemar_hannousse.csv`, `mcnemar_tan.csv`. Figure: `fig_p3_precision_1pct.png`.
