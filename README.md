# Machine Learning for Phishing Detection and Prevention: PhD Research Code

This repository contains the code and results for four related papers. Each paper has its own README.

| Paper | Topic | README |
|-------|-------|--------|
| 1 | Review + comparison of 4 ML models for phishing website detection (beyond accuracy) | this file (below) |
| 2 | Review + shortcut-aware evaluation of detecting LLM-generated phishing emails | `README_paper2.md` |
| 3 | LFP-Stack: low-false-positive stacked ensemble, two datasets | `README_paper3.md` |
| 4 | PreLoad-Guard: URL-only phishing prevention (block / warn / allow), 2020 → 2024 test | `README_paper4.md` |

Datasets are not included; each README links to the original public source.

---

## Paper 1: Phishing Website Detection, Comparing ML Models Beyond Accuracy

Code and results for the paper *"Machine Learning for Phishing Website Detection: A Focused Review and a Comparative Evaluation Beyond Accuracy"*.

We compare four classifiers (Logistic Regression, SVM, Random Forest, Gradient Boosting) on a public benchmark of 11,430 URLs and report accuracy, precision, recall, F1, false positive rate (FPR), false negative rate (FNR) and ROC-AUC.

## Experiments

| ID | What it tests |
|----|---------------|
| E1 | All four models on all 87 features |
| E2 | Same models using only the 56 URL-based features |
| E3 | Precision when phishing is only 5% or 1% of traffic |

## Main results (test set, 2,286 URLs)

| Model | Accuracy | F1 | FPR | FNR | Precision at 1% phishing |
|-------|----------|----|-----|-----|--------------------------|
| Gradient Boosting | 0.9650 | 0.9652 | 0.0420 | 0.0280 | 0.189 |
| Random Forest | 0.9619 | 0.9621 | 0.0420 | 0.0341 | 0.189 |
| SVM (RBF) | 0.9541 | 0.9539 | 0.0420 | 0.0499 | 0.186 |
| Logistic Regression | 0.9361 | 0.9359 | 0.0604 | 0.0674 | 0.135 |

Full results for all experiments are in `results.csv`.

## Dataset

This repository does **not** include the dataset. Download it from the original source:

> A. Hannousse, S. Yahiouche. *Web page phishing detection*. Mendeley Data, 2021. https://data.mendeley.com/datasets/c2gw7fy2j4

Use the file `dataset_B_05_2020.csv` (11,430 rows, 87 features, label column `status`).

## How to run

**Option 1: Google Colab (no installation)**
1. Open `Phishing_Model_Comparison.ipynb` in Google Colab.
2. Click *Runtime → Run all* and upload `dataset_B_05_2020.csv` when asked.

**Option 2: Your own computer**
```bash
pip install -r requirements.txt
python phishing_experiment.py dataset_B_05_2020.csv results
```

Running takes a few minutes on a normal laptop. All randomness uses seed 42, so results should match `results.csv` (small differences can occur with other library versions).

## Files

| File | Description |
|------|-------------|
| `phishing_experiment.py` | Full experiment script |
| `Phishing_Model_Comparison.ipynb` | Same experiment as a Colab notebook |
| `results.csv` | All metrics for all models and experiments |
| `dataset_info.json` | Dataset size and feature groups used |
| `fig1_f1_fpr.png` | Figure 1: F1 and FPR, all vs URL-only features |
| `fig2_base_rate.png` | Figure 2: precision at 50%, 5% and 1% phishing |

## Citation

If you use this code, please cite our paper (details will be added after publication) and the dataset paper:

> A. Hannousse, S. Yahiouche. Towards benchmark datasets for machine learning based website phishing detection: An experimental study. *Engineering Applications of Artificial Intelligence* 104:104347, 2021.
