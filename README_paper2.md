# Paper 2: Detecting LLM-Generated Phishing Emails (Shortcut-Aware Evaluation)

Code and results for the paper *"Phishing in the Age of Large Language Models: A Focused Review and a Shortcut-Aware Evaluation of Detecting LLM-Generated Phishing Emails"*.

## Dataset (not included)

Download `cross-model-phishing.zip` from Zenodo and unzip it:

> R. Gutierrez, W. Villegas-Ch, J. Govea. *Cross-model evaluation of phishing detectors against LLM-generated emails: dataset, code and results.* Zenodo, 2026. DOI 10.5281/zenodo.20250116 (CC BY 4.0)

It contains 5,000 human-written and 4,986 LLM-generated phishing emails (GPT-4.1, DeepSeek 3.2, Llama 3.3 70B) with 17 stylometric features.

## Experiments

| ID | Test |
|----|------|
| E0 | Length-only baseline (characters + words) |
| E1 | 4 models x 2 feature sets (stylometric, TF-IDF), 80/20 split + 5-fold CV |
| E2 | Unseen LLM: train without one LLM, test on it |
| E3 | Artefact-reduced: formatting, placeholders, links, numbers removed; lengths matched |
| E4 | Unseen theme: train without one of five themes, test on it |

## How to run

```bash
pip install scikit-learn pandas numpy
python llm_phishing_experiment.py <unzipped-folder> results   # E0-E3
python e4_unseen_theme.py <unzipped-folder> results           # E4
```

`<unzipped-folder>` is the folder that contains `data/corpus_features.csv`. Seed 42 is used throughout.

## Key results

| Test | Result |
|------|--------|
| E0 length only | 93.44% accuracy |
| E1 TF-IDF models | F1 = 1.000 (all four) |
| E2 stylometric, Llama 3.3 unseen | F1 0.648-0.790; 34-52% of Llama emails missed |
| E3 TF-IDF after cleaning | F1 0.998-1.000 |
| E4 TF-IDF, unseen theme | F1 0.966-1.000 |

Full results: `results_paper2.csv` (E0-E3) and `results_paper2_e4.csv` (E4).

## Second dataset and cross-dataset test (Sections 4.6-4.7)

D2 is the phishing part of the corpus of Greco et al. (ITASEC 2024): 1,000 human-written and 1,000 WormGPT-generated phishing emails (501 + 1,000 after removing duplicates). A copy of the Kaggle files is in https://github.com/anguzz/human-vs-llm-phishing-analyzer (folder `dataset/`).

```bash
git clone https://github.com/anguzz/human-vs-llm-phishing-analyzer greco
python p2_second_dataset.py <unzipped-folder> greco results
```

| Test | Result |
|------|--------|
| D2 length only | 78.07% accuracy (64.91% after length matching) |
| D2 TF-IDF, raw / cleaned | F1 0.995-1.000 / 0.988-1.000 |
| Train D1 → test D2 (cleaned) | accuracy 58.6-65.1%; 69.9-82.4% of LLM emails missed |
| Train D2 → test D1 (cleaned) | accuracy 78.1-89.5%; 12.7-37.5% of LLM emails missed |

Files: `results_paper2_d2.csv`, `dataset_info_paper2_d2.json`.

## DistilBERT baseline (Google Colab)

`Paper2_BERT_Colab.ipynb` fine-tunes DistilBERT with exactly the same splits, cleaning and length matching (E1, E3, D2-E1, D2-E3, X1). Open it in Colab with a T4 GPU, run all, and upload `corpus_features.csv` when asked. It writes `results_paper2_bert.csv` (included).

| Test | DistilBERT |
|------|------------|
| E1 / E3 on D1 | F1 1.000 / 1.000 |
| D2-E1 / D2-E3 | F1 0.998 / 0.994 |
| Train D1 → test D2 | accuracy 59.5%, 81.0% of LLM emails missed |
| Train D2 → test D1 | accuracy 44.5%, 96.9% of LLM emails missed |
