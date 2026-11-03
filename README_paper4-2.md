# Paper 4: PreLoad-Guard, URL-Only Phishing Prevention (Block / Warn / Allow)

PreLoad-Guard scores a URL before the page loads, using only the URL string (character 3-5-grams + 24 lexical features, fused by a Logistic Regression meta-learner), and decides:

- **Block** if the score is above a threshold that blocks at most 0.5% of legitimate training URLs
- **Warn** if it is above a threshold that flags at most 5% of legitimate training URLs
- **Allow** otherwise

## Datasets (not included)

| ID | Dataset | Source |
|----|---------|--------|
| D-2020 | Hannousse & Yahiouche (11,429 unique URLs) | https://data.mendeley.com/datasets/c2gw7fy2j4 (`dataset_B_05_2020.csv`) |
| D-2024 | PhiUSIIL (235,795 URLs; label 1 = legitimate, 0 = phishing) | https://archive.ics.uci.edu/dataset/967 (`PhiUSIIL_Phishing_URL_Dataset.csv`) |

Note: all legitimate URLs in D-2024 are https homepages without a path, so the 2020 → 2024 test uses domain names only (see the paper, Section 4.2).

## Run

```bash
pip install scikit-learn pandas numpy
# 5 repeated splits on D-2020 (full URLs)
python preload_guard.py within dataset_B_05_2020.csv hannousse results
# train on D-2020, test on D-2024 domains, then update with 1,000 new labelled domains
python preload_guard.py cross dataset_B_05_2020.csv PhiUSIIL_Phishing_URL_Dataset.csv hannousse2020_to_phiusiil2024 results
```

## Main results

| Test | PreLoad-Guard | Best baseline |
|------|---------------|---------------|
| D-2020: phishing missed (allowed) | 8.2% | 10.3% (LR, char n-grams) |
| D-2020: legitimate blocked | 0.37% | 0.42% |
| D-2020: ROC-AUC | 0.985 | 0.981 |
| Speed (median per URL, Python) | 38 ms | - |
| D-2024 domains: phishing missed, before → after update | 49.0% → 43.1% | 56.7% → 47.2% |

Files: `p4_within_hannousse.csv`, `p4_cross_hannousse2020_to_phiusiil2024*.csv` (3 runs), `p4_info_hannousse.json`.

Figure: `fig3_p4_prevention.png`: (a) share of phishing URLs blocked, warned and missed on 2020 data; (b) phishing domains missed on 2024 data before and after the update.
