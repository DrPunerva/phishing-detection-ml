"""
Paper 3: A low-false-positive stacked ensemble for phishing website detection.

Proposed method (LFP-Stack)
  1. Base learners: Logistic Regression, SVM (RBF), Random Forest, Extra Trees,
     Gradient Boosting (histogram-based).
  2. Stacking: a Logistic Regression meta-learner trained on the base learners'
     out-of-fold (5-fold) scores from the training set only.
  3. False-positive-constrained threshold: instead of the default 0.5, the decision
     threshold is set on out-of-fold training scores so that at most 1% of legitimate
     training sites would be flagged (target FPR = 1%).

Baselines (as in Paper 1): LR, SVM, RF, GB at the default threshold, and the same four
models with the SAME FPR-constrained threshold (a fair comparison that isolates the
benefit of stacking from the benefit of threshold tuning).

Protocol: 5 repetitions of a stratified 80/20 split (seeds 0-4). Metrics: accuracy,
precision, recall, F1, FPR, FNR, ROC-AUC, precision at 5% and 1% phishing, McNemar's test.

Usage: python proposed_model_experiment.py <dataset.csv> <dataset_name> [output_dir]
"""
import sys, os, json, time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix
from scipy.stats import binomtest

TARGET_FPR = 0.01
SEEDS = [0, 1, 2, 3, 4]


def load(path):
    df = pd.read_csv(path)
    for lab in ("status", "CLASS_LABEL", "class", "label", "Result"):
        if lab in df.columns:
            break
    else:
        lab = df.columns[-1]
    raw = df[lab]
    if not pd.api.types.is_numeric_dtype(raw):
        y = (raw.astype(str).str.lower().str.strip() == "phishing").astype(int).values
    else:
        # numeric labels: 1 = phishing (Tan dataset); -1/1 style handled as 1 = phishing
        y = (raw.values == 1).astype(int)
    X = df.drop(columns=[c for c in ("url", "id", "index", lab) if c in df.columns])
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
    return X, y, lab


def base_models(seed):
    return {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, random_state=seed)),
        "SVM": make_pipeline(StandardScaler(), SVC(kernel="rbf", random_state=seed)),
        "Random Forest": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=seed),
        "Extra Trees": ExtraTreesClassifier(n_estimators=300, n_jobs=-1, random_state=seed),
        "Gradient Boosting": HistGradientBoostingClassifier(random_state=seed),
    }


BASELINES = ["Logistic Regression", "SVM", "Random Forest", "Gradient Boosting"]


def score(m, X):
    return m.predict_proba(X)[:, 1] if hasattr(m, "predict_proba") else m.decision_function(X)


def threshold_for_fpr(scores, y, target):
    """Lowest threshold whose FPR on legitimate examples is <= target."""
    neg = np.sort(scores[y == 0])
    k = int(np.floor(target * len(neg)))          # how many negatives may be above threshold
    idx = len(neg) - k - 1
    return np.nextafter(neg[idx], np.inf) if idx >= 0 else -np.inf


def evaluate(y, s, thr):
    pred = (s >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    tpr, fpr = tp / (tp + fn), fp / (fp + tn)
    prec = tp / (tp + fp) if tp + fp else 0.0
    out = {"Accuracy": (tp + tn) / len(y), "Precision": prec, "Recall": tpr,
           "F1": 2 * prec * tpr / (prec + tpr) if prec + tpr else 0.0,
           "FPR": fpr, "FNR": 1 - tpr, "ROC-AUC": roc_auc_score(y, s),
           "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn)}
    for p in (0.05, 0.01):
        d = tpr * p + fpr * (1 - p)
        out[f"Precision @ {int(p*100)}% phishing"] = tpr * p / d if d else float("nan")
    return out, pred


def mcnemar(y, pa, pb):
    """Exact McNemar test on paired predictions."""
    a_right, b_right = pa == y, pb == y
    b = int(np.sum(a_right & ~b_right)); c = int(np.sum(~a_right & b_right))
    p = binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0
    return b, c, p


def run_seed(X, y, seed):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    names = list(base_models(seed))
    oof = np.zeros((len(y_tr), len(names)))
    for tr_i, va_i in skf.split(X_tr, y_tr):
        for j, (n, m) in enumerate(base_models(seed).items()):
            m.fit(X_tr.iloc[tr_i], y_tr[tr_i])
            oof[va_i, j] = score(m, X_tr.iloc[va_i])
    test = np.zeros((len(y_te), len(names)))
    for j, (n, m) in enumerate(base_models(seed).items()):
        m.fit(X_tr, y_tr)
        test[:, j] = score(m, X_te)

    rows, preds = [], {}
    for j, n in enumerate(names):
        if n not in BASELINES:
            continue
        default_thr = 0.5 if n != "SVM" else 0.0
        r, p = evaluate(y_te, test[:, j], default_thr)
        rows.append({"Seed": seed, "Model": n, "Threshold": "default", **r}); preds[(n, "default")] = p
        thr = threshold_for_fpr(oof[:, j], y_tr, TARGET_FPR)
        r, p = evaluate(y_te, test[:, j], thr)
        rows.append({"Seed": seed, "Model": n, "Threshold": "FPR-constrained", **r}); preds[(n, "cal")] = p

    # Proposed: stacking meta-learner on OOF scores
    meta = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, random_state=seed))
    meta_oof = np.zeros(len(y_tr))
    for tr_i, va_i in skf.split(oof, y_tr):
        meta.fit(oof[tr_i], y_tr[tr_i]); meta_oof[va_i] = meta.predict_proba(oof[va_i])[:, 1]
    meta.fit(oof, y_tr)
    s_te = meta.predict_proba(test)[:, 1]
    r, p = evaluate(y_te, s_te, 0.5)
    rows.append({"Seed": seed, "Model": "LFP-Stack (proposed)", "Threshold": "default", **r})
    thr = threshold_for_fpr(meta_oof, y_tr, TARGET_FPR)
    r, p = evaluate(y_te, s_te, thr)
    rows.append({"Seed": seed, "Model": "LFP-Stack (proposed)", "Threshold": "FPR-constrained", **r})
    preds[("LFP-Stack (proposed)", "cal")] = p

    tests = []
    for n in BASELINES:
        b, c, pv = mcnemar(y_te, preds[("LFP-Stack (proposed)", "cal")], preds[(n, "cal")])
        tests.append({"Seed": seed, "Compared with": n + " (FPR-constrained)",
                      "Proposed right, baseline wrong": b, "Proposed wrong, baseline right": c, "McNemar p": pv})
    return rows, tests, len(y_te)


def main():
    path, name = sys.argv[1], sys.argv[2]
    out = sys.argv[3] if len(sys.argv) > 3 else "results"
    os.makedirs(out, exist_ok=True)
    X, y, lab = load(path)
    info = {"dataset": name, "rows": int(len(y)), "phishing": int(y.sum()), "legitimate": int((1 - y).sum()),
            "features": int(X.shape[1]), "label_column": lab, "target_fpr": TARGET_FPR, "seeds": SEEDS}
    print(json.dumps(info), flush=True)
    rows, tests = [], []
    for seed in SEEDS:
        t0 = time.time()
        r, t, n_te = run_seed(X, y, seed)
        rows += r; tests += t
        info["test_size"] = n_te
        print(f"seed {seed} done in {time.time()-t0:.0f}s", flush=True)
    res = pd.DataFrame(rows); res.insert(0, "Dataset", name)
    tst = pd.DataFrame(tests); tst.insert(0, "Dataset", name)
    res.to_csv(os.path.join(out, f"results_{name}.csv"), index=False)
    tst.to_csv(os.path.join(out, f"mcnemar_{name}.csv"), index=False)
    with open(os.path.join(out, f"info_{name}.json"), "w") as f:
        json.dump(info, f, indent=2)
    cols = ["Accuracy", "F1", "FPR", "FNR", "Recall", "ROC-AUC", "Precision @ 1% phishing"]
    summ = res.groupby(["Model", "Threshold"])[cols].agg(["mean", "std"]).round(4)
    summ.to_csv(os.path.join(out, f"summary_{name}.csv"))
    pd.set_option("display.width", 250)
    print(res.groupby(["Model", "Threshold"])[cols].mean().round(4).to_string())
    print(tst.groupby("Compared with")[["Proposed right, baseline wrong", "Proposed wrong, baseline right", "McNemar p"]].agg(["mean", "max"]).round(4))


if __name__ == "__main__":
    main()
