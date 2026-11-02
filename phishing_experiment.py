"""
Phishing website detection: comparison of 4 ML models on the
Hannousse & Yahiouche (2021) benchmark dataset (11,430 URLs, 87 features).

Experiments
  E1  Standard comparison: stratified 80/20 split + 5-fold CV on the training part.
  E2  Feature-group study: URL-only features vs all features
      (external-service features need WHOIS / web-traffic lookups that are slow
       or unavailable at detection time).
  E3  Realistic base rate: precision when phishing is only 1% or 5% of traffic,
      computed from each model's measured TPR and FPR on the test set.

Usage:  python phishing_experiment.py dataset_phishing.csv  [output_dir]
"""
import sys, time, json, os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix)

SEED = 42

# Features that need third-party services (Hannousse & Yahiouche, 2021: 7 "external" features)
EXTERNAL = ["whois_registered_domain", "domain_registration_length", "domain_age",
            "web_traffic", "dns_record", "google_index", "page_rank"]


def load(path):
    df = pd.read_csv(path)
    label_col = "status" if "status" in df.columns else df.columns[-1]
    y = (df[label_col].astype(str).str.lower().str.strip() == "phishing").astype(int).values
    X = df.drop(columns=[c for c in ["url", label_col] if c in df.columns])
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
    return X, y


def models():
    return {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, random_state=SEED)),
        "SVM (RBF)": make_pipeline(StandardScaler(), SVC(kernel="rbf", random_state=SEED)),
        "Random Forest": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED),
        "Gradient Boosting": HistGradientBoostingClassifier(random_state=SEED),
    }


def metrics(y_true, y_pred, y_score):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "FPR": fp / (fp + tn),
        "FNR": fn / (fn + tp),
        "ROC-AUC": roc_auc_score(y_true, y_score),
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
    }


def precision_at_base_rate(tpr, fpr, p):
    """Precision when a fraction p of all inputs is phishing."""
    denom = tpr * p + fpr * (1 - p)
    return tpr * p / denom if denom > 0 else float("nan")


def run(X, y, tag, out):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows = []
    for name, m in models().items():
        cvres = cross_validate(m, X_tr, y_tr, cv=cv, scoring=["accuracy", "f1"], n_jobs=1)
        t0 = time.time(); m.fit(X_tr, y_tr); train_s = time.time() - t0
        # ROC-AUC needs a score; SVM gives a decision value instead of a probability
        score = m.predict_proba(X_te)[:, 1] if hasattr(m, "predict_proba") else m.decision_function(X_te)
        pred = m.predict(X_te)
        r = {"Features": tag, "Model": name, "n_features": X.shape[1],
             "CV Accuracy mean": cvres["test_accuracy"].mean(), "CV Accuracy std": cvres["test_accuracy"].std(),
             "CV F1 mean": cvres["test_f1"].mean(), "CV F1 std": cvres["test_f1"].std(),
             "Train time (s)": train_s}
        r.update(metrics(y_te, pred, score))
        tpr = 1 - r["FNR"]
        for p in (0.05, 0.01):
            r[f"Precision @ {int(p*100)}% phishing"] = precision_at_base_rate(tpr, r["FPR"], p)
        rows.append(r)
        print(f"[{tag}] {name}: Acc={r['Accuracy']:.4f} F1={r['F1']:.4f} FPR={r['FPR']:.4f} FNR={r['FNR']:.4f}")
    return rows, len(y_te)


def main():
    path = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "results"
    os.makedirs(out, exist_ok=True)
    X, y = load(path)
    info = {"rows": int(len(y)), "phishing": int(y.sum()), "legitimate": int((1 - y).sum()),
            "features_total": int(X.shape[1])}
    ext = [c for c in EXTERNAL if c in X.columns]
    # URL-structure features come first in the dataset (56 of them, per the paper).
    url_cols = list(X.columns[:56])
    info["external_features_found"] = ext
    info["url_feature_range"] = [url_cols[0], url_cols[-1]]
    print(json.dumps(info, indent=2))

    rows, n_test = run(X, y, "All features", out)
    rows2, _ = run(X[url_cols], y, "URL-only features", out)
    info["test_size"] = n_test
    res = pd.DataFrame(rows + rows2)
    res.to_csv(f"{out}/results.csv", index=False)
    with open(f"{out}/dataset_info.json", "w") as f:
        json.dump(info, f, indent=2)
    print(res.round(4).to_string())


if __name__ == "__main__":
    main()
