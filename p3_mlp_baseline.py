"""
Paper 3 extension: a neural-network baseline (multi-layer perceptron, MLP) for LFP-Stack.

The MLP is compared with LFP-Stack under the same protocol as the main experiment:
the same five stratified 80/20 splits (seeds 0-4), the same FPR-constrained threshold
rule (set on 5-fold out-of-fold training scores, target FPR = 1%), and an exact
McNemar test on paired test predictions. LFP-Stack itself is unchanged (the MLP is
NOT added to the stack).

Usage: python p3_mlp_baseline.py <dataset.csv> <dataset_name> [output_dir]
Needs proposed_model_experiment.py in the same folder.
"""
import sys, os, time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from proposed_model_experiment import (load, base_models, score, threshold_for_fpr, evaluate,
                                       mcnemar, TARGET_FPR, SEEDS)


def mlp(seed):
    return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-4,
                                                         early_stopping=True, max_iter=300,
                                                         random_state=seed))


def run_seed(X, y, seed):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    names = list(base_models(seed))
    oof = np.zeros((len(y_tr), len(names))); oof_mlp = np.zeros(len(y_tr))
    for tr_i, va_i in skf.split(X_tr, y_tr):
        for j, (n, m) in enumerate(base_models(seed).items()):
            m.fit(X_tr.iloc[tr_i], y_tr[tr_i]); oof[va_i, j] = score(m, X_tr.iloc[va_i])
        m = mlp(seed).fit(X_tr.iloc[tr_i], y_tr[tr_i]); oof_mlp[va_i] = m.predict_proba(X_tr.iloc[va_i])[:, 1]
    test = np.zeros((len(y_te), len(names)))
    for j, (n, m) in enumerate(base_models(seed).items()):
        m.fit(X_tr, y_tr); test[:, j] = score(m, X_te)
    t0 = time.time(); m = mlp(seed).fit(X_tr, y_tr); fit_s = time.time() - t0
    s_mlp = m.predict_proba(X_te)[:, 1]

    rows = []
    r, _ = evaluate(y_te, s_mlp, 0.5)
    rows.append({"Seed": seed, "Model": "MLP (neural network)", "Threshold": "default", **r, "Fit time (s)": fit_s})
    r, p_mlp = evaluate(y_te, s_mlp, threshold_for_fpr(oof_mlp, y_tr, TARGET_FPR))
    rows.append({"Seed": seed, "Model": "MLP (neural network)", "Threshold": "FPR-constrained", **r, "Fit time (s)": fit_s})

    # LFP-Stack, exactly as in proposed_model_experiment.py
    meta = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, random_state=seed))
    meta_oof = np.zeros(len(y_tr))
    for tr_i, va_i in skf.split(oof, y_tr):
        meta.fit(oof[tr_i], y_tr[tr_i]); meta_oof[va_i] = meta.predict_proba(oof[va_i])[:, 1]
    meta.fit(oof, y_tr)
    r, p_stack = evaluate(y_te, meta.predict_proba(test)[:, 1], threshold_for_fpr(meta_oof, y_tr, TARGET_FPR))
    rows.append({"Seed": seed, "Model": "LFP-Stack (proposed)", "Threshold": "FPR-constrained", **r})
    b, c, pv = mcnemar(y_te, p_stack, p_mlp)
    test_row = {"Seed": seed, "Compared with": "MLP (FPR-constrained)", "Proposed right, baseline wrong": b,
                "Proposed wrong, baseline right": c, "McNemar p": pv}
    return rows, test_row


def main():
    path, name = sys.argv[1], sys.argv[2]
    out = sys.argv[3] if len(sys.argv) > 3 else "results"
    os.makedirs(out, exist_ok=True)
    X, y, _ = load(path)
    rows, tests = [], []
    for seed in SEEDS:
        t0 = time.time(); r, t = run_seed(X, y, seed); rows += r; tests.append(t)
        print(f"seed {seed} done in {time.time()-t0:.0f}s", flush=True)
    res = pd.DataFrame(rows); res.insert(0, "Dataset", name)
    tst = pd.DataFrame(tests); tst.insert(0, "Dataset", name)
    res.to_csv(os.path.join(out, f"p3_mlp_{name}.csv"), index=False)
    tst.to_csv(os.path.join(out, f"p3_mlp_mcnemar_{name}.csv"), index=False)
    cols = ["Accuracy", "F1", "FPR", "FNR", "Recall", "ROC-AUC", "Precision @ 1% phishing"]
    pd.set_option("display.width", 250)
    print(res.groupby(["Model", "Threshold"])[cols].agg(["mean", "std"]).round(4).to_string())
    print(tst.to_string())


if __name__ == "__main__":
    main()
