"""
Paper 1 extensions.

  E1+ Neural-network (MLP) baseline on the Hannousse data (all 87 features), same split as E1.
  E4  Second dataset: Tan (2018), 10,000 web pages, 48 features; the four models + MLP,
      same protocol (stratified 80/20 split, seed 42, 5-fold CV), base-rate precision.
  E5  Cross-dataset test on 16 features that both datasets define in the same way:
      (a) within each dataset on these 16 features, (b) train on one dataset, test on the other.

Usage: python p1_extensions.py dataset_B_05_2020.csv tan_phishing_legitimate.csv [out]
Needs phishing_experiment.py in the same folder.
"""
import sys, os, json, time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from phishing_experiment import load as load_hannousse, models as base_models, metrics, precision_at_base_rate, SEED

# Tan name -> (Hannousse name, transform). Binary Tan flags are matched with "count > 0".
SHARED = {
    "UrlLength": ("length_url", None), "HostnameLength": ("length_hostname", None),
    "NumDots": ("nb_dots", None), "NumDash": ("nb_hyphens", None),
    "AtSymbol": ("nb_at", "binary"), "TildeSymbol": ("nb_tilde", "binary"),
    "NumUnderscore": ("nb_underscore", None), "NumPercent": ("nb_percent", None),
    "NumAmpersand": ("nb_and", None), "IpAddress": ("ip", None),
    "NumSensitiveWords": ("phish_hints", None), "PctExtHyperlinks": ("ratio_extHyperlinks", None),
    "ExtFavicon": ("external_favicon", None), "MissingTitle": ("empty_title", None),
    "DoubleSlashInPath": ("nb_dslash", "binary"), "PopUpWindow": ("popup_window", None),
}


def load_tan(path):
    df = pd.read_csv(path)
    y = (df["CLASS_LABEL"] == 1).astype(int).values
    X = df.drop(columns=["CLASS_LABEL"] + [c for c in ("id",) if c in df.columns])
    return X.apply(pd.to_numeric, errors="coerce").fillna(0), y


def all_models():
    m = base_models()
    m["MLP (neural network)"] = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-4,
                                                                               early_stopping=True, max_iter=300, random_state=SEED))
    return m


def score(m, X):
    return m.predict_proba(X)[:, 1] if hasattr(m, "predict_proba") else m.decision_function(X)


def evaluate(m, X_te, y_te):
    pred = m.predict(X_te)
    r = metrics(y_te, pred, score(m, X_te))
    tpr = 1 - r["FNR"]
    r["Precision @ 5% phishing"] = precision_at_base_rate(tpr, r["FPR"], 0.05)
    r["Precision @ 1% phishing"] = precision_at_base_rate(tpr, r["FPR"], 0.01)
    return r


def within(X, y, exp, data, names=None, cv=True):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    rows = []
    for name, m in all_models().items():
        if names and name not in names:
            continue
        r = {"Experiment": exp, "Train": data, "Test": data, "Model": name, "n_features": X.shape[1]}
        if cv:
            c = cross_validate(m, X_tr, y_tr, cv=StratifiedKFold(5, shuffle=True, random_state=SEED), scoring=["accuracy"])
            r["CV accuracy mean"], r["CV accuracy std"] = c["test_accuracy"].mean(), c["test_accuracy"].std()
        t0 = time.time(); m.fit(X_tr, y_tr); r["Train time (s)"] = time.time() - t0
        r.update(evaluate(m, X_te, y_te)); rows.append(r)
        print(f"[{exp} {data}] {name}: Acc={r['Accuracy']:.4f} F1={r['F1']:.4f} FPR={r['FPR']:.4f}", flush=True)
    return rows


def shared_views(Xh, Xt):
    H = pd.DataFrame({t: (Xh[h] > 0).astype(int) if tr == "binary" else Xh[h] for t, (h, tr) in SHARED.items()})
    T = Xt[list(SHARED)].copy()
    return H, T


def main():
    hpath, tpath = sys.argv[1], sys.argv[2]
    out = sys.argv[3] if len(sys.argv) > 3 else "results"
    os.makedirs(out, exist_ok=True)
    Xh, yh = load_hannousse(hpath)
    Xt, yt = load_tan(tpath)
    info = {"hannousse": [int(len(yh)), int(yh.sum())], "tan": [int(len(yt)), int(yt.sum())],
            "shared_features": {t: h for t, (h, _) in SHARED.items()}}
    rows = []
    rows += within(Xh, yh, "E1+ MLP", "Hannousse", names=["MLP (neural network)"])
    rows += within(Xt, yt, "E4 second dataset", "Tan")
    H, T = shared_views(Xh, Xt)
    rows += within(H, yh, "E5a shared features (within)", "Hannousse", cv=False)
    rows += within(T, yt, "E5a shared features (within)", "Tan", cv=False)
    for a_name, Xa, ya, b_name, Xb, yb in (("Hannousse", H, yh, "Tan", T, yt), ("Tan", T, yt, "Hannousse", H, yh)):
        for name, m in all_models().items():
            m.fit(Xa, ya)
            r = {"Experiment": "E5b cross-dataset", "Train": a_name, "Test": b_name, "Model": name, "n_features": Xa.shape[1]}
            r.update(evaluate(m, Xb, yb)); rows.append(r)
            print(f"[E5b {a_name}->{b_name}] {name}: Acc={r['Accuracy']:.4f} F1={r['F1']:.4f} FPR={r['FPR']:.4f} FNR={r['FNR']:.4f}", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(out, "results_p1_extensions.csv"), index=False)
    json.dump(info, open(os.path.join(out, "dataset_info_p1_extensions.json"), "w"), indent=2)
    pd.set_option("display.width", 250)
    print(res[["Experiment", "Train", "Test", "Model", "Accuracy", "F1", "FPR", "FNR", "ROC-AUC", "Precision @ 1% phishing"]].round(4).to_string())


if __name__ == "__main__":
    main()
