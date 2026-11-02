"""
Human-written vs LLM-generated phishing emails: 4-model comparison.

Data: Gutierrez, Villegas-Ch & Govea (2026), Zenodo DOI 10.5281/zenodo.20250116
      5,000 human-written phishing + 4,986 LLM-generated phishing emails
      (GPT-4.1, DeepSeek 3.2, Llama 3.3 70B), with 17 stylometric features.
Positive class = LLM-generated.

Experiments
  E0  Sanity baseline: text length only (checks for a trivial shortcut).
  E1  4 models x 2 feature sets (17 stylometric features; TF-IDF word n-grams),
      stratified 80/20 split + 5-fold CV on the training part.
  E2  Unseen-LLM test (leave-one-LLM-out): train on human + two LLMs,
      test on held-out human emails + the third LLM.

Usage: python llm_phishing_experiment.py <folder with data/> [output_dir]
"""
import sys, os, json, time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix)

SEED = 42
STYLO = ["ttr", "mean_word_len", "mean_sentence_len_tokens", "yules_k", "clause_density",
         "noun_ratio", "verb_ratio", "mean_parse_depth", "imperative_count", "first_person_ratio",
         "second_person_ratio", "politeness_density", "urgency_density", "url_density",
         "cta_density", "authority_density", "time_pressure_density"]
LLMS = ["gpt-4.1", "deepseek3.2", "llama-3.3-70b"]


def load(folder):
    f = pd.read_csv(os.path.join(folder, "data", "corpus_features.csv"))
    f["y"] = (f["label_origin"] == "llm").astype(int)
    f["text"] = f["text"].fillna("").astype(str)
    f["n_chars"] = f["text"].str.len()
    f["n_words"] = f["text"].str.split().str.len()
    return f


import re
_PATTERNS = [
    (re.compile(r"\[[^\]]{0,40}\]"), " "),                       # [Your Name], [Link]
    (re.compile(r"https?://\S+|www\.\S+"), " "),                  # links
    (re.compile(r"\S+@\S+"), " "),                                # email addresses
    (re.compile(r"\byour ?bank\b|\bx{2,}\b", re.I), " "),         # placeholders (YourBank, xxx)
    (re.compile(r"\bmonkey\b|\bjose\b|\busaa\b", re.I), " "),     # corpus-specific tokens
    (re.compile(r"\d+"), " "),                                    # numbers
    (re.compile(r"\s+"), " "),                                    # all whitespace incl. paragraphs
]


def clean_text(s):
    s = str(s)
    for p, rep in _PATTERNS:
        s = p.sub(rep, s)
    return s.strip().lower()


def to_dense(X):
    return X.toarray().astype(np.float32)


def stylo_models():
    return {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, random_state=SEED)),
        "SVM": make_pipeline(StandardScaler(), SVC(kernel="rbf", random_state=SEED)),
        "Random Forest": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED),
        "Gradient Boosting": HistGradientBoostingClassifier(random_state=SEED),
    }


def tfidf():
    return TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=2000, sublinear_tf=True)


def text_models():
    return {
        "Logistic Regression": make_pipeline(tfidf(), LogisticRegression(max_iter=5000, random_state=SEED)),
        "SVM": make_pipeline(tfidf(), LinearSVC(random_state=SEED)),
        "Random Forest": make_pipeline(tfidf(), RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED)),
        "Gradient Boosting": make_pipeline(tfidf(), FunctionTransformer(to_dense, accept_sparse=True),
                                           HistGradientBoostingClassifier(random_state=SEED)),
    }


def score(m, X):
    return m.predict_proba(X)[:, 1] if hasattr(m, "predict_proba") else m.decision_function(X)


def metrics(y, pred, s):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"Accuracy": accuracy_score(y, pred), "Precision": precision_score(y, pred, zero_division=0),
            "Recall": recall_score(y, pred), "F1": f1_score(y, pred), "FPR": fp / (fp + tn),
            "FNR": fn / (fn + tp), "ROC-AUC": roc_auc_score(y, s),
            "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn)}


def get_X(df, feats):
    return df["text"] if feats == "TF-IDF" else df[STYLO]


def main():
    folder = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "results"
    os.makedirs(out, exist_ok=True)
    df = load(folder)
    info = {"rows": len(df), "human": int((df.y == 0).sum()), "llm": int(df.y.sum()),
            "per_llm": df[df.y == 1]["source"].value_counts().to_dict(),
            "chars_human": df[df.y == 0]["n_chars"].describe().round(1).to_dict(),
            "chars_llm": df[df.y == 1]["n_chars"].describe().round(1).to_dict()}
    print(json.dumps(info, indent=2))

    tr, te = train_test_split(df, test_size=0.2, stratify=df["source"], random_state=SEED)
    rows = []

    # E0: length-only shortcut
    for name, cols in (("Length only (chars + words)", ["n_chars", "n_words"]),):
        m = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED).fit(tr[cols], tr.y)
        r = {"Experiment": "E0 length baseline", "Features": name, "Model": "Random Forest", "Test LLM": "all"}
        r.update(metrics(te.y, m.predict(te[cols]), score(m, te[cols]))); rows.append(r)
        print(f"[E0] {name}: Acc={r['Accuracy']:.4f} F1={r['F1']:.4f}")

    # E1: in-distribution comparison
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for feats, mk in (("Stylometric (17)", stylo_models), ("TF-IDF", text_models)):
        for name, m in mk().items():
            cvr = cross_validate(m, get_X(tr, feats), tr.y, cv=cv, scoring=["accuracy", "f1"])
            t0 = time.time(); m.fit(get_X(tr, feats), tr.y); t = time.time() - t0
            Xt = get_X(te, feats)
            r = {"Experiment": "E1 in-distribution", "Features": feats, "Model": name, "Test LLM": "all",
                 "CV Accuracy mean": cvr["test_accuracy"].mean(), "CV Accuracy std": cvr["test_accuracy"].std(),
                 "CV F1 mean": cvr["test_f1"].mean(), "CV F1 std": cvr["test_f1"].std(), "Train time (s)": t}
            r.update(metrics(te.y, m.predict(Xt), score(m, Xt))); rows.append(r)
            print(f"[E1] {feats} | {name}: Acc={r['Accuracy']:.4f} F1={r['F1']:.4f} FPR={r['FPR']:.4f} FNR={r['FNR']:.4f}")

    # E2: leave-one-LLM-out (human split stays the same as E1)
    hum_tr, hum_te = tr[tr.y == 0], te[te.y == 0]
    for held in LLMS:
        llm_tr = df[(df.y == 1) & (df.source != held)]
        llm_te = df[(df.y == 1) & (df.source == held)]
        trn = pd.concat([hum_tr, llm_tr]); tst = pd.concat([hum_te, llm_te.sample(n=len(hum_te), random_state=SEED)])
        for feats, mk in (("Stylometric (17)", stylo_models), ("TF-IDF", text_models)):
            for name, m in mk().items():
                m.fit(get_X(trn, feats), trn.y)
                Xt = get_X(tst, feats)
                r = {"Experiment": "E2 unseen LLM", "Features": feats, "Model": name, "Test LLM": held}
                r.update(metrics(tst.y, m.predict(Xt), score(m, Xt))); rows.append(r)
                print(f"[E2 held={held}] {feats} | {name}: Acc={r['Accuracy']:.4f} F1={r['F1']:.4f} FNR={r['FNR']:.4f}")

    # E3: artifact-reduced test. Remove formatting, placeholders, links/addresses/numbers,
    # corpus-specific tokens, and length differences, then retrain the TF-IDF models.
    clean = df.copy()
    clean["text"] = clean["text"].map(clean_text)
    lo, hi = df.loc[df.y == 1, "n_chars"].min(), df.loc[df.y == 1, "n_chars"].max()
    hum = clean[(clean.y == 0) & (clean.n_chars.between(lo, hi))]
    llm = clean[clean.y == 1].sample(n=len(hum), random_state=SEED)
    sub = pd.concat([hum, llm])
    info["e3_length_window_chars"] = [int(lo), int(hi)]
    info["e3_rows_per_class"] = int(len(hum))
    tr3, te3 = train_test_split(sub, test_size=0.2, stratify=sub.y, random_state=SEED)
    m = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED).fit(tr3[["n_chars", "n_words"]], tr3.y)
    r = {"Experiment": "E3 artifact-reduced", "Features": "Length only (chars + words)", "Model": "Random Forest", "Test LLM": "all"}
    r.update(metrics(te3.y, m.predict(te3[["n_chars", "n_words"]]), score(m, te3[["n_chars", "n_words"]]))); rows.append(r)
    print(f"[E3] length only: Acc={r['Accuracy']:.4f}")
    for feats, mk in (("Stylometric (17)", stylo_models), ("TF-IDF (cleaned text)", text_models)):
        for name, m in mk().items():
            key = "TF-IDF" if feats.startswith("TF-IDF") else feats
            m.fit(get_X(tr3, key), tr3.y)
            Xt = get_X(te3, key)
            r = {"Experiment": "E3 artifact-reduced", "Features": feats, "Model": name, "Test LLM": "all"}
            r.update(metrics(te3.y, m.predict(Xt), score(m, Xt))); rows.append(r)
            print(f"[E3] {feats} | {name}: Acc={r['Accuracy']:.4f} F1={r['F1']:.4f} FPR={r['FPR']:.4f} FNR={r['FNR']:.4f}")
    # E4: unseen topic (leave-one-theme-out) on cleaned text: train on human + LLM emails
    # from four themes, test on held-out human emails + LLM emails from the fifth theme.
    ctr, cte = train_test_split(clean, test_size=0.2, stratify=clean["source"], random_state=SEED)
    for theme in sorted(clean.loc[clean.y == 1, "category"].dropna().unique()):
        trn = pd.concat([ctr[ctr.y == 0], clean[(clean.y == 1) & (clean.category != theme)]])
        held = clean[(clean.y == 1) & (clean.category == theme)]
        hte = cte[cte.y == 0]
        tst = pd.concat([hte, held.sample(n=min(len(held), len(hte)), random_state=SEED)])
        for feats, mk in (("Stylometric (17)", stylo_models), ("TF-IDF (cleaned text)", text_models)):
            for name, m in mk().items():
                key = "TF-IDF" if feats.startswith("TF-IDF") else feats
                m.fit(get_X(trn, key), trn.y)
                Xt = get_X(tst, key)
                r = {"Experiment": "E4 unseen theme", "Features": feats, "Model": name, "Test LLM": f"theme={theme}"}
                r.update(metrics(tst.y, m.predict(Xt), score(m, Xt))); rows.append(r)
                print(f"[E4 {theme}] {feats} | {name}: Acc={r['Accuracy']:.4f} F1={r['F1']:.4f} FNR={r['FNR']:.4f}")

    # Most informative words left after cleaning (Logistic Regression on cleaned TF-IDF)
    lr = text_models()["Logistic Regression"].fit(sub["text"], sub.y)
    vec, clf = lr.steps[0][1], lr.steps[-1][1]
    names, w = np.array(vec.get_feature_names_out()), clf.coef_[0]
    o = np.argsort(w)
    info["e3_top_llm_terms"] = list(names[o[-15:]][::-1])
    info["e3_top_human_terms"] = list(names[o[:15]])

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(out, "results_paper2.csv"), index=False)
    with open(os.path.join(out, "dataset_info_paper2.json"), "w") as f:
        json.dump(info, f, indent=2, default=str)


if __name__ == "__main__":
    main()
