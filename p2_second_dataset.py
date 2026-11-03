"""
Paper 2 extension: a second, independent dataset and a cross-dataset test.

D2: Greco, Desolda, Esposito & Carelli (ITASEC 2024), "Human-LLM Generated
    Phishing-Legitimate Emails" (Kaggle, by francescogreco97). We use the two PHISHING
    files: 1,000 human-written phishing emails (Nazario / Nigerian-fraud collections) and
    1,000 LLM-generated phishing emails (ChatGPT and WormGPT). Positive class = LLM.
    For human emails only the body is used, because LLM emails have no subject line.

D1: Gutierrez et al. (2026) corpus used in the main experiments (see llm_phishing_experiment.py).

Experiments
  D2-E0  length-only Random Forest
  D2-E1  four TF-IDF models on raw text (80/20 split + 5-fold CV)
  D2-E3  four TF-IDF models after the same artefact cleaning and length matching as E1-E3
  X1     cross-dataset: train on cleaned D1, test on cleaned D2 (and the reverse)

Usage: python p2_second_dataset.py <D1 folder with data/> <D2 folder with dataset/> [out]
Needs llm_phishing_experiment.py in the same folder.
"""
import sys, os, json, time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from llm_phishing_experiment import load as load_d1, clean_text, text_models, metrics, score, SEED


def load_d2(folder):
    hum = pd.read_csv(os.path.join(folder, "dataset", "human-generated", "phishing.csv"))
    hum = pd.DataFrame({"text": hum["body"].fillna("").astype(str), "y": 0})
    lines = open(os.path.join(folder, "dataset", "llm-generated", "phishing.csv"), encoding="utf-8").read().splitlines()[1:]
    llm = pd.DataFrame({"text": [l.rsplit(",", 1)[0] for l in lines if l.strip()], "y": 1})
    d = pd.concat([hum, llm], ignore_index=True)
    d["text"] = d["text"].str.replace("\xa0", " ").str.strip()
    d = d[d.text.str.len() > 0].drop_duplicates("text").reset_index(drop=True)
    d["n_chars"] = d.text.str.len(); d["n_words"] = d.text.str.split().str.len()
    return d


def length_matched_clean(df):
    c = df.copy(); c["text"] = c.text.map(clean_text)
    lo, hi = df.loc[df.y == 1, "n_chars"].min(), df.loc[df.y == 1, "n_chars"].max()
    hum = c[(c.y == 0) & c.n_chars.between(lo, hi)]
    llm = c[c.y == 1]
    n = min(len(hum), len(llm))
    return pd.concat([hum.sample(n=n, random_state=SEED), llm.sample(n=n, random_state=SEED)]), (int(lo), int(hi))


def main():
    d1_folder, d2_folder = sys.argv[1], sys.argv[2]
    out = sys.argv[3] if len(sys.argv) > 3 else "results"
    os.makedirs(out, exist_ok=True)
    d2 = load_d2(d2_folder)
    info = {"D2_rows": len(d2), "D2_human": int((d2.y == 0).sum()), "D2_llm": int(d2.y.sum()),
            "D2_median_chars_human": float(d2[d2.y == 0].n_chars.median()),
            "D2_median_chars_llm": float(d2[d2.y == 1].n_chars.median()),
            "D2_paragraph_breaks_human": float(d2[d2.y == 0].text.str.contains("\n").mean()),
            "D2_paragraph_breaks_llm": float(d2[d2.y == 1].text.str.contains("\n").mean())}
    rows = []
    tr, te = train_test_split(d2, test_size=0.2, stratify=d2.y, random_state=SEED)

    m = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED).fit(tr[["n_chars", "n_words"]], tr.y)
    r = {"Experiment": "D2-E0 length baseline", "Train": "D2", "Test": "D2", "Features": "Length only", "Model": "Random Forest"}
    r.update(metrics(te.y, m.predict(te[["n_chars", "n_words"]]), score(m, te[["n_chars", "n_words"]]))); rows.append(r)
    print(f"[D2-E0] Acc={r['Accuracy']:.4f} F1={r['F1']:.4f}", flush=True)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for name, m in text_models().items():
        cvr = cross_validate(m, tr.text, tr.y, cv=cv, scoring=["accuracy", "f1"])
        m.fit(tr.text, tr.y)
        r = {"Experiment": "D2-E1 in-distribution", "Train": "D2", "Test": "D2", "Features": "TF-IDF", "Model": name,
             "CV Accuracy mean": cvr["test_accuracy"].mean(), "CV Accuracy std": cvr["test_accuracy"].std(),
             "CV F1 mean": cvr["test_f1"].mean(), "CV F1 std": cvr["test_f1"].std()}
        r.update(metrics(te.y, m.predict(te.text), score(m, te.text))); rows.append(r)
        print(f"[D2-E1] {name}: Acc={r['Accuracy']:.4f} F1={r['F1']:.4f}", flush=True)

    sub, win = length_matched_clean(d2)
    info["D2_E3_length_window"] = win; info["D2_E3_rows_per_class"] = int(len(sub) // 2)
    tr3, te3 = train_test_split(sub, test_size=0.2, stratify=sub.y, random_state=SEED)
    m = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED).fit(tr3[["n_chars", "n_words"]], tr3.y)
    r = {"Experiment": "D2-E3 artefact-reduced", "Train": "D2", "Test": "D2", "Features": "Length only", "Model": "Random Forest"}
    r.update(metrics(te3.y, m.predict(te3[["n_chars", "n_words"]]), score(m, te3[["n_chars", "n_words"]]))); rows.append(r)
    for name, m in text_models().items():
        m.fit(tr3.text, tr3.y)
        r = {"Experiment": "D2-E3 artefact-reduced", "Train": "D2", "Test": "D2", "Features": "TF-IDF (cleaned text)", "Model": name}
        r.update(metrics(te3.y, m.predict(te3.text), score(m, te3.text))); rows.append(r)
        print(f"[D2-E3] {name}: Acc={r['Accuracy']:.4f} F1={r['F1']:.4f}", flush=True)

    # X1 cross-dataset (cleaned text, both directions; test sets balanced by class)
    d1 = load_d1(d1_folder)[["text", "y"]].copy()
    d1["text"] = d1.text.map(clean_text)
    d2c = d2[["text", "y"]].copy(); d2c["text"] = d2c.text.map(clean_text)

    def balanced(d):
        n = d.y.value_counts().min()
        return pd.concat([d[d.y == 0].sample(n=n, random_state=SEED), d[d.y == 1].sample(n=n, random_state=SEED)])

    for a_name, a, b_name, b in (("D1", d1, "D2", d2c), ("D2", d2c, "D1", d1)):
        tst = balanced(b)
        for name, m in text_models().items():
            t0 = time.time(); m.fit(a.text, a.y)
            r = {"Experiment": "X1 cross-dataset", "Train": a_name, "Test": b_name, "Features": "TF-IDF (cleaned text)", "Model": name}
            r.update(metrics(tst.y, m.predict(tst.text), score(m, tst.text))); rows.append(r)
            print(f"[X1 {a_name}->{b_name}] {name}: Acc={r['Accuracy']:.4f} F1={r['F1']:.4f} FPR={r['FPR']:.4f} FNR={r['FNR']:.4f} ({time.time()-t0:.0f}s)", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(out, "results_paper2_d2.csv"), index=False)
    json.dump(info, open(os.path.join(out, "dataset_info_paper2_d2.json"), "w"), indent=2)
    print(json.dumps(info, indent=2))
    pd.set_option("display.width", 250)
    print(res[["Experiment", "Train", "Test", "Features", "Model", "Accuracy", "F1", "FPR", "FNR", "ROC-AUC"]].round(4).to_string())


if __name__ == "__main__":
    main()
