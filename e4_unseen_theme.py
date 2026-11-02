"""E4 only: unseen-theme test (leave-one-theme-out) on cleaned text.
Uses the same data loading, cleaning, models and split as llm_phishing_experiment.py,
but skips Gradient Boosting on TF-IDF (too slow on 2,000 dense text features).
Usage: python e4_unseen_theme.py <folder with data/> [output_dir]
"""
import sys, os
import pandas as pd
from sklearn.model_selection import train_test_split
from llm_phishing_experiment import (load, clean_text, stylo_models, text_models, get_X,
                                     metrics, score, SEED)


def main():
    folder = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "results"
    df = load(folder)
    clean = df.copy()
    clean["text"] = clean["text"].map(clean_text)
    ctr, cte = train_test_split(clean, test_size=0.2, stratify=clean["source"], random_state=SEED)
    hte = cte[cte.y == 0]
    rows = []
    for theme in sorted(clean.loc[clean.y == 1, "category"].dropna().unique()):
        trn = pd.concat([ctr[ctr.y == 0], clean[(clean.y == 1) & (clean.category != theme)]])
        held = clean[(clean.y == 1) & (clean.category == theme)]
        tst = pd.concat([hte, held.sample(n=min(len(held), len(hte)), random_state=SEED)])
        for feats, models in (("Stylometric (17)", stylo_models()), ("TF-IDF (cleaned text)", text_models())):
            for name, m in models.items():
                if feats.startswith("TF-IDF") and name == "Gradient Boosting":
                    continue
                key = "TF-IDF" if feats.startswith("TF-IDF") else feats
                m.fit(get_X(trn, key), trn.y)
                Xt = get_X(tst, key)
                r = {"Experiment": "E4 unseen theme", "Features": feats, "Model": name, "Test LLM": f"theme={theme}"}
                r.update(metrics(tst.y, m.predict(Xt), score(m, Xt)))
                rows.append(r)
                print(f"[E4 {theme}] {feats} | {name}: Acc={r['Accuracy']:.4f} F1={r['F1']:.4f} FNR={r['FNR']:.4f}", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(out, "results_paper2_e4.csv"), index=False)


if __name__ == "__main__":
    main()
