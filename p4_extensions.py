"""
Paper 4 extensions (D-2020, the same five 80/20 splits as `preload_guard.py within`):

  1. Neural-network baseline: a multi-layer perceptron (MLP) on the same character
     3-5-gram TF-IDF features, with block / warn thresholds set by the SAME rule as
     every other model (5-fold out-of-fold training scores; FPR <= 0.5% / 5%).
  2. Simple evasion test: phishing test URLs are rewritten by an attacker who changes
     only what he controls, and we measure how many are still blocked / warned / missed.
       A1 https        'http://' -> 'https://'
       A2 no-keywords  remove suspicious words (login, verify, account, ...) from path and query
       A3 homepage     keep only scheme + host + '/', i.e. drop path, query and fragment
       A4 www          add 'www.' in front of the host (if not an IP and not already there)
       A5 all          A1 + A2 + A3 + A4 together
     Legitimate URLs are not changed, so the legitimate block rate is the clean one.

Usage: python p4_extensions.py <dataset_B_05_2020.csv> [output_dir]
Needs preload_guard.py in the same folder.
"""
import sys, os, re, time, json
import numpy as np
import pandas as pd
from urllib.parse import urlsplit
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from preload_guard import (load_urls, PreLoadGuard, Features, threshold_for_fpr, detection_metrics,
                           prevention_metrics, SUSPICIOUS, IP_RE, BLOCK_FPR, WARN_FPR, SEEDS)

MLP_NAME = "MLP (char n-grams, neural network)"


def mlp(seed):
    return MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-4, early_stopping=True, max_iter=40,
                         random_state=seed)


class MLPBaseline:
    def fit(self, urls, y, seed):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        oof = np.zeros(len(y))
        for tr, va in skf.split(urls, y):
            f = Features().fit(urls[tr])
            m = mlp(seed).fit(f.transform(urls[tr])["char"], y[tr])
            oof[va] = m.predict_proba(f.transform(urls[va])["char"])[:, 1]
        self.feat = Features().fit(urls)
        t0 = time.time(); self.m = mlp(seed).fit(self.feat.transform(urls)["char"], y); self.fit_s = time.time() - t0
        self.thr = (threshold_for_fpr(oof, y, BLOCK_FPR), threshold_for_fpr(oof, y, WARN_FPR))
        return self

    def scores(self, urls):
        return self.m.predict_proba(self.feat.transform(urls)["char"])[:, 1]


# ---------------------------------------------------------------- attacks
KW_RE = re.compile("|".join(sorted(SUSPICIOUS, key=len, reverse=True)), re.I)


def _parts(u):
    s = u if "://" in u else "http://" + u
    try:
        p = urlsplit(s)
        return p.scheme or "http", p.netloc, p.path, p.query, p.fragment
    except ValueError:
        return "http", "", u, "", ""


def a_https(u):
    return "https://" + u[len("http://"):] if u.lower().startswith("http://") else u


def a_nokw(u):
    """Remove suspicious words from the path and query (the host is left unchanged)."""
    sc, net, path, q, fr = _parts(u)
    rest = path + ("?" + q if q else "") + ("#" + fr if fr else "")
    new = KW_RE.sub("", rest)
    return u if new == rest else f"{sc}://{net}{new}"


def a_home(u):
    sc, net, *_ = _parts(u)
    return f"{sc}://{net}/"


def a_www(u):
    sc, net, path, q, fr = _parts(u)
    host = net.split("@")[-1].split(":")[0]
    if net.lower().startswith("www.") or IP_RE.match(host) or not net:
        return u
    return u.replace(net, "www." + net, 1)


ATTACKS = {"A1 https": [a_https], "A2 no suspicious words": [a_nokw], "A3 homepage only": [a_home],
           "A4 add www.": [a_www], "A5 all four": [a_nokw, a_home, a_www, a_https]}


def attack(urls, fns):
    out = []
    for u in urls:
        for f in fns:
            u = f(u)
        out.append(u)
    return np.array(out, dtype=object)


# ---------------------------------------------------------------- run
def main():
    path = sys.argv[1]; out = sys.argv[2] if len(sys.argv) > 2 else "results"
    os.makedirs(out, exist_ok=True)
    d = load_urls(path); urls, y = d.url.values, d.y.values
    clean_rows, atk_rows, examples = [], [], {}
    for seed in SEEDS:
        t0 = time.time()
        u_tr, u_te, y_tr, y_te = train_test_split(urls, y, test_size=0.2, stratify=y, random_state=seed)
        g = PreLoadGuard(seed).fit(u_tr, y_tr)
        nn = MLPBaseline().fit(u_tr, y_tr, seed)
        thr = dict(g.thr); thr[MLP_NAME] = nn.thr

        def all_scores(u):
            s = g.scores(u); s[MLP_NAME] = nn.scores(u); return s

        sc = all_scores(u_te)
        for n, s in sc.items():
            tb, tw = thr[n]
            r = {"Seed": seed, "Model": n, "ROC-AUC": roc_auc_score(y_te, s)}
            r.update({f"{k} (default 0.5)": v for k, v in detection_metrics(y_te, s, 0.5).items()})
            r.update({f"{k} (block threshold)": v for k, v in detection_metrics(y_te, s, tb).items()})
            r.update(prevention_metrics(y_te, s, tb, tw))
            if n == MLP_NAME:
                r["Fit time (s)"] = nn.fit_s
            clean_rows.append(r)

        ph = u_te[y_te == 1]
        variants = {"A0 none (clean)": ph}
        variants.update({a: attack(ph, fns) for a, fns in ATTACKS.items()})
        for a, u in variants.items():
            changed = float(np.mean(u != ph))
            s_all = all_scores(u)
            for n, s in s_all.items():
                tb, tw = thr[n]
                atk_rows.append({"Seed": seed, "Attack": a, "Model": n, "URLs changed": changed,
                                 "Phish blocked": float(np.mean(s >= tb)),
                                 "Phish warned": float(np.mean((s >= tw) & (s < tb))),
                                 "Phish allowed (missed)": float(np.mean(s < tw))})
            if seed == SEEDS[0]:
                examples[a] = [str(x) for x in u[:3]]
        print(f"seed {seed} done in {time.time()-t0:.0f}s", flush=True)

    clean = pd.DataFrame(clean_rows); atk = pd.DataFrame(atk_rows)
    clean.to_csv(os.path.join(out, "p4_ext_mlp_hannousse.csv"), index=False)
    atk.to_csv(os.path.join(out, "p4_ext_attacks_hannousse.csv"), index=False)
    json.dump(examples, open(os.path.join(out, "p4_ext_attack_examples.json"), "w"), indent=2)
    pd.set_option("display.width", 300)
    cols = ["ROC-AUC", "F1 (default 0.5)", "Phish blocked", "Phish warned", "Phish allowed (missed)",
            "Legit blocked", "Legit warned"]
    print(clean.groupby("Model")[cols].mean().round(4).to_string())
    print(atk.groupby(["Attack", "Model"])[["URLs changed", "Phish blocked", "Phish allowed (missed)"]]
          .mean().round(4).to_string())


if __name__ == "__main__":
    main()
