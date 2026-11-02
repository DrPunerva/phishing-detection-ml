"""
Paper 4: PreLoad-Guard, a URL-only phishing prevention system that decides
BLOCK / WARN / ALLOW before a page is loaded.

Pipeline
  1. Features computed from the URL string only (no page visit, no third-party lookup):
       a) character 3-5-gram TF-IDF of the URL
       b) 24 lexical features (lengths, counts of special characters, digit ratio,
          IP host, subdomain depth, suspicious words, entropy, ...)
  2. Fusion model: a Logistic Regression meta-learner on out-of-fold scores of
       - Logistic Regression on character n-grams
       - Gradient Boosting on lexical features
  3. Three-zone prevention policy with thresholds set on out-of-fold training scores:
       BLOCK if score >= t_block (training FPR <= 0.5%)
       WARN  if t_warn <= score < t_block (training FPR <= 5%)
       ALLOW otherwise

Modes
  within <urls.csv> <name> [out]            5 repeated 80/20 splits on one dataset
  cross  <train.csv> <test.csv> <name> [out] train on one dataset, test on another,
                                             then update with a small labelled sample
Datasets are auto-detected: Hannousse (url, status) and PhiUSIIL (URL, label; 1 = legitimate).
"""
import sys, os, re, math, json, time
from urllib.parse import urlsplit
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix

BLOCK_FPR, WARN_FPR = 0.005, 0.05
SEEDS = [0, 1, 2, 3, 4]
SUSPICIOUS = ["login", "signin", "verify", "account", "update", "secure", "bank", "confirm",
              "password", "webscr", "ebayisapi", "paypal", "wallet", "support", "billing", "unlock"]
IP_RE = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")


# ---------------------------------------------------------------- data
def load_urls(path):
    df = pd.read_csv(path, low_memory=False)
    cols = {c.lower(): c for c in df.columns}
    if "status" in cols:                                   # Hannousse
        url, y = df[cols["url"]], (df[cols["status"]].astype(str).str.lower() == "phishing")
    elif "label" in cols and "url" in cols:                # PhiUSIIL: 1 = legitimate, 0 = phishing
        url, y = df[cols["url"]], (df[cols["label"]] == 0)
    else:
        raise ValueError("unknown dataset format")
    out = pd.DataFrame({"url": url.astype(str).str.strip(), "y": y.astype(int)})
    return out.drop_duplicates("url").reset_index(drop=True)


# ---------------------------------------------------------------- features
def _entropy(s):
    if not s:
        return 0.0
    c = Counter(s); n = len(s)
    return -sum(v / n * math.log2(v / n) for v in c.values())


def lexical(u):
    s = u if "://" in u else "http://" + u
    try:
        p = urlsplit(s)
        host, path, query = (p.hostname or ""), p.path, p.query
    except ValueError:
        host, path, query = "", u, ""
    lower = u.lower()
    labels = [x for x in host.split(".") if x]
    return [
        len(u), len(host), len(path), len(query),
        u.count("."), u.count("-"), u.count("@"), u.count("?"), u.count("="),
        u.count("_"), u.count("%"), u.count("/"), u.count("~"),
        sum(ch.isdigit() for ch in u) / max(len(u), 1),
        sum(ch.isdigit() for ch in host) / max(len(host), 1),
        int(bool(IP_RE.match(host))), max(len(labels) - 2, 0),
        int(lower.startswith("https")), int("//" in path),
        path.count("/"), sum(w in lower for w in SUSPICIOUS),
        _entropy(u), _entropy(host),
        int(bool(re.search(r"\.(php|html?|asp|exe|zip)$", path.lower()))),
    ]


LEX_NAMES = ["len_url", "len_host", "len_path", "len_query", "n_dot", "n_hyphen", "n_at", "n_qm",
             "n_eq", "n_underscore", "n_percent", "n_slash", "n_tilde", "digit_ratio_url",
             "digit_ratio_host", "ip_host", "n_subdomain", "https", "dslash_in_path", "path_depth",
             "n_suspicious_words", "entropy_url", "entropy_host", "file_ext"]


def lex_matrix(urls):
    return np.array([lexical(u) for u in urls], dtype=np.float32)


def char_vec():
    return TfidfVectorizer(analyzer="char", ngram_range=(3, 5), lowercase=True,
                           max_features=20000, sublinear_tf=True, dtype=np.float32)


# ---------------------------------------------------------------- models
def baselines(seed):
    """name -> (feature type, model)"""
    return {
        "LR (lexical)": ("lex", make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, random_state=seed))),
        "RF (lexical)": ("lex", RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=seed)),
        "GB (lexical)": ("lex", HistGradientBoostingClassifier(random_state=seed)),
        "LR (char n-grams)": ("char", LogisticRegression(max_iter=2000, C=4.0, random_state=seed)),
    }


FUSION_INPUTS = ["LR (char n-grams)", "GB (lexical)"]


class Features:
    def __init__(self):
        self.vec = char_vec()

    def fit(self, urls):
        self.vec.fit(urls); return self

    def transform(self, urls):
        return {"char": self.vec.transform(urls), "lex": lex_matrix(urls)}


def threshold_for_fpr(scores, y, target):
    neg = np.sort(scores[y == 0])
    k = int(np.floor(target * len(neg)))
    idx = len(neg) - k - 1
    return np.nextafter(neg[idx], np.inf) if idx >= 0 else -np.inf


def oof_scores(urls, y, seed, names):
    """Out-of-fold scores for the given base models (features refit inside each fold)."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.zeros((len(y), len(names)))
    for tr, va in skf.split(urls, y):
        f = Features().fit(urls[tr]); Xtr, Xva = f.transform(urls[tr]), f.transform(urls[va])
        b = baselines(seed)
        for j, n in enumerate(names):
            kind, m = b[n]
            m.fit(Xtr[kind], y[tr]); oof[va, j] = m.predict_proba(Xva[kind])[:, 1]
    return oof


class PreLoadGuard:
    def __init__(self, seed):
        self.seed = seed

    def fit(self, urls, y):
        names = list(baselines(self.seed))
        self.names = names
        oof = oof_scores(urls, y, self.seed, names)
        self.base_oof = oof
        self.feat = Features().fit(urls); X = self.feat.transform(urls)
        self.models = {}
        for n in names:
            kind, m = baselines(self.seed)[n]; m.fit(X[kind], y); self.models[n] = (kind, m)
        idx = [names.index(n) for n in FUSION_INPUTS]
        self.meta = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, random_state=self.seed))
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)
        meta_oof = np.zeros(len(y))
        for tr, va in skf.split(oof, y):
            self.meta.fit(oof[tr][:, idx], y[tr]); meta_oof[va] = self.meta.predict_proba(oof[va][:, idx])[:, 1]
        self.meta.fit(oof[:, idx], y)
        self.idx = idx
        # thresholds for every model (baselines get the same rule, for a fair comparison)
        self.thr = {}
        for j, n in enumerate(names):
            self.thr[n] = (threshold_for_fpr(oof[:, j], y, BLOCK_FPR), threshold_for_fpr(oof[:, j], y, WARN_FPR))
        self.thr["PreLoad-Guard (proposed)"] = (threshold_for_fpr(meta_oof, y, BLOCK_FPR),
                                                threshold_for_fpr(meta_oof, y, WARN_FPR))
        return self

    def scores(self, urls):
        X = self.feat.transform(urls)
        base = np.column_stack([self.models[n][1].predict_proba(X[self.models[n][0]])[:, 1] for n in self.names])
        out = {n: base[:, j] for j, n in enumerate(self.names)}
        out["PreLoad-Guard (proposed)"] = self.meta.predict_proba(base[:, self.idx])[:, 1]
        return out


# ---------------------------------------------------------------- evaluation
def detection_metrics(y, s, thr):
    pred = (s >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    tpr, fpr = tp / (tp + fn), fp / (fp + tn)
    prec = tp / (tp + fp) if tp + fp else 0.0
    r = {"Accuracy": (tp + tn) / len(y), "Precision": prec, "Recall": tpr,
         "F1": 2 * prec * tpr / (prec + tpr) if prec + tpr else 0.0, "FPR": fpr, "FNR": 1 - tpr}
    d = tpr * 0.01 + fpr * 0.99
    r["Precision @ 1% phishing"] = tpr * 0.01 / d if d else float("nan")
    return r


def prevention_metrics(y, s, t_block, t_warn):
    zone = np.where(s >= t_block, "block", np.where(s >= t_warn, "warn", "allow"))
    ph, lg = zone[y == 1], zone[y == 0]
    return {"Phish blocked": np.mean(ph == "block"), "Phish warned": np.mean(ph == "warn"),
            "Phish allowed (missed)": np.mean(ph == "allow"),
            "Legit blocked": np.mean(lg == "block"), "Legit warned": np.mean(lg == "warn"),
            "Legit allowed": np.mean(lg == "allow")}


def evaluate_all(guard, urls, y):
    sc = guard.scores(urls)
    rows = []
    for n, s in sc.items():
        tb, tw = guard.thr[n]
        r = {"Model": n, "ROC-AUC": roc_auc_score(y, s)}
        r.update({f"{k} (default 0.5)": v for k, v in detection_metrics(y, s, 0.5).items()})
        r.update({f"{k} (block threshold)": v for k, v in detection_metrics(y, s, tb).items()})
        r.update(prevention_metrics(y, s, tb, tw))
        rows.append(r)
    return rows


def latency(guard, urls, n=300):
    """Per-URL time (ms) for feature extraction + all models + fusion, one URL at a time."""
    sample = urls[:n]; times = []
    for u in sample:
        t0 = time.perf_counter(); guard.scores(np.array([u])); times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times)), float(np.percentile(times, 95))


def mode_within(path, name, out):
    d = load_urls(path); urls, y = d.url.values, d.y.values
    info = {"dataset": name, "urls": int(len(y)), "phishing": int(y.sum()), "legitimate": int((1 - y).sum())}
    print(json.dumps(info), flush=True)
    rows = []
    for seed in SEEDS:
        t0 = time.time()
        u_tr, u_te, y_tr, y_te = train_test_split(urls, y, test_size=0.2, stratify=y, random_state=seed)
        g = PreLoadGuard(seed).fit(u_tr, y_tr)
        for r in evaluate_all(g, u_te, y_te):
            rows.append({"Seed": seed, **r})
        if seed == SEEDS[0]:
            info["latency_ms_median"], info["latency_ms_p95"] = latency(g, u_te)
            info["test_size"] = int(len(y_te))
        print(f"seed {seed} done in {time.time()-t0:.0f}s", flush=True)
    res = pd.DataFrame(rows); res.insert(0, "Dataset", name)
    res.to_csv(os.path.join(out, f"p4_within_{name}.csv"), index=False)
    json.dump(info, open(os.path.join(out, f"p4_info_{name}.json"), "w"), indent=2)
    show(res)


def to_host(u):
    """Domain-only view of a URL (scheme, path, query removed; leading 'www.' dropped).
    Used for the cross-dataset test because the 2024 dataset's legitimate URLs are all
    bare https homepages, which would let a model separate classes by scheme/path alone."""
    s = u if "://" in u else "http://" + u
    try:
        h = (urlsplit(s).hostname or "").lower()
    except ValueError:
        h = ""
    return h[4:] if h.startswith("www.") else h


def host_view(d):
    d = d.assign(url=d.url.map(to_host))
    d = d[d.url != ""]
    # a host that appears with both labels is ambiguous at domain level; drop it
    amb = d.groupby("url").y.nunique()
    d = d[~d.url.isin(amb[amb > 1].index)]
    return d.drop_duplicates("url").reset_index(drop=True)


def mode_cross(train_path, test_path, name, out, n_update=1000, n_test=20000, seed=0):
    a = host_view(load_urls(train_path)); b = host_view(load_urls(test_path))
    b = b[~b.url.isin(set(a.url))].reset_index(drop=True)            # no overlapping URLs
    upd, rest = train_test_split(b, train_size=n_update, stratify=b.y, random_state=seed)
    test = rest.sample(n=min(n_test, len(rest)), random_state=seed)
    info = {"name": name, "train_urls": int(len(a)), "update_urls": int(len(upd)), "test_urls": int(len(test)),
            "test_phishing_share": float(test.y.mean())}
    rows = []
    g0 = PreLoadGuard(seed).fit(a.url.values, a.y.values)
    for r in evaluate_all(g0, test.url.values, test.y.values):
        rows.append({"Setting": "Old model (trained on 2020 data only)", **r})
    both = pd.concat([a, upd])
    g1 = PreLoadGuard(seed).fit(both.url.values, both.y.values)
    for r in evaluate_all(g1, test.url.values, test.y.values):
        rows.append({"Setting": f"Updated (+{n_update} new labelled URLs)", **r})
    res = pd.DataFrame(rows); res.insert(0, "Experiment", name)
    res.to_csv(os.path.join(out, f"p4_cross_{name}.csv"), index=False)
    json.dump(info, open(os.path.join(out, f"p4_info_cross_{name}.json"), "w"), indent=2)
    print(json.dumps(info)); show(res, by="Setting")


def show(res, by=None):
    cols = ["ROC-AUC", "F1 (default 0.5)", "FPR (block threshold)", "Recall (block threshold)",
            "Precision @ 1% phishing (block threshold)", "Phish blocked", "Phish warned",
            "Phish allowed (missed)", "Legit blocked", "Legit warned"]
    keys = ["Model"] if by is None else [by, "Model"]
    pd.set_option("display.width", 300)
    print(res.groupby(keys)[cols].mean().round(4).to_string())


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "within":
        out = sys.argv[4] if len(sys.argv) > 4 else "results"; os.makedirs(out, exist_ok=True)
        mode_within(sys.argv[2], sys.argv[3], out)
    else:
        out = sys.argv[5] if len(sys.argv) > 5 else "results"; os.makedirs(out, exist_ok=True)
        mode_cross(sys.argv[2], sys.argv[3], sys.argv[4], out)
