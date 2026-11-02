"""Convert the Tan phishing dataset (Phishing_Legitimate_full.arff) to CSV.
Usage: python arff_to_csv.py Phishing_Legitimate_full.arff tan_phishing_legitimate.csv
"""
import sys, io
import pandas as pd

src, dst = sys.argv[1], sys.argv[2]
lines = open(src, encoding="utf-8").read().splitlines()
names = [l.split()[1] for l in lines if l.lower().startswith("@attribute")]
start = next(i for i, l in enumerate(lines) if l.lower().startswith("@data"))
rows = [l for l in lines[start + 1:] if l.strip() and not l.startswith("%")]
df = pd.read_csv(io.StringIO("\n".join(rows)), header=None, names=names)
df.to_csv(dst, index=False)
print(df.shape, df["CLASS_LABEL"].value_counts().to_dict())
