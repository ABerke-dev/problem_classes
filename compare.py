import pandas as pd

L = ["use", "transform", "reference", "none"]
tables = []
for name, f in [("real", "transfer.csv"), ("control", "control.csv")]:
    df = pd.read_csv(f, encoding="utf-8-sig")
    df = df[df["label"].isin(L)]
    t = (df.groupby("category")["label"]
           .value_counts(normalize=True)
           .unstack(fill_value=0)
           .round(2))
    t.insert(0, "set", name)
    print(name)
    print(t)
    print()
    tables.append(t)

out = pd.concat(tables)
out.to_csv("compare.csv", encoding="utf-8-sig")
print("Saved compare.csv")