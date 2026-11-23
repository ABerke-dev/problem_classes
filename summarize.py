import pandas as pd

df = pd.read_csv("transfer.csv", encoding="utf-8-sig")
df["source"] = df["inspiration"].str.upper().str.startswith("AI_").map({True: "AI", False: "WEB"})
df = df[df["label"].isin(["use", "transform", "reference", "none"])]

table = (df.groupby(["source", "category"])["label"]
           .value_counts(normalize=True)
           .unstack(fill_value=0)
           .round(2))
print(table)
table.to_csv("transfer_by_source.csv", encoding="utf-8-sig")