import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from settings import CATEGORIES

CATS = list(CATEGORIES)
LABELS = ["use", "transform", "reference", "none"]
COL = {"use": "#1b9e77", "transform": "#d95f02",
       "reference": "#7570b3", "none": "#d9d9d9"}

df = pd.read_csv("transfer.csv", encoding="utf-8-sig")
df["source"] = np.where(df["inspiration"].str.upper().str.startswith("AI_"), "AI", "Web")

# 1) Debug haritasi: her ilham x kategori
for part, g in df.groupby("participant"):
    insp = sorted(g["inspiration"].unique(),
                  key=lambda s: (not s.upper().startswith("AI_"), s.upper()))
    fig, ax = plt.subplots(figsize=(10, 0.45 * len(insp) + 2))
    for i, name in enumerate(insp):
        ax.text(-0.1, -i - 0.5, name, ha="right", va="center", fontsize=8)
        for j, c in enumerate(CATS):
            row = g[(g["inspiration"] == name) & (g["category"] == c)]
            lab = row["label"].iloc[0] if len(row) else "n/a"
            ax.add_patch(Rectangle((j, -i - 1), 1, 1, fc=COL.get(lab, "white"),
                                   ec="white", lw=2))
            ax.text(j + 0.5, -i - 0.5, lab, ha="center", va="center", fontsize=7)
    for j, c in enumerate(CATS):
        ax.text(j + 0.5, 0.2, c.replace("_", "\n"), ha="center", va="bottom", fontsize=8)
    ax.set_xlim(0, len(CATS))
    ax.set_ylim(-len(insp), 1.5)
    ax.axis("off")
    ax.set_title(f"Relations per inspiration and category: {part}", pad=30)
    fig.savefig(f"debug_{part}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved", f"debug_{part}.png")

# 2) General profile: AI vs Web, sayilarla
d = df[df["label"].isin(LABELS)]
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
for ax, src in zip(axes, ["AI", "Web"]):
    cnt = (d[d["source"] == src].groupby(["category", "label"]).size()
             .unstack(fill_value=0).reindex(index=CATS, columns=LABELS, fill_value=0))
    left = np.zeros(len(CATS))
    for lab in LABELS:
        vals = cnt[lab].values
        ax.barh(CATS, vals, left=left, color=COL[lab], label=lab)
        for y, (l, v) in enumerate(zip(left, vals)):
            if v > 0:
                ax.text(l + v / 2, y, int(v), ha="center", va="center",
                        color="white", fontsize=8)
        left += vals
    ax.set_title(f"{src} inspirations (general profile)")
    ax.invert_yaxis()
axes[0].legend(loc="lower left", fontsize=8)
fig.suptitle("Which relation did the judge assign? (counts, both participants)")
fig.savefig("profile.png", dpi=150, bbox_inches="tight")
print("Saved profile.png")