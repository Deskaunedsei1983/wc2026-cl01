"""Comparison visuals for the WC2026 model suite. Reads the CSVs written by wc2026_model_suite.py."""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap

OUT = Path(__file__).parent                  # marketing/models/
df = pd.read_csv(OUT/"model_title_probabilities.csv")
picks = pd.read_csv(OUT/"model_picks.csv")

MODELS = ["elo","poisson","negbin","logit","knn","rf","xgb","nn","colley","pagerank","market"]
LABELS = {"elo":"Elo (Ch8)","poisson":"Poisson (Ch4)","negbin":"Neg.Binom (Ch4)","logit":"Logistic (Ch5)",
          "knn":"KNN (Ch4/5)","rf":"Random Forest (Ch6)","xgb":"XGBoost (Ch6)","nn":"Neural Net (Ch7)",
          "colley":"Colley (Ch8)","pagerank":"PageRank (Ch8)","market":"Market (Ch9)"}
NAVY="#1E2761"; ACCENT="#B85042"

top = df.head(12).copy()
sim_models = [m for m in MODELS if m != "market"]

# ---------- 1) Heatmap: title % by model ----------
mat = top[MODELS].values.astype(float)
cmap = LinearSegmentedColormap.from_list("nb", ["#FFFFFF", "#9FB1D8", NAVY])
fig, ax = plt.subplots(figsize=(13, 7.5))
im = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=0, vmax=np.nanmax(mat))
ax.set_xticks(range(len(MODELS))); ax.set_xticklabels([LABELS[m] for m in MODELS], rotation=40, ha="right")
ax.set_yticks(range(len(top))); ax.set_yticklabels(top["team"])
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        v = mat[i, j]
        if np.isnan(v): txt="—"; col="#bbbbbb"
        else: txt=f"{v:.0f}"; col="white" if v>np.nanmax(mat)*0.55 else "#222222"
        ax.text(j, i, txt, ha="center", va="center", fontsize=9, color=col)
ax.set_title("Who wins the 2026 World Cup? Title probability (%) by model",
             fontweight="bold", loc="left", fontsize=14)
fig.text(0.01,0.005,"A suite of models from 'Soccer Analytics with Machine Learning' (O'Reilly, 2026) · "
         "20,000 simulations each · trained on 358 real internationals · illustrative strength snapshot",
         fontsize=7.5, color="#888")
plt.tight_layout(); plt.savefig(OUT/"model_comparison_heatmap.png", dpi=160, bbox_inches="tight"); plt.close()

# ---------- 2) Consensus with disagreement range ----------
sim_cols = top[sim_models].values
cons = top["consensus"].values
lo = sim_cols.min(1); hi = sim_cols.max(1)
o = np.argsort(cons)
fig, ax = plt.subplots(figsize=(10, 7))
y = np.arange(len(top))
ax.hlines(y, lo[o], hi[o], color="#cfd6e8", lw=6, zorder=1)
ax.scatter(cons[o], y, color=NAVY, s=70, zorder=3, label="Consensus (mean of models)")
ax.scatter(top["market"].values[o], y, color=ACCENT, marker="D", s=45, zorder=4, label="Market-implied")
ax.set_yticks(y); ax.set_yticklabels(top["team"].values[o])
for yi, c in zip(y, cons[o]): ax.text(c, yi+0.16, f"{c:.0f}%", ha="center", fontsize=9, color=NAVY)
ax.set_xlabel("Probability of winning the 2026 World Cup (%)")
ax.set_title("Model consensus vs. the spread of opinion across 10 models",
             fontweight="bold", loc="left")
ax.spines[["top","right"]].set_visible(False); ax.legend(loc="lower right", frameon=False)
fig.text(0.99,0.005,"Bar = min–max across the 10 book models · diamond = de-vigged market-implied odds",
         ha="right", fontsize=7.5, color="#888")
plt.tight_layout(); plt.savefig(OUT/"model_consensus_range.png", dpi=160, bbox_inches="tight"); plt.close()
print("Saved model_comparison_heatmap.png and model_consensus_range.png")
print("\nChampion by model:\n", picks.to_string(index=False))
