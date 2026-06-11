"""Two extra visuals: (1) models vs market 'edge', (2) classifier CV comparison."""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).parent
df = pd.read_csv(OUT/"model_title_probabilities.csv")
cv = pd.read_csv(OUT/"classifier_cv_metrics.csv")
NAVY="#1E2761"; ACCENT="#B85042"; GREEN="#2C5F2D"; GREY="#888888"

# ---------- 1) Models vs market edge ----------
mk = df.dropna(subset=["market"]).copy()
mk["edge"] = mk["consensus"] - mk["market"]
mk = mk.sort_values("edge")
fig, ax = plt.subplots(figsize=(10, 7))
colors = [GREEN if e > 0 else ACCENT for e in mk["edge"]]
ax.barh(mk["team"], mk["edge"], color=colors)
for y,(e,t) in enumerate(zip(mk["edge"], mk["team"])):
    ax.text(e + (0.15 if e>=0 else -0.15), y, f"{e:+.1f}", va="center",
            ha="left" if e>=0 else "right", fontsize=10, color="#333")
ax.axvline(0, color="#333", lw=1)
ax.set_xlabel("Model consensus − market-implied probability (percentage points)")
ax.set_title("Where the models disagree with the market\n"
             "Green = models rate the team higher than the market does",
             fontweight="bold", loc="left")
ax.spines[["top","right"]].set_visible(False)
pad = max(abs(mk["edge"]))*1.25
ax.set_xlim(-pad, pad)
fig.text(0.99,0.01,"'Soccer Analytics with Machine Learning' (O'Reilly, 2026) · consensus of 10 models vs de-vigged market-implied odds",
         ha="right", fontsize=7.5, color=GREY)
plt.tight_layout(); plt.savefig(OUT/"model_vs_market.png", dpi=160, bbox_inches="tight"); plt.close()

# ---------- 2) Classifier CV comparison ----------
order = cv.sort_values("cv_logloss")
labels = {"logit":"Logistic\n(Ch5)","rf":"Random Forest\n(Ch6)","knn":"KNN\n(Ch4/5)",
          "nn":"Neural Net\n(Ch7)","xgb":"XGBoost\n(Ch6)"}
names = [labels.get(m,m) for m in order["model"]]
fig, ax = plt.subplots(figsize=(10, 6.2))
bars = ax.bar(names, order["cv_logloss"], color=NAVY, width=0.6)
bars[0].set_color(GREEN); bars[-1].set_color(ACCENT)
ax.axhline(np.log(3), color=GREY, ls="--", lw=1)
ax.text(len(names)-0.5, np.log(3)+0.004, "random guess (ln 3 ≈ 1.099)", ha="right", fontsize=9, color=GREY)
for b, ll, acc in zip(bars, order["cv_logloss"], order["cv_accuracy"]):
    ax.text(b.get_x()+b.get_width()/2, ll+0.004, f"{ll:.3f}\n{acc*100:.0f}% acc",
            ha="center", va="bottom", fontsize=9, color="#222")
ax.set_ylabel("5-fold cross-validated log-loss  (lower = better)")
ax.set_ylim(0.95, max(order["cv_logloss"])*1.06)
ax.set_title("On 358 matches, the simplest model wins\n"
             "Cross-validated fit: logistic regression beats the boosted trees",
             fontweight="bold", loc="left")
ax.spines[["top","right"]].set_visible(False)
fig.text(0.99,0.01,"'Soccer Analytics with Machine Learning' (O'Reilly, 2026) · 5-fold CV on 358 internationals",
         ha="right", fontsize=7.5, color=GREY)
plt.tight_layout(); plt.savefig(OUT/"classifier_cv_comparison.png", dpi=160, bbox_inches="tight"); plt.close()
print("Saved model_vs_market.png and classifier_cv_comparison.png")
print(mk[["team","consensus","market","edge"]].round(1).to_string(index=False))
