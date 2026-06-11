"""Two extra visuals for TDS Article 2: model-agreement heatmap + pipeline schematic."""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from pathlib import Path

OUT = Path(__file__).parent
NAVY="#1E2761"; ACCENT="#B85042"; INK="#222222"; GREY="#888888"
LABELS={"elo":"Elo (Ch8)","poisson":"Poisson (Ch4)","negbin":"Neg.Binom (Ch4)","logit":"Logistic (Ch5)",
        "knn":"KNN (Ch4/5)","rf":"Random Forest (Ch6)","xgb":"XGBoost (Ch6)","nn":"Neural Net (Ch7)",
        "colley":"Colley (Ch8)","pagerank":"PageRank (Ch8)"}

# ---------------- 1) Model-agreement heatmap ----------------
df = pd.read_csv(OUT/"model_title_probabilities.csv")
models=list(LABELS)
C = np.corrcoef(df[models].values.T)                 # 10x10 correlation across 48 teams
D = np.clip(1-C, 0, None); np.fill_diagonal(D, 0)
order = leaves_list(linkage(squareform(D, checks=False), method="average"))  # cluster similar models
Co = C[np.ix_(order,order)]; names=[LABELS[models[i]] for i in order]

cmap = LinearSegmentedColormap.from_list("agree", ["#F2EAE6", "#C99", ACCENT, NAVY])
fig, ax = plt.subplots(figsize=(9.5, 8))
im = ax.imshow(Co, cmap=cmap, vmin=np.min(C), vmax=1.0, aspect="equal")
ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=9)
for i in range(len(names)):
    for j in range(len(names)):
        ax.text(j,i,f"{Co[i,j]:.2f}",ha="center",va="center",fontsize=7.5,
                color="white" if Co[i,j]>0.85 else INK)
ax.set_title("How much the models agree\nCorrelation of title-probability vectors across all 48 teams",
             fontweight="bold", loc="left", fontsize=13)
cb=fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.set_label("Pearson correlation", fontsize=9)
fig.text(0.01,0.005,"'Soccer Analytics with Machine Learning' (O'Reilly, 2026) · clustered so similar models sit together",
         fontsize=7.5, color=GREY)
plt.tight_layout(); plt.savefig(OUT/"model_agreement_heatmap.png", dpi=160, bbox_inches="tight"); plt.close()
print("Saved model_agreement_heatmap.png")
# report rough clusters
print("Clustered order:", [models[i] for i in order])

# ---------------- 2) Pipeline schematic ----------------
fig, ax = plt.subplots(figsize=(12, 7)); ax.set_xlim(0,12); ax.set_ylim(0,10); ax.axis("off")
def box(x,y,w,h,text,fc,tc="white",fs=10,bold=True):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.04,rounding_size=0.12",
                 fc=fc,ec="none"))
    ax.text(x+w/2,y+h/2,text,ha="center",va="center",color=tc,fontsize=fs,
            fontweight="bold" if bold else "normal",wrap=True)
def arrow(x1,y1,x2,y2):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=16,
                 lw=1.6,color=GREY))

ax.text(0.1,9.6,"Eleven models, one tournament engine",fontsize=15,fontweight="bold",color=NAVY)
# left: model groups
groups=[("RATINGS · Ch 8",["Elo","Colley","PageRank"]),
        ("GOALS · Ch 4",["Poisson","Neg. Binomial"]),
        ("CLASSIFIERS · Ch 5–7",["Logistic","KNN","Random Forest","XGBoost","Neural Net"]),
        ("MARKET · Ch 9",["Market-implied odds"])]
y=8.7
for title,items in groups:
    ax.text(0.15,y+0.15,title,fontsize=8.5,fontweight="bold",color=ACCENT)
    for it in items:
        box(0.15,y-0.45,2.5,0.42,it,"#EAEDF5",tc=NAVY,fs=9)
        arrow(2.75,y-0.24,4.35,5.0)
        y-=0.55
    y-=0.25
# center: interface
box(4.4,4.3,3.0,1.4,"Common interface\nmatch_probs(a, b)\n→ P(win / draw / loss)\n+ expected goal diff",NAVY,fs=10)
arrow(7.4,5.0,8.0,5.0)
# center-right: simulator
box(8.0,4.2,3.6,1.6,"Vectorised simulator\n× 20,000 tournaments\n12 groups → 8 best thirds\n→ 32-team knockout",ACCENT,fs=10)
arrow(9.8,4.2,9.8,2.7)
# output
box(7.6,1.3,4.4,1.3,"Title probability\nfor each of 48 teams\n(per model → heatmap)","#2C5F2D",fs=10)
fig.text(0.99,0.02,"'Soccer Analytics with Machine Learning' (O'Reilly, 2026)",ha="right",fontsize=8,color=GREY)
plt.savefig(OUT/"pipeline_schematic.png", dpi=160, bbox_inches="tight"); plt.close()
print("Saved pipeline_schematic.png")
