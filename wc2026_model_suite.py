"""
A SUITE OF COMPETING WORLD CUP 2026 MODELS — companion to
'Soccer Analytics with Machine Learning' (O'Reilly, 2026).

Each model is drawn from a chapter of the book and predicts the champion independently,
so they can be raced against each other ("a model a day"):

  Ch4  Poisson regression          (poisson)
  Ch4  Negative Binomial           (negbin)
  Ch4/5 K-Nearest Neighbours       (knn)
  Ch5  Logistic regression         (logit)
  Ch6  Random Forest               (rf)
  Ch6  Gradient boosting / XGBoost (xgb)
  Ch7  Neural network (MLP)        (nn)
  Ch8  Elo ratings                 (elo)
  Ch8  Colley ratings              (colley)
  Ch8  PageRank ratings            (pagerank)
  Ch9  Betting-market benchmark    (market)

DATA (real, in ./data/): 256 World Cup matches 2010-2022 + 102 recent Euro matches.
The classifiers/goal-models are TRAINED on these; the rating models are computed from them.
STRENGTH below is a current Elo-style snapshot used as the strength prior — REFRESH before publishing.
"""
import warnings; warnings.filterwarnings("ignore")
import os
import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import poisson as pois, nbinom
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import log_loss, accuracy_score
try:
    from xgboost import XGBClassifier; HAS_XGB = True
except Exception:
    HAS_XGB = False

rng = np.random.default_rng(7)
DATA = Path(__file__).parent / "data"

# ---------------------------------------------------------------- groups (resolved)
GROUPS = {
    "A": ["Mexico","South Africa","South Korea","Czechia"],
    "B": ["Canada","Bosnia-Herzegovina","Qatar","Switzerland"],
    "C": ["Brazil","Morocco","Haiti","Scotland"],
    "D": ["United States","Paraguay","Australia","Türkiye"],
    "E": ["Germany","Curacao","Ivory Coast","Ecuador"],
    "F": ["Netherlands","Japan","Sweden","Tunisia"],
    "G": ["Belgium","Egypt","Iran","New Zealand"],
    "H": ["Spain","Cape Verde","Saudi Arabia","Uruguay"],
    "I": ["France","Senegal","Iraq","Norway"],
    "J": ["Argentina","Algeria","Austria","Jordan"],
    "K": ["Portugal","Congo DR","Uzbekistan","Colombia"],
    "L": ["England","Croatia","Ghana","Panama"],
}
TEAMS = [t for g in GROUPS.values() for t in g]

# Strength prior (Elo-style snapshot). 48 contenders + historical teams (for training only).
STRENGTH = {
    "Spain":2165,"Argentina":2120,"France":2100,"England":2055,"Brazil":2025,"Netherlands":2030,
    "Portugal":2010,"Germany":1965,"Belgium":1950,"Croatia":1945,"Uruguay":1930,"Colombia":1915,
    "Morocco":1900,"Japan":1900,"Senegal":1895,"Switzerland":1860,"Norway":1855,"Austria":1850,
    "Ecuador":1840,"Türkiye":1840,"Mexico":1820,"Czechia":1815,"Sweden":1815,"United States":1805,
    "Iran":1800,"Ivory Coast":1800,"Algeria":1795,"South Korea":1790,"Scotland":1780,"Egypt":1780,
    "Canada":1780,"Ghana":1750,"Paraguay":1720,"Australia":1720,"Congo DR":1720,"Bosnia-Herzegovina":1710,
    "Tunisia":1700,"Qatar":1680,"Uzbekistan":1680,"Saudi Arabia":1655,"Iraq":1650,"Panama":1650,
    "South Africa":1640,"Jordan":1600,"Cape Verde":1555,"Curacao":1530,"Haiti":1500,"New Zealand":1500,
    # historical-only teams (training features only)
    "Italy":1990,"Denmark":1870,"Poland":1825,"Serbia":1820,"Chile":1815,"Russia":1815,"Nigeria":1810,
    "Ukraine":1800,"Greece":1800,"Hungary":1795,"Wales":1790,"Cameroon":1760,"Slovakia":1760,
    "Slovenia":1760,"Romania":1755,"Peru":1750,"Iceland":1740,"Georgia":1735,"Albania":1730,
    "North Macedonia":1720,"Finland":1715,"Costa Rica":1700,"Honduras":1620,"North Korea":1600,
}

# Current outright (futures) odds — top of market, American format (REFRESH before publishing).
FUTURES_AMERICAN = {
    "Spain":+430,"France":+500,"England":+650,"Brazil":+800,"Argentina":+950,"Germany":+1100,
    "Portugal":+1200,"Netherlands":+1400,"Belgium":+2200,"Croatia":+3300,"Uruguay":+3300,
    "Morocco":+4000,"Colombia":+4000,"Japan":+5000,"Senegal":+6600,"Switzerland":+8000,
}

# ---------------------------------------------------------------- load real matches
def load_matches():
    wc = pd.read_csv(DATA/"worldcup_matches.csv")
    ri = pd.read_csv(DATA/"recent_internationals.csv")
    cols = ["home_team","away_team","home_score","away_score","knockout"]
    m = pd.concat([wc[cols], ri[cols]], ignore_index=True)
    m = m[m.home_team.isin(STRENGTH) & m.away_team.isin(STRENGTH)].copy()
    m["sdiff"] = m.home_team.map(STRENGTH) - m.away_team.map(STRENGTH)
    m["sum_elo"] = m.home_team.map(STRENGTH) + m.away_team.map(STRENGTH)
    m["ko"] = m.knockout.astype(str).str.upper().eq("TRUE").astype(int)
    m["res"] = np.sign(m.home_score - m.away_score).astype(int)        # -1,0,1
    m["y"] = m.res + 1                                                 # 0 away,1 draw,2 home
    return m.reset_index(drop=True)

M = load_matches()
print(f"Training on {len(M)} real matches "
      f"({(M.res==1).mean()*100:.0f}% home/first-named wins, {(M.res==0).mean()*100:.0f}% draws).")

# ---------------------------------------------------------------- calibrations
# Poisson goal model: goals ~ strength diff (two rows per match: each team's view)
gl = pd.DataFrame({
    "goals": np.r_[M.home_score, M.away_score],
    "sd":    np.r_[M.sdiff, -M.sdiff],
})
glX = sm.add_constant(gl[["sd"]])
pois_fit = sm.GLM(gl.goals, glX, family=sm.families.Poisson()).fit()
B0, B1 = pois_fit.params["const"], pois_fit.params["sd"]
# Negative-binomial dispersion from Pearson chi2
alpha = max(1e-3, (pois_fit.pearson_chi2/pois_fit.df_resid - 1))
print(f"Poisson: lambda = exp({B0:.3f} + {B1:.5f}*sdiff)  | NB alpha={alpha:.3f}")

def lam(a, b):
    d = STRENGTH[a]-STRENGTH[b]
    return float(np.exp(B0+B1*d)), float(np.exp(B0-B1*d))

# Draw-rate model: P(draw) ~ |sdiff|  (logistic), for rating-based models
dz = sm.add_constant(pd.DataFrame({"abs": M.sdiff.abs()}))
draw_fit = sm.GLM((M.res==0).astype(int), dz, family=sm.families.Binomial()).fit()
def pdraw(absd):
    return float(draw_fit.predict(sm.add_constant(pd.DataFrame({"abs":[absd]}), has_constant="add"))[0])

# expected goal-diff per 1 strength point (for group tiebreaks of outcome models)
EGD_PER = B1*np.exp(B0)*2     # derivative-ish scale
SCORE_GRID = np.arange(0, 11)

# ---------------------------------------------------------------- rating systems
def elo_winexp(da):  # da = strength diff (a-b)
    return 1.0/(1.0+10**(-da/400.0))

def colley_ratings():
    teams = sorted(set(M.home_team)|set(M.away_team))
    idx = {t:i for i,t in enumerate(teams)}; n=len(teams)
    C = np.eye(n)*2; b = np.ones(n)
    for _,r in M.iterrows():
        i,j = idx[r.home_team], idx[r.away_team]
        C[i,i]+=1; C[j,j]+=1; C[i,j]-=1; C[j,i]-=1
        wi = 1.0 if r.res>0 else (0.5 if r.res==0 else 0.0)
        b[i]+= wi-0.5; b[j]+=(1-wi)-0.5
    r = np.linalg.solve(C,b)
    return {t:r[idx[t]] for t in teams}

def pagerank_ratings(d=0.85, it=100):
    teams = sorted(set(M.home_team)|set(M.away_team))
    idx={t:i for i,t in enumerate(teams)}; n=len(teams)
    W=np.zeros((n,n))   # loser -> winner weight
    for _,r in M.iterrows():
        i,j=idx[r.home_team],idx[r.away_team]
        if r.res>0:   W[j,i]+=1
        elif r.res<0: W[i,j]+=1
        else: W[i,j]+=0.5; W[j,i]+=0.5
    cs=W.sum(1,keepdims=True); cs[cs==0]=1; T=W/cs
    pr=np.ones(n)/n
    for _ in range(it): pr=(1-d)/n + d*(T.T@pr)
    return {t:pr[idx[t]] for t in teams}

def to_elo_scale(ratings):
    """Map an arbitrary rating dict onto an Elo-like scale; teams absent -> their prior."""
    vals=np.array(list(ratings.values())); mu,sd=vals.mean(),vals.std()+1e-9
    out={}
    for t in TEAMS:
        if t in ratings: out[t]=1850+(ratings[t]-mu)/sd*130
        else:            out[t]=STRENGTH[t]      # fallback to prior for absent teams
    return out

COLLEY = to_elo_scale(colley_ratings())
PAGER  = to_elo_scale(pagerank_ratings())

# ---------------------------------------------------------------- classifiers (Ch5/6/7)
feat_cols=["sdiff","sum_elo","ko"]
X=M[feat_cols].values; y=M.y.values
clf_defs={
    "logit": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)),
    "knn":   make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=30, weights="uniform")),
    "rf":    RandomForestClassifier(n_estimators=400, max_depth=5, min_samples_leaf=8, random_state=1),
    "nn":    make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(16,8), max_iter=3000, random_state=1)),
}
clf_defs["xgb"] = (XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                                 subsample=0.9, colsample_bytree=0.9, eval_metric="mlogloss",
                                 random_state=1) if HAS_XGB else
                   GradientBoostingClassifier(n_estimators=250, max_depth=3, learning_rate=0.05, random_state=1))
clf_classes={}
cv_rows=[]
for name,clf in clf_defs.items():
    pp = cross_val_predict(clf, X, y, cv=5, method="predict_proba")
    cv_rows.append({"model":name,
                    "cv_logloss":round(log_loss(y, pp, labels=[0,1,2]),3),
                    "cv_accuracy":round(accuracy_score(y, pp.argmax(1)),3)})
    clf.fit(X,y); clf_classes[name]=(clf, list(clf.classes_))
cv = pd.DataFrame(cv_rows).sort_values("cv_logloss")
print("\nClassifier 5-fold CV on real matches:\n", cv.to_string(index=False))

# ---------------------------------------------------------------- match-prob matrices
n=len(TEAMS); ix={t:i for i,t in enumerate(TEAMS)}
def grids_from_lambdas(la,lb):
    ph=pois.pmf(SCORE_GRID,la); pa=pois.pmf(SCORE_GRID,lb)
    P=np.outer(ph,pa)                       # P[h,a] = P(home=h, away=a)
    pHome=np.tril(P,-1).sum()               # home goals > away goals
    pAway=np.triu(P,1).sum(); pD=np.trace(P)
    s=pHome+pAway+pD
    return pHome/s, pD/s, pAway/s, la-lb
def grids_nb(la,lb):
    # NB with mean lam, dispersion alpha: var=lam+alpha*lam^2 ; nbinom(nn,pp)
    def pmf(l):
        if alpha<=1e-6: return pois.pmf(SCORE_GRID,l)
        rr=1.0/alpha; pp=rr/(rr+l); return nbinom.pmf(SCORE_GRID,rr,pp)
    ph,pa=pmf(la),pmf(lb); P=np.outer(ph,pa)
    pHome=np.tril(P,-1).sum()               # home goals > away goals
    pAway=np.triu(P,1).sum(); pD=np.trace(P); s=pHome+pAway+pD
    return pHome/s,pD/s,pAway/s,la-lb

def build_matrices():
    models={}
    # goals + rating models (cheap numpy per cell)
    for key in ["poisson","negbin","elo","colley","pagerank"]:
        PW=np.zeros((n,n)); PD=np.zeros((n,n)); PL=np.zeros((n,n)); EGD=np.zeros((n,n))
        for a in TEAMS:
            for b in TEAMS:
                if a==b: continue
                i,j=ix[a],ix[b]
                if key in ("poisson","negbin"):
                    la,lb=lam(a,b)
                    pw,pd_,pl,egd=(grids_from_lambdas if key=="poisson" else grids_nb)(la,lb)
                else:
                    S = STRENGTH if key=="elo" else (COLLEY if key=="colley" else PAGER)
                    da=S[a]-S[b]; we=elo_winexp(da); pdr=pdraw(abs(da))
                    pw=(1-pdr)*we; pl=(1-pdr)*(1-we); pd_=pdr; egd=da*EGD_PER
                PW[i,j],PD[i,j],PL[i,j],EGD[i,j]=pw,pd_,pl,egd
        models[key]={"PW":PW,"PD":PD,"PL":PL,"EGD":EGD}
    # classifiers (batched predict_proba over all ordered pairs)
    pairs=[(i,j) for i in range(n) for j in range(n) if i!=j]
    feat=np.array([[STRENGTH[TEAMS[i]]-STRENGTH[TEAMS[j]],
                    STRENGTH[TEAMS[i]]+STRENGTH[TEAMS[j]], 0] for i,j in pairs], float)
    for key,(clf,classes) in clf_classes.items():
        proba=clf.predict_proba(feat); cidx={c:k for k,c in enumerate(classes)}
        PW=np.zeros((n,n)); PD=np.zeros((n,n)); PL=np.zeros((n,n)); EGD=np.zeros((n,n))
        for r,(i,j) in enumerate(pairs):
            pl=proba[r,cidx.get(0,0)] if 0 in cidx else 0.0
            pd_=proba[r,cidx.get(1,0)] if 1 in cidx else 0.0
            pw=proba[r,cidx.get(2,0)] if 2 in cidx else 0.0
            PW[i,j],PD[i,j],PL[i,j],EGD[i,j]=pw,pd_,pl,(pw-pl)*1.7
        models[key]={"PW":PW,"PD":PD,"PL":PL,"EGD":EGD}
    return models

print("\nBuilding match-probability matrices for 10 models ...")
MODELS=build_matrices()

# Export per-model match probabilities for a few illustrative matchups (for the article worked example)
EX_PAIRS=[("Spain","Morocco"),("Argentina","Croatia"),("Netherlands","Germany")]
ex_rows=[]
for a,b in EX_PAIRS:
    i,j=ix[a],ix[b]
    for key in MODELS:
        m=MODELS[key]
        ex_rows.append({"matchup":f"{a} vs {b}","model":key,
                        "p_win":round(float(m['PW'][i,j])*100,1),
                        "p_draw":round(float(m['PD'][i,j])*100,1),
                        "p_loss":round(float(m['PL'][i,j])*100,1)})
pd.DataFrame(ex_rows).to_csv(DATA.parent/"example_matchups.csv",index=False)
print("Saved example_matchups.csv")

# ---------------------------------------------------------------- tournament engine
gi=[[ix[t] for t in GROUPS[g]] for g in GROUPS]
def bracket_seed_order(N):
    s=[1,2]
    while len(s)<N:
        m=len(s)*2+1; s=[x for v in s for x in (v,m-v)]
    return s
SEED=bracket_seed_order(32)

STR_ARR=np.array([STRENGTH[t] for t in TEAMS], float)
LOCAL_PAIRS=[(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
def _sort_by_strength(arr):                       # arr (N,k) team idx -> sorted by strength desc
    o=np.argsort(-STR_ARR[arr], axis=1); return np.take_along_axis(arr,o,1)

def simulate(model, N=4000):
    """Fully vectorised Monte Carlo over N tournaments."""
    PW,PD,PL,EGD=model["PW"],model["PD"],model["PL"],model["EGD"]
    win_g=[]; run_g=[]; third_t=[]; third_s=[]
    for grp in gi:
        gt=np.array(grp)
        pts=np.zeros((N,4)); gd=np.zeros((N,4))
        for a,b in LOCAL_PAIRS:
            i,j=grp[a],grp[b]; pw,pd_=PW[i,j],PD[i,j]
            u=rng.random(N)
            hw=u<pw; dr=(u>=pw)&(u<pw+pd_); aw=u>=pw+pd_
            pts[:,a]+=hw*3+dr; pts[:,b]+=aw*3+dr
            g=rng.normal(EGD[i,j],1.4,N); gd[:,a]+=g; gd[:,b]-=g
        score=pts*1000+gd+rng.random((N,4))*1e-3
        order=np.argsort(-score,axis=1)
        win_g.append(gt[order[:,0]]); run_g.append(gt[order[:,1]])
        third_local=order[:,2]
        third_t.append(gt[third_local])
        third_s.append(np.take_along_axis(pts,third_local[:,None],1)[:,0]*1000
                       +np.take_along_axis(gd,third_local[:,None],1)[:,0])
    winners=np.stack(win_g,1); runners=np.stack(run_g,1)            # (N,12)
    tt=np.stack(third_t,1); ts=np.stack(third_s,1)                  # (N,12)
    bo=np.argsort(-ts,axis=1)[:,:8]; thirds=np.take_along_axis(tt,bo,1)  # (N,8)
    seeded=np.concatenate([_sort_by_strength(winners),_sort_by_strength(runners),
                           _sort_by_strength(thirds)],axis=1)        # (N,32)
    field=seeded[:, np.array(SEED)-1]
    while field.shape[1]>1:
        a=field[:,0::2]; b=field[:,1::2]
        pw=PW[a,b]; pl=PL[a,b]; p=np.where((pw+pl)>0, pw/(pw+pl+1e-12), 0.5)
        u=rng.random(a.shape)
        field=np.where(u<p, a, b)
    return np.bincount(field[:,0], minlength=n)/N*100

# ---------------------------------------------------------------- run all models
N_SIM=int(os.environ.get("WC_N","4000"))
print(f"\nSimulating tournaments ({N_SIM:,} each) ...")
res={}
for key in MODELS:
    res[key]=simulate(MODELS[key], N_SIM); print(f"  {key:9s} top pick: {TEAMS[int(np.argmax(res[key]))]}")

# market benchmark (Ch9): de-vig futures odds
dec={t:1+a/100 if a>0 else 1+100/(-a) for t,a in FUTURES_AMERICAN.items()}
imp={t:1/d for t,d in dec.items()}; ov=sum(imp.values())
market={t:imp[t]/ov*100 for t in imp}
res["market"]=np.array([market.get(t,np.nan) for t in TEAMS])

df=pd.DataFrame({"team":TEAMS,"strength":[STRENGTH[t] for t in TEAMS]})
for key in res: df[key]=res[key]
model_cols=[c for c in ["elo","poisson","negbin","logit","knn","rf","xgb","nn","colley","pagerank","market"] if c in df]
df["consensus"]=df[[c for c in model_cols if c!="market"]].mean(1)
df=df.sort_values("consensus",ascending=False).reset_index(drop=True)
df.to_csv(DATA.parent/"model_title_probabilities.csv",index=False)
cv.to_csv(DATA.parent/"classifier_cv_metrics.csv",index=False)

picks=pd.DataFrame({"model":model_cols,
    "top_pick":[df.loc[df[c].idxmax(),"team"] for c in model_cols],
    "top_pct":[round(df[c].max(),1) for c in model_cols]})
picks.to_csv(DATA.parent/"model_picks.csv",index=False)

pd.set_option("display.width",200)
print("\n=== TITLE PROBABILITY (%) BY MODEL — top 12 by consensus ===")
show=["team"]+model_cols+["consensus"]
print(df[show].head(12).round(1).to_string(index=False))
print("\n=== EACH MODEL'S CHAMPION PICK ===")
print(picks.to_string(index=False))
print("\nSaved: model_title_probabilities.csv, model_picks.csv, classifier_cv_metrics.csv")
