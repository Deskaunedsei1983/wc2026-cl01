"""
World Cup 2026 Monte Carlo simulator — companion to
'Soccer Analytics with Machine Learning' (O'Reilly, 2026).

Method: World Football Elo -> Poisson goal model -> 10,000 tournament simulations.
This mirrors the book's match-outcome modeling (Ch 4/6) and ratings (Ch 8).

>>> IMPORTANT <<<  The Elo values below are an illustrative early-2026 snapshot.
Refresh them with live numbers (e.g. eloratings.net) the day before you publish.
Everything else stays the same.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# --- Confirmed 2026 draw (Dec 5 2025) ---
GROUPS = {
    "A": ["Mexico", "South Africa", "South Korea", "Czechia"],
    "B": ["Canada", "Bosnia-Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Türkiye"],
    "E": ["Germany", "Curacao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "Congo DR", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

# --- Illustrative Elo snapshot (REFRESH BEFORE PUBLISHING) ---
ELO = {
    "Spain": 2165, "Argentina": 2120, "France": 2100, "England": 2055,
    "Brazil": 2025, "Netherlands": 2030, "Portugal": 2010, "Germany": 1965,
    "Belgium": 1950, "Croatia": 1945, "Uruguay": 1930, "Colombia": 1915,
    "Morocco": 1900, "Japan": 1900, "Senegal": 1895, "Switzerland": 1860,
    "Norway": 1855, "Austria": 1850, "Ecuador": 1840, "Türkiye": 1840,
    "Mexico": 1820, "Czechia": 1815, "Sweden": 1815, "United States": 1805,
    "Iran": 1800, "Ivory Coast": 1800, "Algeria": 1795, "South Korea": 1790,
    "Scotland": 1780, "Egypt": 1780, "Canada": 1780, "Ghana": 1750,
    "Paraguay": 1720, "Australia": 1720, "Congo DR": 1720, "Bosnia-Herzegovina": 1710,
    "Tunisia": 1700, "Qatar": 1680, "Uzbekistan": 1680, "Saudi Arabia": 1655,
    "Iraq": 1650, "Panama": 1650, "South Africa": 1640, "Jordan": 1600,
    "Cape Verde": 1555, "Curacao": 1530, "Haiti": 1500, "New Zealand": 1500,
}

GOALS_BASE = 2.7          # league-average total goals per match
GOALS_PER_400_ELO = 1.0   # ~1.0 goal of supremacy per 400 Elo points

def lambdas(a, b):
    """Expected goals for a and b from their Elo gap (neutral venue)."""
    diff = (ELO[a] - ELO[b]) / 400.0 * GOALS_PER_400_ELO
    la = max(0.15, GOALS_BASE / 2 + diff / 2)
    lb = max(0.15, GOALS_BASE / 2 - diff / 2)
    return la, lb

def play(a, b, knockout=False):
    la, lb = lambdas(a, b)
    ga, gb = rng.poisson(la), rng.poisson(lb)
    if ga != gb:
        return (a, ga, gb) if ga > gb else (b, ga, gb)
    if not knockout:
        return (None, ga, gb)            # draw
    # penalties: edge to the stronger side, capped
    p = 0.5 + (ELO[a] - ELO[b]) / 4000.0
    p = min(0.75, max(0.25, p))
    return (a if rng.random() < p else b, ga, gb)

def run_group(teams):
    pts = {t: 0 for t in teams}; gd = {t: 0 for t in teams}; gf = {t: 0 for t in teams}
    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            a, b = teams[i], teams[j]
            w, ga, gb = play(a, b)
            gf[a] += ga; gf[b] += gb; gd[a] += ga - gb; gd[b] += gb - ga
            if w is None:
                pts[a] += 1; pts[b] += 1
            else:
                pts[w] += 3
    order = sorted(teams, key=lambda t: (pts[t], gd[t], gf[t], rng.random()), reverse=True)
    rows = [{"team": t, "pts": pts[t], "gd": gd[t], "gf": gf[t]} for t in order]
    return order, rows

def bracket_seed_order(n):
    seeds = [1, 2]
    while len(seeds) < n:
        m = len(seeds) * 2 + 1
        seeds = [s for x in seeds for s in (x, m - x)]
    return seeds

def run_tournament():
    winners, runners, thirds = [], [], []
    for g, teams in GROUPS.items():
        order, rows = run_group(teams)
        winners.append(order[0]); runners.append(order[1])
        thirds.append({"team": order[2], "pts": rows[2]["pts"], "gd": rows[2]["gd"], "gf": rows[2]["gf"]})
    thirds.sort(key=lambda r: (r["pts"], r["gd"], r["gf"], rng.random()), reverse=True)
    best_thirds = [r["team"] for r in thirds[:8]]
    # Seed: winners (by Elo) 1-12, runners 13-24, thirds 25-32
    winners.sort(key=lambda t: ELO[t], reverse=True)
    runners.sort(key=lambda t: ELO[t], reverse=True)
    best_thirds.sort(key=lambda t: ELO[t], reverse=True)
    seeded = winners + runners + best_thirds          # 32 teams, index 0 = seed 1
    order = bracket_seed_order(32)
    bracket = [seeded[s - 1] for s in order]          # R32 matchups = consecutive pairs
    reached = {}                                       # team -> deepest round
    rnd_names = ["R32", "R16", "QF", "SF", "Final", "Champion"]
    for t in bracket:
        reached[t] = "R32"
    field = bracket
    for ridx in range(5):                              # R32->...->Final winner
        nxt = []
        for i in range(0, len(field), 2):
            w, _, _ = play(field[i], field[i + 1], knockout=True)
            nxt.append(w)
            reached[w] = rnd_names[ridx + 1]
        field = nxt
    champion = field[0]
    return champion, reached

def main(N=10000):
    teams = list(ELO.keys())
    title = {t: 0 for t in teams}
    finalist = {t: 0 for t in teams}
    semi = {t: 0 for t in teams}
    for _ in range(N):
        champ, reached = run_tournament()
        title[champ] += 1
        for t, r in reached.items():
            if r in ("Final", "Champion"):
                finalist[t] += 1
            if r in ("SF", "Final", "Champion"):
                semi[t] += 1
    df = pd.DataFrame({
        "team": teams,
        "title_pct": [100 * title[t] / N for t in teams],
        "final_pct": [100 * finalist[t] / N for t in teams],
        "semi_pct": [100 * semi[t] / N for t in teams],
        "elo": [ELO[t] for t in teams],
    }).sort_values("title_pct", ascending=False).reset_index(drop=True)
    df.to_csv("wc2026_probabilities.csv", index=False)
    print(f"=== World Cup 2026 — title probabilities ({N:,} simulations) ===\n")
    print(df.head(15).to_string(index=False,
          formatters={"title_pct": "{:.1f}%".format, "final_pct": "{:.1f}%".format,
                      "semi_pct": "{:.1f}%".format}))
    print(f"\nSum of title %: {df.title_pct.sum():.1f}  (sanity check ~100)")

    # --- Hero chart: title probability, top 15 ---
    top = df.head(15).iloc[::-1]
    plt.rcParams.update({"font.size": 12, "font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(top["team"], top["title_pct"], color="#1E2761")
    bars[-1].set_color("#B85042")  # highlight favorite
    for b, v in zip(bars, top["title_pct"]):
        ax.text(v + 0.2, b.get_y() + b.get_height() / 2, f"{v:.1f}%", va="center", fontsize=11)
    ax.set_xlabel("Probability of winning the 2026 World Cup")
    ax.set_title("Who wins the 2026 World Cup?\n10,000 simulations — an Elo + Poisson model",
                 fontweight="bold", loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, top["title_pct"].max() * 1.18)
    fig.text(0.99, 0.01, "Soccer Analytics with Machine Learning (O'Reilly, 2026) · illustrative Elo snapshot",
             ha="right", fontsize=8, color="#888888")
    plt.tight_layout()
    plt.savefig("wc2026_title_probabilities.png", dpi=160, bbox_inches="tight")
    print("\nSaved wc2026_title_probabilities.png and wc2026_probabilities.csv")

if __name__ == "__main__":
    main()
