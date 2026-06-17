"""
refresh_data.py — download up-to-date match data for the World Cup 2026 models.

All data comes from the public-domain openfootball project
(https://github.com/openfootball). Outbound HTTPS to raw.githubusercontent.com
is required.

Usage:
    python refresh_data.py                 # refresh the live 2026 results -> data/wc2026_results.csv
    python refresh_data.py --training      # ALSO rebuild the historical training sets
                                           #   (written to data/*_refreshed.csv, non-destructive)

The live file `data/wc2026_results.csv` is what `wc2026_live_update.ipynb` reads.
Re-run this whenever new 2026 matches have been played, then re-run the notebook.
"""
import sys, json, csv, urllib.request
from pathlib import Path

DATA = Path(__file__).parent / "data"
RAW  = "https://raw.githubusercontent.com/openfootball"
WC   = RAW + "/worldcup.json/master/{year}/worldcup.json"
EURO = RAW + "/euro.json/master/{year}/euro.json"

# openfootball spelling -> the spelling used by the models' STRENGTH dict
NAME_MAP = {
    "Czech Republic": "Czechia", "Bosnia & Herzegovina": "Bosnia-Herzegovina",
    "Turkey": "Türkiye", "USA": "United States", "Curaçao": "Curacao",
    "DR Congo": "Congo DR", "Korea Republic": "South Korea", "IR Iran": "Iran",
    "China PR": "China", "Côte d'Ivoire": "Ivory Coast", "Czechia": "Czechia",
}
def norm(t):
    return NAME_MAP.get(t, t)

def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "wc2026-model/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))

KO_ROUNDS = {"Round of 32", "Round of 16", "Quarter-final", "Semi-final",
             "Match for third place", "Third place", "Final"}

def parse_matches(doc):
    """Yield normalized rows from an openfootball tournament document.

    Goals use the full-time (`ft`, 90-minute) score, matching the convention of
    the committed training files and the Poisson goal model. Knockout ties that
    went to extra time / penalties are therefore recorded at their 90-minute
    score (i.e. a draw); the `knockout` flag marks them.
    """
    for m in doc.get("matches", []):
        ft = (m.get("score") or {}).get("ft")
        grp = (m.get("group") or "").replace("Group ", "").strip()
        rnd = m.get("round", "")
        yield {
            "date": m.get("date", ""), "round": rnd, "group": grp,
            "team1": norm(m.get("team1", "")), "team2": norm(m.get("team2", "")),
            "score1": "" if not ft else ft[0], "score2": "" if not ft else ft[1],
            "knockout": "TRUE" if (not grp and rnd in KO_ROUNDS) else "FALSE",
        }

# ---------------------------------------------------------------- live 2026 file
def refresh_2026():
    rows = list(parse_matches(fetch(WC.format(year=2026))))
    out = DATA / "wc2026_results.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["date", "round", "group", "team1", "team2",
                                          "score1", "score2", "knockout"], lineterminator="\n")
        w.writeheader(); w.writerows(rows)
    played = sum(1 for r in rows if r["score1"] != "")
    grp = sum(1 for r in rows if r["group"])
    print(f"wc2026_results.csv  ·  {len(rows)} matches, {played} played "
          f"({sum(1 for r in rows if r['group'] and r['score1']!='')}/{grp} group games).")
    return out

# ---------------------------------------------------------------- training rebuild
def rebuild_training():
    # World Cups 2010-2022 -> worldcup_matches.csv schema (non-destructive: *_refreshed)
    wc_rows = []
    for year in (2010, 2014, 2018, 2022):
        for r in parse_matches(fetch(WC.format(year=year))):
            if r["score1"] == "":
                continue
            wc_rows.append({"date": r["date"], "year": (r["date"] or "0000")[:4],
                            "home_team": r["team1"], "away_team": r["team2"],
                            "home_score": r["score1"], "away_score": r["score2"],
                            "knockout": r["knockout"]})
    p = DATA / "worldcup_matches_refreshed.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["date", "year", "home_team", "away_team",
                                          "home_score", "away_score", "knockout"], lineterminator="\n")
        w.writeheader(); w.writerows(wc_rows)
    print(f"worldcup_matches_refreshed.csv  ·  {len(wc_rows)} matches (2010-2022).")

    # Euro 2020 + 2024 -> recent_internationals.csv schema
    eu_rows = []
    for year, tag in ((2020, "Euro 2020"), (2024, "Euro 2024")):
        for r in parse_matches(fetch(EURO.format(year=year))):
            if r["score1"] == "":
                continue
            eu_rows.append({"date": r["date"], "tournament": tag,
                            "home_team": r["team1"], "away_team": r["team2"],
                            "home_score": r["score1"], "away_score": r["score2"],
                            "knockout": r["knockout"]})
    p = DATA / "recent_internationals_refreshed.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["date", "tournament", "home_team", "away_team",
                                          "home_score", "away_score", "knockout"], lineterminator="\n")
        w.writeheader(); w.writerows(eu_rows)
    print(f"recent_internationals_refreshed.csv  ·  {len(eu_rows)} matches (Euro 2020/2024).")
    print("(Refreshed training files written alongside the originals — diff before replacing.)")

if __name__ == "__main__":
    try:
        refresh_2026()
        if "--training" in sys.argv:
            rebuild_training()
    except Exception as e:
        print(f"Download failed ({e.__class__.__name__}: {e}).")
        print("Check outbound network access to raw.githubusercontent.com.")
        sys.exit(1)
