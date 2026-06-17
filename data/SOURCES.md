# Data sources & licensing

All match data here is **public domain** and cleared for any use, including commercial.

| File | Contents | Source | License |
|------|----------|--------|---------|
| `worldcup_matches.csv` | 256 World Cup matches, 2010–2022 (date, teams, scores, knockout flag) | openfootball [`worldcup.json`](https://github.com/openfootball/worldcup.json) | Public domain |
| `recent_internationals.csv` | 102 UEFA Euro matches (2020, 2024) | openfootball [`euro.json`](https://github.com/openfootball/euro.json) | Public domain |
| `wc2026_groups.csv`, `wc2026_fixtures.csv` | The confirmed 2026 World Cup draw and fixtures (incl. the official knockout bracket) | openfootball [`worldcup.json`](https://github.com/openfootball/worldcup.json) | Public domain |
| `wc2026_results.csv` | The 2026 schedule **and live results as they are played** (refreshed by `refresh_data.py`) | openfootball [`worldcup.json`](https://github.com/openfootball/worldcup.json) | Public domain |

`python refresh_data.py` downloads the latest 2026 results into `wc2026_results.csv` (and, with `--training`, rebuilds the historical sets into `*_refreshed.csv` for review). It normalizes team names to the spellings used by the models (e.g. *Czech Republic → Czechia*, *Turkey → Türkiye*). Goals use the full-time (90-minute) score, so knockout ties decided in extra time / on penalties are stored at their 90-minute score and flagged via the `knockout` column.

Verbatim from the openfootball repositories:

> "The worldcup.json schema, data and scripts are dedicated to the public domain. Use as you please with no restrictions whatsoever."

These are factual match results and fixtures (not copyrightable as facts) and are additionally dedicated to the public domain by the openfootball project.

### Inputs that are *not* in these files
- **Team strength ratings** used by the models are an illustrative Elo-style snapshot set in the code (`STRENGTH` dict). Refresh them from a public source such as [World Football Elo Ratings](https://www.eloratings.net/) before relying on the numbers.
- **Market-implied odds** (used only as a forecasting benchmark) are public sportsbook figures entered in the code; refresh before use.

*Credit (courtesy): match data from the [openfootball](https://github.com/openfootball) project, public domain.*
