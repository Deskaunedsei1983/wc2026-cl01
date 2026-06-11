# Data sources & licensing

All match data here is **public domain** and cleared for any use, including commercial.

| File | Contents | Source | License |
|------|----------|--------|---------|
| `worldcup_matches.csv` | 256 World Cup matches, 2010–2022 (date, teams, scores, knockout flag) | openfootball [`worldcup.json`](https://github.com/openfootball/worldcup.json) | Public domain |
| `recent_internationals.csv` | 102 UEFA Euro matches (2020, 2024) | openfootball [`euro.json`](https://github.com/openfootball/euro.json) | Public domain |
| `wc2026_groups.csv`, `wc2026_fixtures.csv` | The confirmed 2026 World Cup draw and fixtures | openfootball [`worldcup.json`](https://github.com/openfootball/worldcup.json) | Public domain |

Verbatim from the openfootball repositories:

> "The worldcup.json schema, data and scripts are dedicated to the public domain. Use as you please with no restrictions whatsoever."

These are factual match results and fixtures (not copyrightable as facts) and are additionally dedicated to the public domain by the openfootball project.

### Inputs that are *not* in these files
- **Team strength ratings** used by the models are an illustrative Elo-style snapshot set in the code (`STRENGTH` dict). Refresh them from a public source such as [World Football Elo Ratings](https://www.eloratings.net/) before relying on the numbers.
- **Market-implied odds** (used only as a forecasting benchmark) are public sportsbook figures entered in the code; refresh before use.

*Credit (courtesy): match data from the [openfootball](https://github.com/openfootball) project, public domain.*
