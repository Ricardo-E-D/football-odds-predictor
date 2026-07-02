# Football Odds Predictor

Personal analysis/education project. It pulls historical football match results and
bookmaker odds, fits a probabilistic model of match outcomes, backtests that model
honestly (in time order, scored with proper probability metrics), and serves upcoming
fixtures with model-predicted probabilities next to current market odds in a small web app.

This is **not** a betting product — no accounts, no stakes, no payment anything. The
point is to learn sports-prediction modeling and find out, truthfully, whether a simple
model has any edge over the bookmaker (spoiler expectation: probably not much — the
backtest will tell us either way).

## Project structure

```
data/
  raw/            CSVs downloaded from football-data.co.uk (gitignored, re-downloadable)
  processed/      cleaned/merged match+odds tables (gitignored, rebuildable)
models/           model code (Poisson / Elo) and saved fitted parameters
backend/          API serving fixtures, model probabilities, current odds
frontend/         web UI
notebooks/        exploratory analysis and backtest write-ups
scripts/          standalone data download/refresh scripts (also used by CI)
tests/            tests
```

## Setup

```
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Dependencies are added phase by phase; see git history.

## Data sources

- [football-data.co.uk](https://www.football-data.co.uk/) — historical results + odds
  from multiple bookmakers, major European leagues, free CSVs. Primary backtest dataset.
- [The Odds API](https://the-odds-api.com/) — current/upcoming odds (free tier, key via env var).
- [football-data.org](https://www.football-data.org/) — upcoming fixtures (free tier, key via env var).

## Status

- [x] Phase 0 — repo, structure, environment
- [ ] Phase 1 — historical data collection
- [ ] Phase 2 — prediction model
- [ ] Phase 3 — backtest
- [ ] Phase 4 — backend API
- [ ] Phase 5 — frontend
- [ ] Phase 6 — deployment
- [ ] Phase 7 — automation & final write-up
