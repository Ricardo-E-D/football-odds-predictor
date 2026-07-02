# Football Odds Prediction Web App — Build Prompt for Claude Code

Paste everything below into Claude Code to kick off the project.

---

## Context for you (Claude Code)

I'm technical (comfortable with Python, git, APIs, SQLite) but a **complete beginner in sports prediction modeling and betting-market concepts**. I've built a Python trading system before, so the general shape of this — data layer → model → backtest → dashboard → deploy — is familiar, but the domain specifics (Poisson models, Elo, overround, log-loss, walk-forward validation, etc.) are new to me.

Rules for how we work together:
- **Explain new concepts in a sentence or two the first time you use them.** Don't explain basic programming/git things I'd already know — just the sports-modeling and backtesting-specific stuff.
- **Work in phases** (listed below). At the end of each phase, stop, summarize what you built and why, and wait for my go-ahead before moving to the next one.
- **Commit to git after every phase**, with a clear message. Set up the GitHub repo in Phase 0 and push from the start, not at the end.
- **Be honest, not promotional, about results.** If the backtest shows no real edge over the bookmaker, say so plainly. I'd rather know the truth than get a rosy summary.
- This is a personal analysis/education project, not a betting product — no need for account systems, payments, or anything gambling-adjacent.

---

## Goal

A fullstack web app that:
1. Pulls historical football match results + odds
2. Fits a prediction model estimating match outcome probabilities
3. Backtests that model properly (time-ordered, not random split) against historical odds to see whether it would have found any real edge over the bookmaker
4. Shows upcoming fixtures with model-predicted probabilities next to current bookmaker odds, in a clean web UI
5. Is hosted somewhere free with a public URL
6. Is version-controlled on GitHub from day one

---

## Data sources (verify current limits yourself — free tiers change often; this was accurate mid-2026)

- **football-data.co.uk** — free CSV downloads, no API key needed. Historical match results *and* odds from multiple bookmakers (Bet365, Pinnacle, etc.), covering most major European leagues from the 2000/01 season onward. **Use this as the primary backtesting dataset** — it's the best free source of matched results+odds that exists for this purpose.
- **football-data.org** — free tier, 12 major competitions (Premier League, Champions League, La Liga, Bundesliga, Serie A, Ligue 1, etc.), 10 requests/minute, scores delayed. Good for fixtures/results, not odds.
- **The Odds API** — has a free tier for current/upcoming odds across bookmakers; check the current quota when you sign up.
- **API-Football (api-sports.io)** — free tier for current-season fixtures/stats, useful as a cross-check.
- Skip any "free odds API" you find that looks like an SEO content-farm or asks for crypto payment — there are a lot of low-quality resale sites in this space. Stick to the ones above, or verify independently that something is an established, reputable source before using it.

---

## Phases

### Phase 0 — Setup
- Init git repo, create GitHub repo (ask me for name/visibility, or use `gh` CLI if configured), `.gitignore`, README with project description and setup steps
- Python virtual environment, project structure (e.g. `/data`, `/models`, `/backend`, `/frontend`, `/notebooks`)
- Explain the proposed structure before writing code

### Phase 1 — Data collection
- Download historical results+odds from football-data.co.uk for leagues I care about (ask me — default to Premier League + top 5 European leagues if I don't specify)
- Store raw data, write a small script to refresh/append new seasons
- Explain which columns matter (goals, odds columns, dates) and why

### Phase 2 — Modeling (start simple)
- Start with something interpretable — a Poisson goal-scoring model or Elo ratings — before anything fancier. Explain the concept plainly before building it.
- Convert the model's output into home/draw/away probabilities
- Compute the bookmaker's implied probability from the odds (explain overround/vig — why the odds don't sum to 100%)

### Phase 3 — Backtesting
- Explain what backtesting means here and why it has to be done in time order (train only on matches before a cutoff date, test on matches after) — random splits leak future information into training and give falsely good results
- Score with log-loss and Brier score, not raw accuracy — explain briefly why accuracy is misleading for probability predictions
- Compare the model's probabilities against the bookmaker's: where they disagree, was the model right more or less often than the market implied?
- Present results in a clear table/chart, and give me a direct, unhyped verdict on whether there's any real edge

### Phase 4 — Backend + API
- Small backend (Flask or FastAPI — reuse whichever I used on the trading dashboard if that helps consistency) serving upcoming fixtures with model probabilities + current odds
- A refresh job/endpoint for new data — explain the tradeoffs of cron vs. on-request given free hosting constraints

### Phase 5 — Frontend
- Clean, presentable frontend (simple React, or server-rendered templates if lower-maintenance — ask my preference) showing upcoming matches, model vs. market probability side-by-side, and basic backtest stats
- Keep it uncluttered — this is a personal analysis tool

### Phase 6 — Deployment
- Deploy to free hosting. As of mid-2026: **Render** has a genuine no-card free tier for web services (cold starts after inactivity are the tradeoff); **Railway and Fly.io no longer have real free tiers** (trial credits only, card required). Use Render for the backend, and a static host (Vercel/Netlify/GitHub Pages) for the frontend if it's split out — but verify current terms yourself, this changes often.
- API keys/secrets via environment variables, never committed to GitHub
- Walk me through getting the live URL working

### Phase 7 — Automation & polish
- Optional: GitHub Actions scheduled job to refresh fixture/odds data automatically (free on public repos)
- Final README: setup steps, architecture overview, and a plain-language summary of what the backtest actually showed

---

## Reminder
Stop after each phase, explain what you did and why in plain terms, commit + push, and wait for my review before continuing.