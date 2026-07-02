"""FastAPI backend: upcoming fixtures with model vs market probabilities.

Run locally:
    uvicorn backend.app:app --reload

Endpoints:
    GET  /api/health
    GET  /api/leagues
    GET  /api/fixtures/{league}     model + market probabilities, upcoming matches
    GET  /api/backtest              headline backtest results (the honest verdict)
    POST /api/refresh               re-download current season, refit models
"""

import os
from functools import lru_cache

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend import model_store
from backend.config import LEAGUES, PROCESSED_DIR, ROOT
from backend.odds_api import TeamMatcher, fetch_events
from models.metrics import brier_score, log_loss, outcome_index
from models.odds import implied_probabilities

load_dotenv(ROOT / ".env")

app = FastAPI(title="Football Odds Predictor", version="0.1")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_state: dict = {"models": None, "fitted_through": None}


def get_models():
    if _state["models"] is None:
        _state["models"], _state["fitted_through"] = model_store.get_models()
    return _state["models"]


@app.get("/api/health")
def health():
    return {"status": "ok", "fitted_through": _state["fitted_through"]}


@app.get("/api/leagues")
def leagues():
    return [{"code": code, "name": name} for code, (name, _) in LEAGUES.items()]


@app.get("/api/fixtures/{league}")
def fixtures(league: str):
    if league not in LEAGUES:
        raise HTTPException(404, f"unknown league {league!r}")
    api_key = os.environ.get("ODDS_API_KEY")
    if not api_key:
        raise HTTPException(503, "ODDS_API_KEY not configured")

    model = get_models()[league]
    matcher = TeamMatcher(list(model.attack))
    events, fetched_at = fetch_events(league, api_key)

    out = []
    for ev in events:
        home = matcher.match(ev["home"])
        away = matcher.match(ev["away"])
        market_h, market_d, market_a, overround = implied_probabilities(
            ev["odds"]["h"], ev["odds"]["d"], ev["odds"]["a"])
        row = {
            "home": ev["home"], "away": ev["away"],
            "commence_time": ev["commence_time"],
            "odds": ev["odds"], "bookmakers": ev["bookmakers"],
            "market": {"h": round(float(market_h), 4), "d": round(float(market_d), 4),
                       "a": round(float(market_a), 4)},
            "overround": round(float(overround) - 1, 4),
            "model": None,
        }
        if home and away:
            p_h, p_d, p_a = model.match_probabilities(home, away)
            row["model"] = {"h": round(p_h, 4), "d": round(p_d, 4), "a": round(p_a, 4)}
        out.append(row)

    return {"league": league, "name": LEAGUES[league][0], "fetched_at": fetched_at,
            "fitted_through": _state["fitted_through"], "fixtures": out}


@lru_cache(maxsize=1)
def _backtest_summary() -> dict:
    csv = PROCESSED_DIR / "backtest_predictions.csv"
    if not csv.exists():
        return {"available": False}
    preds = pd.read_csv(csv)
    preds = preds[preds["known"]]
    y = outcome_index(preds["result"])
    p_model = preds[["p_h", "p_d", "p_a"]].to_numpy()
    bh, bd, ba, _ = implied_probabilities(preds["odds_h"], preds["odds_d"], preds["odds_a"])
    p_book = np.column_stack([bh, bd, ba])

    per_league = []
    for code, g in preds.groupby("league"):
        gy = outcome_index(g["result"])
        gm = g[["p_h", "p_d", "p_a"]].to_numpy()
        gbh, gbd, gba, _ = implied_probabilities(g["odds_h"], g["odds_d"], g["odds_a"])
        per_league.append({
            "league": code, "name": LEAGUES[code][0], "matches": len(g),
            "model_logloss": round(log_loss(gy, gm), 4),
            "book_logloss": round(log_loss(gy, np.column_stack([gbh, gbd, gba])), 4),
        })

    return {
        "available": True,
        "matches": len(preds),
        "model_logloss": round(log_loss(y, p_model), 4),
        "book_logloss": round(log_loss(y, p_book), 4),
        "model_brier": round(brier_score(y, p_model), 4),
        "book_brier": round(brier_score(y, p_book), 4),
        "verdict": "The bookmaker beats the model in every league tested. "
                   "Value betting on model-market disagreements lost ~9.5% ROI "
                   "over 44k simulated bets (2013-2026). No edge.",
        "per_league": per_league,
    }


@app.get("/api/backtest")
def backtest():
    return _backtest_summary()


@app.post("/api/refresh")
def refresh(token: str = ""):
    # if REFRESH_TOKEN is set in the environment, require it as ?token=...
    expected = os.environ.get("REFRESH_TOKEN")
    if expected and token != expected:
        raise HTTPException(403, "bad refresh token")
    from scripts.download_data import download
    download()
    _state["models"], _state["fitted_through"] = model_store.get_models(force_refit=True)
    _backtest_summary.cache_clear()
    return {"status": "refreshed", "fitted_through": _state["fitted_through"]}
