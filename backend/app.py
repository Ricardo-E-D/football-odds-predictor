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
from datetime import datetime, timezone
from functools import lru_cache

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from backend import model_store, summary
from backend.config import LEAGUES, ROOT
from backend.odds_api import TeamMatcher, fetch_events
from models.odds import implied_probabilities

load_dotenv(ROOT / ".env")

app = FastAPI(title="Football Odds Predictor", version="0.1")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=ROOT / "frontend" / "static"), name="static")
templates = Jinja2Templates(directory=ROOT / "frontend" / "templates")

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


def _build_fixtures(league: str) -> dict:
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


@app.get("/api/fixtures/{league}")
def fixtures(league: str):
    return _build_fixtures(league)


@app.get("/", response_class=HTMLResponse)
def index(request: Request, league: str = "E0"):
    error, data = None, None
    try:
        data = _build_fixtures(league)
    except HTTPException as exc:
        error = exc.detail
    except requests.RequestException as exc:
        error = f"could not reach The Odds API: {exc}"

    if data:
        for f in data["fixtures"]:
            kickoff = datetime.fromisoformat(f["commence_time"].replace("Z", "+00:00"))
            f["kickoff"] = kickoff.strftime("%a %d %b, %H:%M UTC")
            # flag the outcome with the biggest model-market disagreement (if >=5pp)
            f["flag"] = None
            if f["model"]:
                diffs = {k: abs(f["model"][k] - f["market"][k]) for k in "hda"}
                worst = max(diffs, key=diffs.get)
                if diffs[worst] >= 0.05:
                    f["flag"] = worst

    return templates.TemplateResponse(request, "index.html", {
        "leagues": LEAGUES, "current": league, "data": data, "error": error,
        "summary": _backtest_summary(),
        "fetched_ago": _age_label(data["fetched_at"]) if data else None,
    })


@app.get("/chart.png")
def chart():
    path = ROOT / "notebooks" / "backtest_chart.png"
    if not path.exists():
        raise HTTPException(404, "chart not generated")
    return FileResponse(path)


def _age_label(fetched_at: float) -> str:
    minutes = (datetime.now(timezone.utc).timestamp() - fetched_at) / 60
    if minutes < 1:
        return "just now"
    if minutes < 60:
        return f"{minutes:.0f} min ago"
    return f"{minutes / 60:.1f} h ago"


@lru_cache(maxsize=1)
def _backtest_summary() -> dict:
    return summary.load()


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
