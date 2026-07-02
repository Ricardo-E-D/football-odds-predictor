"""The Odds API client: current odds for upcoming fixtures, with a file cache.

The free tier is ~500 credits/month and one league fetch costs one credit,
so responses are cached on disk (default 12h TTL) and fetched lazily per
league — the server never polls on its own.

The Odds API spells team names in full ("Manchester United"); our historical
data uses football-data.co.uk short names ("Man United"). `TeamMatcher` maps
between them with explicit aliases plus fuzzy matching.
"""

import difflib
import json
import os
import time
import unicodedata

import requests

from backend.config import LEAGUES, PROCESSED_DIR

CACHE_FILE = PROCESSED_DIR / "odds_cache.json"
CACHE_TTL_HOURS = float(os.environ.get("ODDS_CACHE_TTL_HOURS", "12"))
BASE_URL = "https://api.the-odds-api.com/v4/sports/{sport}/odds"

# The Odds API name -> football-data.co.uk name, where fuzzy matching fails
ALIASES = {
    "Wolverhampton Wanderers": "Wolves",
    "Nottingham Forest": "Nott'm Forest",
    "Brighton and Hove Albion": "Brighton",
    "West Bromwich Albion": "West Brom",
    "Sheffield Wednesday": "Sheffield Weds",
    "Milton Keynes Dons": "Milton Keynes Dons",
    "Paris Saint Germain": "Paris SG",
    "Borussia Monchengladbach": "M'gladbach",
    "1. FC Koln": "FC Koln",
    "Heart of Midlothian": "Hearts",
    "Hamilton Academical": "Hamilton",
}


def _normalize(name: str) -> str:
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    name = name.lower().replace("&", "and")
    drop = {"fc", "cf", "afc", "ac", "sc", "cd", "ud", "sd", "if", "bk", "1.", "de"}
    return " ".join(w for w in name.split() if w not in drop)


def _token_match(a: str, b: str) -> bool:
    """Tokens match if equal, or one prefixes the other (3+ chars, e.g. man/manchester)."""
    return a == b or (len(a) >= 3 and b.startswith(a)) or (len(b) >= 3 and a.startswith(b))


def _tokens_compatible(short: list[str], long: list[str]) -> bool:
    """True if every token of `short`, in order, matches a distinct token of
    `long` — so 'man united' matches 'manchester united' but 'united' alone
    matches nothing specific."""
    it = iter(long)
    return all(any(_token_match(tok, t) for t in it) for tok in short)


class TeamMatcher:
    """Map an Odds API team name onto our historical team names."""

    def __init__(self, known_teams: list[str]):
        self._by_norm = {_normalize(t): t for t in known_teams}

    def match(self, api_name: str) -> str | None:
        if api_name in ALIASES and ALIASES[api_name] in self._by_norm.values():
            return ALIASES[api_name]
        norm = _normalize(api_name)
        if norm in self._by_norm:
            return self._by_norm[norm]

        # token-prefix match, accepted only when unambiguous
        a = norm.split()
        candidates = [
            known for cand_norm, known in self._by_norm.items()
            if (b := cand_norm.split()) and
            _tokens_compatible(*((a, b) if len(a) <= len(b) else (b, a)))
        ]
        if len(candidates) == 1:
            return candidates[0]

        close = difflib.get_close_matches(norm, self._by_norm.keys(), n=1, cutoff=0.75)
        return self._by_norm[close[0]] if close else None


def _load_cache() -> dict:
    if CACHE_FILE.exists():
        return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    return {}


def fetch_events(league: str, api_key: str) -> tuple[list[dict], float]:
    """Return (events, fetched_at_unix) for a league, from cache when fresh.

    Each event: {home, away, commence_time, odds: {h, d, a}, bookmakers: n}
    with odds averaged across all bookmakers quoting the match (consensus
    price — more stable than any single book).
    """
    cache = _load_cache()
    entry = cache.get(league)
    if entry and time.time() - entry["fetched_at"] < CACHE_TTL_HOURS * 3600:
        return entry["events"], entry["fetched_at"]

    sport = LEAGUES[league][1]
    resp = requests.get(
        BASE_URL.format(sport=sport),
        params={"apiKey": api_key, "regions": "eu", "markets": "h2h", "oddsFormat": "decimal"},
        timeout=30,
    )
    resp.raise_for_status()

    events = []
    for ev in resp.json():
        prices = {"h": [], "d": [], "a": []}
        for bm in ev.get("bookmakers", []):
            for market in bm.get("markets", []):
                if market["key"] != "h2h":
                    continue
                for outcome in market["outcomes"]:
                    if outcome["name"] == ev["home_team"]:
                        prices["h"].append(outcome["price"])
                    elif outcome["name"] == ev["away_team"]:
                        prices["a"].append(outcome["price"])
                    else:
                        prices["d"].append(outcome["price"])
        if not all(prices.values()):
            continue
        events.append({
            "home": ev["home_team"],
            "away": ev["away_team"],
            "commence_time": ev["commence_time"],
            "odds": {k: round(sum(v) / len(v), 3) for k, v in prices.items()},
            "bookmakers": len(ev.get("bookmakers", [])),
        })

    fetched_at = time.time()
    cache[league] = {"fetched_at": fetched_at, "events": events}
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(cache), encoding="utf-8")
    return events, fetched_at
