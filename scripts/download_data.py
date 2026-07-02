"""Download historical results + odds CSVs from football-data.co.uk.

Files land in data/raw/<season>/<league>.csv. Existing files are skipped,
except the current season, which is always re-downloaded so new match rounds
get picked up. Run with --force to re-download everything.

Usage:
    python scripts/download_data.py [--force]
"""

import argparse
import sys
import time
from pathlib import Path

import requests

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# Leagues chosen for softer (less efficiently priced) markets, plus the
# Premier League as an efficient-market baseline for the backtest.
LEAGUES = {
    "E0": "England Premier League",
    "E1": "England Championship",
    "E2": "England League One",
    "E3": "England League Two",
    "SC0": "Scotland Premiership",
    "D2": "Germany 2. Bundesliga",
    "I2": "Italy Serie B",
    "SP2": "Spain Segunda Division",
    "F2": "France Ligue 2",
    "N1": "Netherlands Eredivisie",
    "B1": "Belgium Pro League",
    "P1": "Portugal Primeira Liga",
    "T1": "Turkey Super Lig",
    "G1": "Greece Super League",
}

FIRST_SEASON = 2010  # 2010/11 onward: consistent odds coverage across books
LAST_SEASON = 2025   # 2025/26 — bump when a new season starts


def season_code(start_year: int) -> str:
    """2010 -> '1011' (football-data.co.uk's season URL format)."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def download(force: bool = False) -> list[str]:
    session = requests.Session()
    session.headers["User-Agent"] = "football-odds-predictor (personal research)"
    ok, skipped, failed = 0, 0, []

    for year in range(FIRST_SEASON, LAST_SEASON + 1):
        season = season_code(year)
        season_dir = RAW_DIR / season
        season_dir.mkdir(parents=True, exist_ok=True)
        is_current = year == LAST_SEASON

        for league in LEAGUES:
            dest = season_dir / f"{league}.csv"
            if dest.exists() and not force and not is_current:
                skipped += 1
                continue

            url = BASE_URL.format(season=season, league=league)
            resp = session.get(url, timeout=30)
            if resp.status_code != 200 or not resp.content.strip():
                failed.append(f"{season}/{league} (HTTP {resp.status_code})")
                continue

            dest.write_bytes(resp.content)
            ok += 1
            print(f"downloaded {season}/{league}.csv ({len(resp.content):,} bytes)")
            time.sleep(0.3)  # be polite to a free service

    print(f"\n{ok} downloaded, {skipped} skipped (already present), {len(failed)} failed")
    if failed:
        print("failed:", ", ".join(failed))
    return failed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download all files")
    if download(force=parser.parse_args().force):
        sys.exit(1)
