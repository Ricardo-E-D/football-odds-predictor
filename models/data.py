"""Load and clean the raw football-data.co.uk CSVs into one tidy matches table."""

from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# football-data.co.uk renamed the market-average columns from BbAv* to Av* in 2019;
# coalesce both eras into one series.
_ODDS_COLUMNS = {
    "b365_h": ["B365H"], "b365_d": ["B365D"], "b365_a": ["B365A"],
    "ps_h": ["PSH"], "ps_d": ["PSD"], "ps_a": ["PSA"],
    "avg_h": ["AvgH", "BbAvH"], "avg_d": ["AvgD", "BbAvD"], "avg_a": ["AvgA", "BbAvA"],
}


def load_matches(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Return all matches, cleaned and time-sorted.

    Columns: date, season, league, home, away, hg, ag, result,
    b365_h/d/a, ps_h/d/a, avg_h/d/a. Rows missing the result or Bet365
    odds are dropped (they are unusable for both modeling and backtesting).
    """
    frames = []
    for path in sorted(raw_dir.glob("*/*.csv")):
        df = pd.read_csv(path, encoding="latin-1", on_bad_lines="skip")
        out = pd.DataFrame({
            "date": pd.to_datetime(df["Date"], format="mixed", dayfirst=True, errors="coerce"),
            "season": path.parent.name,
            "league": path.stem,
            "home": df["HomeTeam"].astype("string").str.strip(),
            "away": df["AwayTeam"].astype("string").str.strip(),
            "hg": pd.to_numeric(df["FTHG"], errors="coerce"),
            "ag": pd.to_numeric(df["FTAG"], errors="coerce"),
            "result": df["FTR"].astype("string").str.strip(),
        })
        for name, candidates in _ODDS_COLUMNS.items():
            series = pd.Series(pd.NA, index=df.index, dtype="Float64")
            for col in candidates:
                if col in df.columns:
                    series = series.fillna(pd.to_numeric(df[col], errors="coerce"))
            out[name] = series
        frames.append(out)

    matches = pd.concat(frames, ignore_index=True)
    matches = matches.dropna(subset=["date", "home", "away", "hg", "ag", "result"])
    # decimal odds must exceed 1.0; zeros/ones are data errors
    for col in _ODDS_COLUMNS:
        matches.loc[matches[col] <= 1.0, col] = pd.NA
    matches = matches.dropna(subset=["b365_h", "b365_d", "b365_a"])
    matches = matches[matches["result"].isin(["H", "D", "A"])]
    matches[["hg", "ag"]] = matches[["hg", "ag"]].astype(int)
    return matches.sort_values("date", ignore_index=True)
