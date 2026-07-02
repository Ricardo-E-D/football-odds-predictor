"""Fit, persist, and load the per-league Poisson models.

Models are refit only when the historical data has newer matches than the
saved parameters — so server restarts are cheap and refresh is explicit.
"""

import json
from datetime import date

import pandas as pd

from backend.config import PROCESSED_DIR
from models.data import load_matches
from models.poisson import PoissonModel

PARAMS_FILE = PROCESSED_DIR / "model_params.json"
XI = 0.0019  # time-decay: ~one-year half-life, same as the backtest


def fit_all(matches: pd.DataFrame) -> tuple[dict[str, PoissonModel], str]:
    models = {}
    for league, lg in matches.groupby("league"):
        models[league] = PoissonModel(xi=XI).fit(lg)
    fitted_through = str(matches["date"].max().date())
    return models, fitted_through


def save(models: dict[str, PoissonModel], fitted_through: str) -> None:
    PARAMS_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "fitted_through": fitted_through,
        "leagues": {code: m.to_dict() for code, m in models.items()},
    }
    PARAMS_FILE.write_text(json.dumps(payload), encoding="utf-8")


def load_saved() -> tuple[dict[str, PoissonModel], str] | None:
    if not PARAMS_FILE.exists():
        return None
    payload = json.loads(PARAMS_FILE.read_text(encoding="utf-8"))
    models = {code: PoissonModel.from_dict(d) for code, d in payload["leagues"].items()}
    return models, payload["fitted_through"]


def get_models(force_refit: bool = False) -> tuple[dict[str, PoissonModel], str]:
    """Load saved models if they cover all downloaded data, else refit and save.

    On a deployed server there is no raw data — the committed parameter file
    is the only source, so it is served as-is.
    """
    saved = None if force_refit else load_saved()
    try:
        matches = load_matches()
    except ValueError:  # no raw CSVs on disk (ephemeral hosting)
        if saved is not None:
            return saved
        raise RuntimeError("no raw data and no saved model parameters") from None
    latest = str(matches["date"].max().date())
    if saved is not None and saved[1] >= latest:
        return saved
    models, fitted_through = fit_all(matches)
    save(models, fitted_through)
    return models, fitted_through
