"""League metadata and shared paths."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"

# code -> (display name, The Odds API sport key)
LEAGUES = {
    "E0": ("Premier League", "soccer_epl"),
    "E1": ("Championship", "soccer_efl_champ"),
    "E2": ("League One", "soccer_england_league1"),
    "E3": ("League Two", "soccer_england_league2"),
    "SC0": ("Scottish Premiership", "soccer_spl"),
    "D2": ("2. Bundesliga", "soccer_germany_bundesliga2"),
    "I2": ("Serie B", "soccer_italy_serie_b"),
    "SP2": ("La Liga 2", "soccer_spain_segunda_division"),
    "F2": ("Ligue 2", "soccer_france_ligue_two"),
    "N1": ("Eredivisie", "soccer_netherlands_eredivisie"),
    "B1": ("Belgian Pro League", "soccer_belgium_first_div"),
    "P1": ("Primeira Liga", "soccer_portugal_primeira_liga"),
    "T1": ("Süper Lig", "soccer_turkey_super_league"),
    "G1": ("Greek Super League", "soccer_greece_super_league"),
}
