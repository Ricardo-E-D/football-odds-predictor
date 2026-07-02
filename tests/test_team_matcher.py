from backend.odds_api import TeamMatcher

E0_TEAMS = ["Man United", "Man City", "Wolves", "Nott'm Forest", "Brighton",
            "Newcastle", "Tottenham", "West Ham", "Leeds", "Arsenal", "Liverpool"]


def test_full_names_map_to_short_names():
    m = TeamMatcher(E0_TEAMS)
    assert m.match("Manchester United") == "Man United"
    assert m.match("Manchester City") == "Man City"
    assert m.match("Wolverhampton Wanderers") == "Wolves"
    assert m.match("Nottingham Forest") == "Nott'm Forest"
    assert m.match("Brighton and Hove Albion") == "Brighton"
    assert m.match("Newcastle United") == "Newcastle"
    assert m.match("Tottenham Hotspur") == "Tottenham"
    assert m.match("Leeds United") == "Leeds"


def test_exact_names_pass_through():
    m = TeamMatcher(E0_TEAMS)
    assert m.match("Arsenal") == "Arsenal"
    assert m.match("Liverpool") == "Liverpool"


def test_unknown_team_returns_none():
    m = TeamMatcher(E0_TEAMS)
    assert m.match("Real Madrid") is None
