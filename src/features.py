from pathlib import Path
import pandas as pd
import numpy as np


def build_pitcher_games_table(statcast_df):
    """Aggregate Statcast pitch-level data into one row per pitcher appearance."""

    df = statcast_df.copy()

    df["is_called_strike"] = (df["description"] == "called_strike")

    df["is_whiff"] = (df["description"].isin(["swinging_strike", "swinging_strike_blocked"]))

    df["is_csw"] = (df["is_called_strike"] | df["is_whiff"])

    pitcher_games = (
        df.groupby(
            ["game_date", "game_pk", "pitcher", "player_name"]
        )
        .agg(
            pitches=("pitcher", "count"),
            strikeouts=("events", lambda x: (x == "strikeout").sum()),
            avg_velocity=("release_speed", "mean"),
            avg_spin=("release_spin_rate", "mean"),
            avg_break_x=("pfx_x", "mean"),
            avg_break_z=("pfx_z", "mean"),
            csw=("is_csw", "sum"),
            whiffs=("is_whiff", "sum"),
            called_strikes=("is_called_strike", "sum"),
            rest_days=("pitcher_days_since_prev_game", "first"),
            max_times_through_order=("n_thruorder_pitcher", "max"),
        )
        .reset_index()
    )

    pitcher_games["CSW%"] = (pitcher_games["csw"] / pitcher_games["pitches"])

    game_teams = (df[["game_pk", "home_team", "away_team",]].drop_duplicates("game_pk"))

    pitcher_games = pitcher_games.merge(game_teams, on="game_pk", how="left",)

    pitching_team = (df.groupby(["game_pk", "pitcher"])["inning_topbot"].first().reset_index())

    pitcher_games = pitcher_games.merge(pitching_team, on=["game_pk", "pitcher"], how="left",)

    pitcher_games["pitcher_team"] = np.where(pitcher_games["inning_topbot"] == "Top", pitcher_games["home_team"], pitcher_games["away_team"],)

    pitcher_games["opponent"] = np.where(pitcher_games["pitcher_team"] == pitcher_games["home_team"], pitcher_games["away_team"], pitcher_games["home_team"],)

    pitcher_games = pitcher_games.drop(columns=["inning_topbot"])

    return pitcher_games



def add_rolling_features(df):
    """
    Add leakage-safe rolling features.

    Every rolling feature only uses games BEFORE the current game.
    """

    df = df.copy()

    df["game_date"] = pd.to_datetime(df["game_date"])

    df = df.sort_values(["pitcher", "game_date", "game_pk"])

    group = df.groupby("pitcher")

    rolling_features = {
        "strikeouts": "k_last3",
        "avg_velocity": "velo_last3",
        "avg_spin": "spin_last3",
        "CSW%": "csw_last3",
        "pitches": "pitches_last3"
    }

    for original, new_name in rolling_features.items():

        df[new_name] = (group[original].transform(lambda s: s.shift(1).rolling(window=3,min_periods=1).mean()))

    df["velo_last6"] = (group["avg_velocity"].transform(lambda s: s.shift(1).rolling(6, min_periods=3).mean()))

    df["velo_trend"] = (df["velo_last3"] - df["velo_last6"])

    df["k_std_last5"] = (group["strikeouts"].transform(lambda s: s.shift(1).rolling(5, min_periods=2).std()))

    df["pitches_last5"] = (group["pitches"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean()))

    df["whiff_last3"] = (group["whiffs"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean()))

    df["csw_last5"] = (group["CSW%"].transform(lambda s: s.shift(1).rolling(5, min_periods=2).mean()))

    return df


def identify_starters(df):
    """
    Identify starting pitchers from Statcast data.

    A starter is the first pitcher to appear for a team in a game.
    """

    first_pitchers = (df.sort_values(["game_pk", "inning", "at_bat_number", "pitch_number"]).groupby(["game_pk", "pitcher"]).first().reset_index())

    starters = (first_pitchers[first_pitchers["inning"] == 1][["game_pk", "pitcher"]].drop_duplicates())

    starters["is_starter"] = True

    return starters


def add_pitch_mix_features(statcast_df, pitcher_games):
    """Add pitch usage percentages for each pitcher appearance."""

    pitch_counts = ( statcast_df .groupby( [ "game_pk", "pitcher", "pitch_type" ] ) .size() .reset_index(name="pitch_count") )

    pitch_counts["total_pitches"] = ( pitch_counts .groupby( [ "game_pk", "pitcher" ] )["pitch_count"] .transform("sum") )

    pitch_counts["usage"] = ( pitch_counts["pitch_count"] / pitch_counts["total_pitches"] )

    pitch_mix = ( pitch_counts .pivot_table( index=[ "game_pk", "pitcher" ], columns="pitch_type", values="usage", fill_value=0 ) .reset_index() )

    pitch_mix.columns = [ f"{col}_usage" if col not in ["game_pk", "pitcher"] else col for col in pitch_mix.columns ]

    pitcher_games = pitcher_games.merge( pitch_mix, on=[ "game_pk", "pitcher" ], how="left" )

    return pitcher_games


def build_team_game_stats(statcast_df):

    df = statcast_df.copy()

    pa_events = df[df["events"].notna()].copy()

    home = pa_events.copy()
    home["team"] = home["home_team"]
    home["opponent"] = home["away_team"]
    home["runs"] = home["home_score"]

    away = pa_events.copy()
    away["team"] = away["away_team"]
    away["opponent"] = away["home_team"]
    away["runs"] = away["away_score"]

    batting = pd.concat([home, away])

    batting["hit"] = batting["events"].isin(["single", "double", "triple", "home_run"])

    batting["home_run"] = batting["events"].eq("home_run")

    batting["strikeout"] = batting["events"].isin(["strikeout", "strikeout_double_play"])

    batting["walk"] = batting["events"].isin(["walk", "intent_walk"])


    team_games = (batting.groupby(["game_date", "game_pk", "team", "opponent"])
        .agg(
            runs=("runs", "max"),
            plate_appearances=("events", "count"),
            hits=("hit", "sum"),
            home_runs=("home_run", "sum"),
            strikeouts=("strikeout", "sum"),
            walks=("walk", "sum"),
            iso=("iso_value", "sum")
        )
        .reset_index()
    )

    team_games["k_rate"] = (team_games["strikeouts"] / team_games["plate_appearances"])

    team_games["bb_rate"] = (team_games["walks"] / team_games["plate_appearances"])

    team_games["hr_rate"] = (team_games["home_runs"] / team_games["plate_appearances"])

    team_games["iso"] = (team_games["iso"] / team_games["plate_appearances"])

    return team_games


def add_team_rolling_features(team_games):

    team_games = team_games.sort_values(["team", "game_date", "game_pk"])

    metrics = ["runs", "k_rate", "bb_rate", "hr_rate", "iso"]

    for metric in metrics:
        team_games[f"{metric}_last14"] = (team_games.groupby("team")[metric].transform(lambda x: x.shift(1).rolling(14,min_periods=3).mean()))

    return team_games


def add_historical_features(df, columns, rolling_windows=(3, 5)):
    """
    Creates leakage-free historical pitcher features.
    """

    df = df.copy()

    df = df.sort_values(["pitcher", "game_date", "game_pk"])

    grouped = df.groupby("pitcher")

    for col in columns:
        df[f"last_{col}"] = (grouped[col].shift(1))

        for window in rolling_windows:
            df[f"rolling{window}_{col}"] = (grouped[col].shift(1).rolling(window, min_periods=1).mean().reset_index(level=0, drop=True))

        df[f"season_{col}"] = (grouped[col].shift(1).expanding().mean().reset_index(level=0, drop=True))

    return df


def add_opponent_features(pitcher_games, team_games):
    """
    Merge leakage-safe opponent offensive features into pitcher-game data.

    The opponent's rolling offensive statistics are calculated from games
    BEFORE the current game, so no future information is used.
    """

    pitcher_games = pitcher_games.copy()
    team_games = team_games.copy()

    pitcher_games["game_date"] = pd.to_datetime(pitcher_games["game_date"])
    team_games["game_date"] = pd.to_datetime(team_games["game_date"])

    opponent_features = team_games[
        [
            "game_date",
            "game_pk",
            "team",
            "opponent",
            "runs_last14",
            "k_rate_last14",
            "bb_rate_last14",
            "hr_rate_last14",
            "iso_last14",
        ]
    ].copy()

    opponent_features = opponent_features.rename(
        columns={
            "team": "opponent",
            "opponent": "pitcher_team",
            "runs_last14": "opp_runs_last14",
            "k_rate_last14": "opp_k_rate_last14",
            "bb_rate_last14": "opp_bb_rate_last14",
            "hr_rate_last14": "opp_hr_rate_last14",
            "iso_last14": "opp_iso_last14",
        }
    )

    pitcher_games = pitcher_games.merge(
        opponent_features[
            [
                "game_date",
                "game_pk",
                "opponent",
                "opp_runs_last14",
                "opp_k_rate_last14",
                "opp_bb_rate_last14",
                "opp_hr_rate_last14",
                "opp_iso_last14",
            ]
        ],
        on=["game_date", "game_pk", "opponent"],
        how="left",
    )

    return pitcher_games


def add_park_factors(pitcher_games):
    """
    Add manually verified 2024 FanGraphs park factors.

    These values were manually transcribed from FanGraphs
    because the site could not be reliably scraped.

    The same park-factor values were used when training
    the current XGBoost model.
    """

    park_factors = {
        "ATH": (0.97, 0.90),
        "ATL": (1.00, 0.99),
        "AZ": (1.00, 0.91),
        "BAL": (0.98, 0.99),
        "BOS": (1.02, 0.98),
        "CHC": (0.96, 0.98),
        "CIN": (1.02, 1.14),
        "CLE": (1.00, 0.98),
        "COL": (1.14, 1.07),
        "CWS": (0.98, 1.05),
        "DET": (1.03, 0.96),
        "HOU": (1.00, 1.02),
        "KC": (1.03, 0.95),
        "LAA": (1.01, 1.05),
        "LAD": (0.98, 1.10),
        "MIA": (1.02, 0.97),
        "MIL": (0.98, 1.04),
        "MIN": (1.03, 0.99),
        "NYM": (0.98, 0.99),
        "NYY": (0.99, 1.04),
        "PHI": (1.02, 1.05),
        "PIT": (1.02, 0.93),
        "SD": (0.97, 1.01),
        "SEA": (0.92, 0.96),
        "SF": (0.96, 0.91),
        "STL": (0.99, 0.94),
        "TB": (0.98, 0.96),
        "TEX": (0.97, 1.02),
        "TOR": (1.00, 1.03),
        "WSH": (1.00, 1.00),
    }

    park_factor_df = pd.DataFrame(
        [
            {
                "home_team": team,
                "park_run_factor": run_factor,
                "park_hr_factor": hr_factor,
            }
            for team, (run_factor, hr_factor)
            in park_factors.items()
        ]
    )

    pitcher_games = pitcher_games.merge(park_factor_df, on="home_team", how="left")

    return pitcher_games


def add_bullpen_features(pitcher_games, pitching_targets):
    """
    Calculate bullpen ERA/FIP features.

    Bullpen statistics for a game are calculated from relief pitchers only.
    """

    pitcher_games = pitcher_games.copy()
    pitching_targets = pitching_targets.copy()

    pitcher_games["game_date"] = pd.to_datetime(pitcher_games["game_date"])

    target_team_info = pitcher_games[["game_pk", "pitcher", "game_date", "pitcher_team", "is_starter"]].copy()

    bullpen = pitching_targets.merge(target_team_info, on=["game_pk", "pitcher"], how="left")

    bullpen = bullpen[bullpen["pitcher_team"].notna()].copy()

    bullpen = bullpen[bullpen["is_starter"] == 0].copy()

    bullpen_game = (bullpen.groupby(["game_date","game_pk","pitcher_team",])
        .agg(
            bullpen_outs=("outs", "sum"),
            bullpen_er=("earned_runs", "sum"),
            bullpen_bb=("walks", "sum"),
            bullpen_hbp=("hit_batters", "sum"),
            bullpen_k=("strikeouts_official", "sum"),
            bullpen_hr=("home_runs_allowed", "sum"),
        )
        .reset_index()
    )

    bullpen_game["bullpen_ip"] = (bullpen_game["bullpen_outs"] / 3.0)

    bullpen_game["bullpen_era"] = np.where(bullpen_game["bullpen_ip"] > 0,(bullpen_game["bullpen_er"] / bullpen_game["bullpen_ip"]) * 9, np.nan,)

    bullpen_game["bullpen_fip"] = np.where(bullpen_game["bullpen_ip"] > 0,((13 * bullpen_game["bullpen_hr"] + 3 * (bullpen_game["bullpen_bb"] + bullpen_game["bullpen_hbp"]) - 2 * bullpen_game["bullpen_k"]) / bullpen_game["bullpen_ip"]), np.nan)

    bullpen_game = bullpen_game.sort_values(["pitcher_team", "game_date", "game_pk"])

    group = bullpen_game.groupby("pitcher_team")

    bullpen_game["bullpen_era_last14"] = (group["bullpen_era"].transform(lambda x: x.shift(1).rolling(14, min_periods=3).mean()))

    bullpen_game["bullpen_fip_last14"] = (group["bullpen_fip"].transform(lambda x: x.shift(1).rolling(14, min_periods=3).mean()))

    return bullpen_game[["game_date", "game_pk", "pitcher_team", "bullpen_era_last14", "bullpen_fip_last14"]]


def add_win_probability_features(pitcher_games, team_games, bullpen_games):
    """
    Add team and bullpen features.
    """

    pitcher_games = pitcher_games.copy()
    team_games = team_games.copy()
    bullpen_games = bullpen_games.copy()

    team_features = team_games[["game_date", "game_pk", "team", "runs_last14"]].copy()

    team_features = team_features.rename(columns={"team": "pitcher_team", "runs_last14": "team_runs_last14",})

    pitcher_games = pitcher_games.merge(team_features, on=["game_date", "game_pk", "pitcher_team",], how="left")

    opponent_features = team_games[["game_date", "game_pk", "team", "runs_last14"]].copy()

    opponent_features = opponent_features.rename(columns={"team": "opponent", "runs_last14": "opp_team_runs_last14"})

    pitcher_games = pitcher_games.merge(opponent_features, on=["game_date", "game_pk", "opponent"], how="left")

    bullpen_features = bullpen_games.rename(columns={"pitcher_team": "pitcher_team"})

    pitcher_games = pitcher_games.merge(bullpen_features, on=["game_date", "game_pk", "pitcher_team"], how="left")

    opponent_bullpen = bullpen_games.rename(columns={"pitcher_team": "opponent", "bullpen_era_last14": "opp_bullpen_era_last14", "bullpen_fip_last14": "opp_bullpen_fip_last14",})

    pitcher_games = pitcher_games.merge(opponent_bullpen, on=["game_date", "game_pk", "opponent"], how="left",)

    pitcher_games["run_support_diff"] = (pitcher_games["team_runs_last14"] - pitcher_games["opp_team_runs_last14"])

    pitcher_games["bullpen_era_diff"] = (pitcher_games["opp_bullpen_era_last14"] - pitcher_games["bullpen_era_last14"])

    pitcher_games["bullpen_fip_diff"] = (pitcher_games["opp_bullpen_fip_last14"] - pitcher_games["bullpen_fip_last14"])

    return pitcher_games
