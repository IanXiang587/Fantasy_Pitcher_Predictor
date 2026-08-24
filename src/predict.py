from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODELS = PROJECT_ROOT / "models"
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "pitcher_games_features_base.csv"
TEAM_FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "team_game_features.csv"
BULLPEN_FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "bullpen_game_features.csv"


def load_component_models():
    """Load all component regression models and the shared feature list."""

    models = {
        "strikeouts": joblib.load(MODELS / "xgboost.pkl"),
        "outs": joblib.load(MODELS / "outs_model.pkl"),
        "hits": joblib.load(MODELS / "hits_model.pkl"),
        "earned_runs": joblib.load(MODELS / "er_model.pkl"),
        "walks": joblib.load(MODELS / "bb_model.pkl"),
        "hit_batters": joblib.load(MODELS / "hbp_model.pkl"),
    }

    with open(MODELS / "model_features.json", "r") as f:
        feature_cols = json.load(f)

    return models, feature_cols


def load_win_model():
    """Load the win-probability model and its feature list."""

    model = joblib.load(MODELS / "win_model.pkl")

    with open(MODELS / "win_model_features.json", "r") as f:
        feature_cols = json.load(f)

    return model, feature_cols


def load_historical_data(data_path=None):
    """Load the historical pitcher feature table."""

    if data_path is None:
        data_path = DEFAULT_DATA_PATH

    data = pd.read_csv(data_path)
    data["game_date"] = pd.to_datetime(data["game_date"])

    return data


def load_win_feature_data():
    """Load the team and bullpen feature tables used by the win model."""

    team_games = pd.read_csv(TEAM_FEATURES_PATH)
    bullpen_games = pd.read_csv(BULLPEN_FEATURES_PATH)

    team_games["game_date"] = pd.to_datetime(team_games["game_date"])
    bullpen_games["game_date"] = pd.to_datetime(bullpen_games["game_date"])

    return team_games, bullpen_games


def normalize_pitcher_name(name):
    """
    Convert MLB Stats API names into the naming format used
    by the historical Statcast data.

    Examples:
        Shane Baz -> Baz, Shane
        José Soriano -> Soriano, José
        Daniel Lynch IV -> Lynch IV, Daniel
    """

    if pd.isna(name):
        return name

    name = str(name).strip()

    if "," in name:
        return name

    parts = name.split()

    if len(parts) < 2:
        return name

    suffixes = {
        "Jr.",
        "Jr",
        "Sr.",
        "Sr",
        "II",
        "III",
        "IV",
        "V",
    }

    if parts[-1] in suffixes:
        last_name = f"{parts[-2]} {parts[-1]}"
        first_name = " ".join(parts[:-2])
    else:
        last_name = parts[-1]
        first_name = " ".join(parts[:-1])

    return f"{last_name}, {first_name}"


def get_pitcher_history(historical_data, pitcher_name, game_date):
    """Return all pitcher games before the prediction date."""

    game_date = pd.Timestamp(game_date)
    pitcher_name = normalize_pitcher_name(pitcher_name)

    history = historical_data[(historical_data["player_name"] == pitcher_name) & (historical_data["game_date"] < game_date)].copy()

    return history.sort_values(["game_date", "game_pk"])


def get_pitcher_team(historical_data, pitcher_name, game_date):
    """Get the pitcher's most recent team before the prediction date."""

    pitcher_history = get_pitcher_history(historical_data, pitcher_name, game_date)

    if pitcher_history.empty:
        raise ValueError(f"No historical data found for {pitcher_name}")

    latest = pitcher_history.sort_values("game_date").iloc[-1]

    if ("pitcher_team" in pitcher_history.columns and pd.notna(latest["pitcher_team"])):
        return latest["pitcher_team"]

    home_team = latest.get("home_team")
    away_team = latest.get("away_team")
    opponent = latest.get("opponent")

    if pd.isna(home_team) or pd.isna(away_team) or pd.isna(opponent):
        raise ValueError(f"Missing team information for {pitcher_name}. Home={home_team}, Away={away_team}, Opponent={opponent}")

    if opponent == home_team:
        return away_team

    if opponent == away_team:
        return home_team

    raise ValueError(f"Could not determine team for {pitcher_name}. Home={home_team}, Away={away_team}, Opponent={opponent}")


def get_opponent_features(historical_data, opponent, game_date):
    """Get opponent's most recent pregame offensive metrics."""

    game_date = pd.Timestamp(game_date)

    opponent_history = historical_data[(historical_data["opponent"] == opponent) & (historical_data["game_date"] < game_date)].copy()

    if opponent_history.empty:
        raise ValueError(f"No historical opponent data found for {opponent}")

    latest = opponent_history.sort_values(["game_date", "game_pk"]).iloc[-1]

    return {
        "opp_runs_last14": latest["opp_runs_last14"],
        "opp_k_rate_last14": latest["opp_k_rate_last14"],
        "opp_bb_rate_last14": latest["opp_bb_rate_last14"],
        "opp_hr_rate_last14": latest["opp_hr_rate_last14"],
        "opp_iso_last14": latest["opp_iso_last14"],
    }


def get_park_factors(historical_data, home_team, game_date):
    """Get the most recent park factors for the home stadium."""

    game_date = pd.Timestamp(game_date)

    park_history = historical_data[(historical_data["home_team"] == home_team) & (historical_data["game_date"] < game_date)].copy()

    if park_history.empty:
        raise ValueError(f"No historical park data found for {home_team}")

    latest = park_history.sort_values(["game_date", "game_pk"]).iloc[-1]

    return {
        "park_run_factor": latest["park_run_factor"],
        "park_hr_factor": latest["park_hr_factor"],
    }


def build_prediction_features(pitcher_name, opponent, game_date, historical_data, feature_cols, location="Away",):
    """Build leakage-safe pregame features for one pitcher."""

    game_date = pd.Timestamp(game_date)

    historical_data = historical_data.copy()
    historical_data["game_date"] = pd.to_datetime(historical_data["game_date"])

    pitcher_history = get_pitcher_history(historical_data, pitcher_name, game_date)

    if pitcher_history.empty:
        raise ValueError(f"No historical data found for {pitcher_name}")

    latest = pitcher_history.iloc[-1].copy()

    features = latest.reindex(feature_cols).copy()

    features["is_starter"] = 1

    features["rest_days"] = (game_date - latest["game_date"]).days

    opponent_features = get_opponent_features(historical_data, opponent, game_date)

    for column, value in opponent_features.items():
        if column in features.index:
            features[column] = value

    pitcher_team = get_pitcher_team(historical_data, pitcher_name, game_date,)

    if location == "Home":
        home_team = pitcher_team
    else:
        home_team = opponent

    park_features = get_park_factors(historical_data, home_team, game_date,)

    for column, value in park_features.items():
        if column in features.index:
            features[column] = value

    result = pd.DataFrame([features], columns=feature_cols)

    result = result.replace({pd.NA: np.nan})

    for col in result.columns:
        result[col] = pd.to_numeric(result[col], errors="coerce")

    return result


def _get_latest_team_features(team_features, team, game_date,):
    """Get latest team-level features available before a game."""

    history = team_features[(team_features["team"] == team) & (team_features["game_date"] < game_date)].sort_values(["game_date", "game_pk"])

    if history.empty:
        raise ValueError(f"No historical team features found for {team}")

    return history.iloc[-1]


def _get_latest_bullpen_features(bullpen_features, team, game_date,):
    """Get latest bullpen features available before a game."""

    history = bullpen_features[(bullpen_features["pitcher_team"] == team) & (bullpen_features["game_date"] < game_date)].sort_values(["game_date", "game_pk"])

    if history.empty:
        raise ValueError(f"No historical bullpen features found for {team}")

    return history.iloc[-1]


def _enrich_win_pitcher_features(pitcher_features, pitcher_team, opponent, game_date, team_features, bullpen_features,):
    """Add team and bullpen information to a pitcher feature vector."""

    pitcher_features = pitcher_features.copy()

    team_latest = _get_latest_team_features(team_features, pitcher_team, game_date)

    opponent_latest = _get_latest_team_features(team_features, opponent, game_date)

    bullpen_latest = _get_latest_bullpen_features(bullpen_features, pitcher_team, game_date)

    opponent_bullpen_latest = _get_latest_bullpen_features(bullpen_features, opponent, game_date)

    pitcher_features["team_runs_last14"] = (team_latest["runs_last14"])

    pitcher_features["opp_team_runs_last14"] = (opponent_latest["runs_last14"])

    pitcher_features["bullpen_era_last14"] = (bullpen_latest["bullpen_era_last14"])

    pitcher_features["opp_bullpen_era_last14"] = (opponent_bullpen_latest["bullpen_era_last14"])

    pitcher_features["bullpen_fip_last14"] = (bullpen_latest["bullpen_fip_last14"])

    pitcher_features["opp_bullpen_fip_last14"] = (opponent_bullpen_latest["bullpen_fip_last14"])

    pitcher_features["run_support_diff"] = (pitcher_features["team_runs_last14"] - pitcher_features["opp_team_runs_last14"])

    pitcher_features["bullpen_era_diff"] = (pitcher_features["opp_bullpen_era_last14"] - pitcher_features["bullpen_era_last14"])

    pitcher_features["bullpen_fip_diff"] = (pitcher_features["opp_bullpen_fip_last14"] - pitcher_features["bullpen_fip_last14"])

    return pitcher_features


def build_win_prediction_features(home_pitcher_features, away_pitcher_features, home_team, away_team, game_date, team_features, bullpen_features, win_feature_cols, home_component_predictions, away_component_predictions):
    """
    Construct the exact HOME-vs-AWAY representation used by the
    game-level win model during training.
    """

    game_date = pd.Timestamp(game_date)

    if isinstance(home_pitcher_features, pd.Series):
        home_pitcher_features = home_pitcher_features.copy()
    else:
        home_pitcher_features = pd.Series(home_pitcher_features)

    if isinstance(away_pitcher_features, pd.Series):
        away_pitcher_features = away_pitcher_features.copy()
    else:
        away_pitcher_features = pd.Series(away_pitcher_features)

    home_pitcher_features = _enrich_win_pitcher_features(home_pitcher_features, home_team, away_team, game_date, team_features, bullpen_features)
    away_pitcher_features = _enrich_win_pitcher_features(away_pitcher_features, away_team, home_team, game_date, team_features, bullpen_features)

    difference_features = [
        "team_runs_last14",
        "opp_team_runs_last14",
        "bullpen_era_last14",
        "opp_bullpen_era_last14",
        "bullpen_fip_last14",
        "opp_bullpen_fip_last14",
        "run_support_diff",
        "bullpen_era_diff",
        "bullpen_fip_diff",
        "k_last3",
        "velo_last3",
        "spin_last3",
        "csw_last3",
        "pitches_last3",
        "velo_last6",
        "velo_trend",
        "k_std_last5",
        "pitches_last5",
        "whiff_last3",
        "csw_last5",
    ]

    matchup_features = [
        "opp_runs_last14",
        "opp_k_rate_last14",
        "opp_bb_rate_last14",
        "opp_hr_rate_last14",
        "opp_iso_last14",
        "park_run_factor",
        "park_hr_factor",
        "rest_days",
    ]

    row = {}

    for col in difference_features:
        if col not in home_pitcher_features.index:
            raise ValueError(f"Missing home win feature: {col}")
        if col not in away_pitcher_features.index:
            raise ValueError(f"Missing away win feature: {col}")

        home_value = pd.to_numeric(home_pitcher_features[col], errors="coerce")
        away_value = pd.to_numeric(away_pitcher_features[col], errors="coerce")

        if pd.notna(home_value) and pd.notna(away_value):
            row[f"diff_{col}"] = home_value - away_value
        else:
            row[f"diff_{col}"] = np.nan

    for col in matchup_features:
        if col not in home_pitcher_features.index:
            raise ValueError(f"Missing home win feature: {col}")
        if col not in away_pitcher_features.index:
            raise ValueError(f"Missing away win feature: {col}")

        row[f"home_{col}"] = pd.to_numeric(home_pitcher_features[col], errors="coerce")
        row[f"away_{col}"] = pd.to_numeric(away_pitcher_features[col], errors="coerce")

    required_targets = [ "strikeouts", "outs", "hits", "earned_runs", "walks", "hit_batters", ]

    for target in required_targets:
        if target not in home_component_predictions:
            raise ValueError( f"Missing home component prediction: {target}" )
        if target not in away_component_predictions:
            raise ValueError( f"Missing away component prediction: {target}" )

    row["diff_projected_strikeouts"] = (home_component_predictions["strikeouts"] - away_component_predictions["strikeouts"])
    row["diff_projected_outs"] = (home_component_predictions["outs"] - away_component_predictions["outs"])
    row["diff_projected_hits"] = (home_component_predictions["hits"] - away_component_predictions["hits"])
    row["diff_projected_earned_runs"] = (home_component_predictions["earned_runs"] - away_component_predictions["earned_runs"])
    row["diff_projected_walks"] = (home_component_predictions["walks"] - away_component_predictions["walks"])
    row["diff_projected_hit_batters"] = (home_component_predictions["hit_batters"] - away_component_predictions["hit_batters"])
    
    result = pd.DataFrame([row])

    missing = [col for col in win_feature_cols if col not in result.columns]
    extra = [col for col in result.columns if col not in win_feature_cols]

    if missing:
        raise ValueError(f"Missing win-model prediction features: {missing}")
    if extra:
        print(f"Warning: dropping unexpected win-model features: {extra}")

    result = result.reindex(columns=win_feature_cols)

    if result.isna().all(axis=1).iloc[0]:
        raise ValueError(f"All win features are missing for {away_team} @ {home_team}")

    return result


def predict_pitcher(historical_data, pitcher_name, opponent, game_date, location="Away", opponent_pitcher_name=None,):
    """
    Predict all pitching components.

    If opponent_pitcher_name is supplied, also calculate the
    probability that this pitcher's team wins.
    """

    models, feature_cols = load_component_models()

    features = build_prediction_features(
        pitcher_name=pitcher_name,
        opponent=opponent,
        game_date=game_date,
        historical_data=historical_data,
        feature_cols=feature_cols,
        location=location,
    )

    predictions = {}

    for target, model in models.items():

        prediction = float(model.predict(features)[0])

        predictions[target] = max(0.0, prediction,)

    predictions["win_probability"] = None

    if opponent_pitcher_name is None:
        return predictions

    try:
        team_features, bullpen_features = load_win_feature_data()

        win_model, win_feature_cols = load_win_model()

        pitcher_team = get_pitcher_team(historical_data, pitcher_name, game_date)

        opponent_location = ("Away" if location == "Home" else "Home")

        opponent_features = build_prediction_features(
            pitcher_name=opponent_pitcher_name,
            opponent=pitcher_team,
            game_date=game_date,
            historical_data=historical_data,
            feature_cols=feature_cols,
            location=opponent_location,
        )

        if location == "Home":
            home_features = features.iloc[0]
            away_features = opponent_features.iloc[0]
            home_team = pitcher_team
            away_team = opponent

            home_component_predictions = predictions
            away_component_predictions = {}

            for target, model in models.items():
                prediction = float(model.predict(opponent_features)[0])
                away_component_predictions[target] = max(0.0, prediction)

        else:
            home_features = opponent_features.iloc[0]
            away_features = features.iloc[0]
            home_team = opponent
            away_team = pitcher_team

            away_component_predictions = predictions
            home_component_predictions = {}

            for target, model in models.items():
                prediction = float(model.predict(opponent_features)[0])
                home_component_predictions[target] = max(0.0, prediction)

        win_features = build_win_prediction_features(
            home_pitcher_features=home_features,
            away_pitcher_features=away_features,
            home_team=home_team,
            away_team=away_team,
            game_date=game_date,
            team_features=team_features,
            bullpen_features=bullpen_features,
            win_feature_cols=win_feature_cols,
            home_component_predictions=home_component_predictions,
            away_component_predictions=away_component_predictions,
        )

        home_win_probability = float(win_model.predict_proba(win_features)[0, 1])

        if location == "Home":
            predictions["win_probability"] = (home_win_probability)
        else:
            predictions["win_probability"] = (1.0 - home_win_probability)

    except (ValueError, KeyError, FileNotFoundError, IndexError) as e:
        print(f"Win probability prediction unavailable for {pitcher_name}: {e}")

    return predictions


def predict_matchup(historical_data, home_pitcher_name, away_pitcher_name, home_team, away_team, game_date):
    """
    Generate predictions for both pitchers in a matchup.

    Returns a DataFrame with one row per pitcher.
    """

    game_date = pd.Timestamp(game_date)

    home_predictions = predict_pitcher(
        historical_data=historical_data,
        pitcher_name=home_pitcher_name,
        opponent=away_team,
        game_date=game_date,
        location="Home",
        opponent_pitcher_name=away_pitcher_name,
    )

    away_predictions = predict_pitcher(
        historical_data=historical_data,
        pitcher_name=away_pitcher_name,
        opponent=home_team,
        game_date=game_date,
        location="Away",
        opponent_pitcher_name=home_pitcher_name,
    )

    home_row = {
        "game_date": game_date,
        "pitcher": home_pitcher_name,
        "team": home_team,
        "opponent": away_team,
        "location": "Home",
        **home_predictions,
    }

    away_row = {
        "game_date": game_date,
        "pitcher": away_pitcher_name,
        "team": away_team,
        "opponent": home_team,
        "location": "Away",
        **away_predictions,
    }

    return pd.DataFrame([home_row, away_row])