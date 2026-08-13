from pathlib import Path
import json
import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "xgboost.pkl"
FEATURE_PATH = PROJECT_ROOT / "models" / "model_features.json"
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "pitcher_games_features_base.csv"

def load_model():
    """Load the saved XGBoost model and feature list."""

    model = joblib.load(MODEL_PATH)

    with open(FEATURE_PATH, "r") as f:
        feature_cols = json.load(f)

    return model, feature_cols

def load_historical_data(data_path=None): 
    """ Load the historical pitcher feature table. If no path is provided, use the existing processed dataset. """ 
    if data_path is None: 
        data_path = DEFAULT_DATA_PATH 

    data = pd.read_csv(data_path) 

    data["game_date"] = pd.to_datetime( data["game_date"] ) 

    return data

def get_pitcher_history(historical_data, pitcher_name, game_date):
    """Return all pitcher's games before the prediction date."""

    game_date = pd.Timestamp(game_date)

    pitcher_name = normalize_pitcher_name(pitcher_name)

    history = historical_data[(historical_data["player_name"] == pitcher_name) & (historical_data["game_date"] < game_date)].copy()

    return history.sort_values("game_date")

def get_pitcher_team(historical_data, pitcher_name, game_date):
    """Get the pitcher's most recent team."""

    pitcher_history = get_pitcher_history(historical_data, pitcher_name, game_date)

    if pitcher_history.empty:
        raise ValueError(f"No historical data found for {pitcher_name}")

    latest = pitcher_history.iloc[-1]

    home_team = latest["home_team"]
    away_team = latest["away_team"]
    opponent = latest["opponent"]

    if opponent == home_team:
        return away_team

    if opponent == away_team:
        return home_team

    raise ValueError(
        f"Could not determine team for {pitcher_name}. "
        f"Home={home_team}, Away={away_team}, "
        f"Opponent={opponent}"
    )

def get_opponent_features(historical_data, opponent, game_date):
    """Get the opponent's most recent pregame offensive metrics."""

    game_date = pd.Timestamp(game_date)

    opponent_history = historical_data[(historical_data["opponent"] == opponent) & (historical_data["game_date"] < game_date)].copy()

    if opponent_history.empty:
        raise ValueError(f"No historical opponent data found for {opponent}")

    latest = opponent_history.sort_values("game_date").iloc[-1]

    return {
        "opp_runs_last14": latest["opp_runs_last14"],
        "opp_k_rate_last14": latest["opp_k_rate_last14"],
        "opp_bb_rate_last14": latest["opp_bb_rate_last14"],
        "opp_hr_rate_last14": latest["opp_hr_rate_last14"],
        "opp_iso_last14": latest["opp_iso_last14"],
    }

def get_park_factors(historical_data, home_team, game_date):
    """Get the most recent park factors for the home team's park."""

    game_date = pd.Timestamp(game_date)

    park_history = historical_data[historical_data["home_team"] == home_team].copy()

    if park_history.empty:
        raise ValueError(f"No historical park data found for {home_team}")

    latest = park_history.sort_values("game_date").iloc[-1]

    return {
        "park_run_factor": latest["park_run_factor"],
        "park_hr_factor": latest["park_hr_factor"],
    }

def build_prediction_features(pitcher_name, opponent, game_date, historical_data, feature_cols, location="Away"):
    """Build the 49 pregame features for a matchup."""

    game_date = pd.Timestamp(game_date)

    historical_data = historical_data.copy()

    historical_data['game_date'] = pd.to_datetime(historical_data['game_date'])

    pitcher_history = get_pitcher_history(historical_data, pitcher_name, game_date)

    if pitcher_history.empty:
        raise ValueError(f"No historical data found for {pitcher_name}")

    latest = pitcher_history.iloc[-1].copy()

    features = latest[feature_cols].copy()

    features["is_starter"] = 1

    features["rest_days"] = (game_date - latest["game_date"]).days

    features["is_starter"] = 1

    opponent_features = get_opponent_features(historical_data, opponent, game_date)

    for column, value in opponent_features.items():
        features[column] = value

    pitcher_team = get_pitcher_team(historical_data, pitcher_name, game_date)

    if location == "Home":
        home_team = pitcher_team
    else:
        home_team = opponent

    park_features = get_park_factors(historical_data, home_team, game_date)

    for column, value in park_features.items():
        features[column] = value

    return pd.DataFrame([features], columns=feature_cols)

def predict_pitcher(historical_data, pitcher_name, opponent, game_date, location="Away"):
    """Predict strikeouts for a future pitcher matchup."""

    model, feature_cols = load_model()

    features = build_prediction_features(
        pitcher_name=pitcher_name, 
        opponent=opponent, 
        game_date=game_date, 
        historical_data=historical_data, 
        feature_cols=feature_cols,
        location=location,
    )

    prediction = model.predict(features)[0]

    return float(prediction)

def normalize_pitcher_name(name):
    """
    Convert MLB Stats API names into the naming format
    used by the historical Statcast data.

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