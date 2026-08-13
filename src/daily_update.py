from pathlib import Path
import json

import joblib
import pandas as pd
import numpy as np

from src.data_pull import refresh_current_season_statcast, get_tomorrow_schedule
from src.features import build_pitcher_games_table, add_pitch_mix_features, build_team_game_stats, add_team_rolling_features, add_opponent_features, add_rolling_features, add_historical_features, identify_starters, add_park_factors
from src.predict import build_prediction_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DATA = PROJECT_ROOT / "data" / "processed"

MODEL_PATH = PROJECT_ROOT / "models" / "xgboost.pkl"

FEATURE_PATH = PROJECT_ROOT / "models" / "model_features.json"

PROCESSED_DATA.mkdir(parents=True, exist_ok=True,)

SEASON_START = "2026-03-20"

FEATURE_TABLE_PATH = PROCESSED_DATA / "pitcher_games_features_base.csv"

PREDICTIONS_PATH = PROCESSED_DATA / "latest_predictions.csv"


def load_model():
    """Load the saved XGBoost model and feature list."""

    model = joblib.load(MODEL_PATH)

    with open(FEATURE_PATH, "r") as f:
        feature_cols = json.load(f)

    return model, feature_cols


def refresh_statcast():
    """
    Download Statcast data for the requested date range.
    """

    print("Refreshing current-season Statcast...")

    statcast_df = refresh_current_season_statcast(season_start=SEASON_START, lookback_days=3)

    if statcast_df.empty:
        raise RuntimeError("Statcast refresh returned no data.")

    print(f"Statcast rows available: {len(statcast_df):,}")

    return statcast_df


def build_feature_table(statcast_df):
    """
    Rebuild the leakage-safe pitcher feature table.
    """

    print("Building pitcher-game table...")

    pitcher_games = build_pitcher_games_table(statcast_df)

    print("Adding pitch-mix features...")

    pitcher_games = add_pitch_mix_features(statcast_df, pitcher_games)

    print("Identifying starting pitchers...")

    starters = identify_starters(statcast_df)

    pitcher_games = pitcher_games.merge(starters, on=["game_pk", "pitcher"], how="left")

    pitcher_games["is_starter"] = (pitcher_games["is_starter"].fillna(False).astype(int))

    print("Building team offensive statistics...")

    team_games = build_team_game_stats(statcast_df)

    print("Adding team rolling features...")

    team_games = add_team_rolling_features(team_games)

    print("Adding opponent features...")

    pitcher_games = add_opponent_features(pitcher_games, team_games)

    print("Adding park factors...")

    pitcher_games = add_park_factors(pitcher_games)

    print("Adding pitcher rolling features...")

    pitcher_games = add_rolling_features(pitcher_games)

    print("Adding historical features...")

    historical_columns = [
        "avg_velocity",
        "avg_spin",
        "avg_break_x",
        "avg_break_z",
        "csw",
        "whiffs",
        "called_strikes",
    ]

    pitcher_games = add_historical_features(pitcher_games, historical_columns)

    return pitcher_games


def save_feature_table(df):
    """Save the refreshed feature table."""

    df.to_csv(FEATURE_TABLE_PATH, index=False)

    print(f"Saved feature table: {FEATURE_TABLE_PATH}")


def generate_predictions(feature_table, probable_pitchers, model, feature_cols):
    """
    Generate predictions for tomorrow's probable starters.

    Games without an announced probable pitcher are skipped.
    """

    predictions = []

    for _, game in probable_pitchers.iterrows():

        game_date = game["game_date"]

        away_pitcher = game["away_pitcher"]

        if pd.notna(away_pitcher):
            try:
                away_features = build_prediction_features(
                    pitcher_name=away_pitcher,
                    opponent=game["home_team"],
                    game_date=game_date,
                    historical_data=feature_table,
                    feature_cols=feature_cols,
                    location="Away",
                )

                away_features = away_features[feature_cols].copy()
                away_features = away_features.apply(pd.to_numeric, errors="coerce")

                prediction = float(model.predict(away_features)[0])

                predictions.append(
                    {
                        "game_date": game_date,
                        "pitcher": away_pitcher,
                        "team": game["away_team"],
                        "opponent": game["home_team"],
                        "location": "Away",
                        "projected_strikeouts": max(0, prediction),
                    }
                )

            except ValueError as e:
                print(f"Could not predict {away_pitcher}: {e}")

        home_pitcher = game["home_pitcher"]

        if pd.notna(home_pitcher):
            try:
                home_features = build_prediction_features(
                    pitcher_name=home_pitcher,
                    opponent=game["away_team"],
                    game_date=game_date,
                    historical_data=feature_table,
                    feature_cols=feature_cols,
                    location="Home",
                )

                home_features = home_features[feature_cols].copy()
                home_features = home_features.apply(pd.to_numeric, errors="coerce")

                prediction = float(model.predict(home_features)[0])

                predictions.append(
                    {
                        "game_date": game_date,
                        "pitcher": home_pitcher,
                        "team": game["home_team"],
                        "opponent": game["away_team"],
                        "location": "Home",
                        "projected_strikeouts": max(0, prediction),
                    }
                )

            except ValueError as e:
                print(f"Could not predict {home_pitcher}: {e}")

    return pd.DataFrame(predictions)


def main():

    today = pd.Timestamp.today().normalize()
    tomorrow = today + pd.Timedelta(days=1)

    print("=" * 60)
    print("Fantasy Pitcher Predictor — Daily Update")
    print("=" * 60)

    print(f"Today:    {today.date()}")
    print(f"Tomorrow: {tomorrow.date()}")

    statcast_df = refresh_statcast()

    feature_table = build_feature_table(statcast_df)

    save_feature_table(feature_table)

    model, feature_cols = load_model()

    print("Getting tomorrow's MLB schedule...")

    probable_pitchers = get_tomorrow_schedule(today)

    print(f"Tomorrow's games: {len(probable_pitchers)}")

    predictions = generate_predictions(
        feature_table=feature_table,
        probable_pitchers=probable_pitchers,
        model=model,
        feature_cols=feature_cols,
    )

    predictions.to_csv(PREDICTIONS_PATH, index=False)

    print(f"Saved {len(predictions)} predictions: {PREDICTIONS_PATH}")

    print("=" * 60)
    print("Daily update complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
