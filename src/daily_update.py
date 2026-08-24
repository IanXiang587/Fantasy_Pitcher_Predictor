from pathlib import Path
import pandas as pd
import numpy as np

from src.data_pull import refresh_current_season_statcast, get_schedule
from src.features import (
    build_pitcher_games_table,
    add_pitch_mix_features,
    build_team_game_stats,
    add_team_rolling_features,
    add_opponent_features,
    add_rolling_features,
    add_historical_features,
    identify_starters,
    add_park_factors,
    add_bullpen_features,
)
from src.predict import build_prediction_features, build_win_prediction_features, load_component_models, load_win_model
from src.scoring import calculate_points


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DATA = PROJECT_ROOT / "data" / "processed"
MODELS = PROJECT_ROOT / "models"

FEATURE_PATH = MODELS / "model_features.json"

PREDICTIONS_PATH = PROCESSED_DATA / "latest_predictions.csv"
FEATURE_TABLE_PATH = PROCESSED_DATA / "pitcher_games_features_base.csv"
TEAM_FEATURES_PATH = PROCESSED_DATA / "team_game_features.csv"
BULLPEN_FEATURES_PATH = PROCESSED_DATA / "bullpen_game_features.csv"
PITCHING_TARGETS_PATH = PROCESSED_DATA / "pitching_targets.csv"

SEASON_START = "2026-03-20"

PROCESSED_DATA.mkdir(parents=True, exist_ok=True)


def refresh_statcast():
    print("Refreshing current-season Statcast...")

    statcast_df = refresh_current_season_statcast(season_start=SEASON_START, lookback_days=3,)

    if statcast_df.empty:
        raise RuntimeError("Statcast refresh returned no data.")

    print(f"Statcast rows available: {len(statcast_df):,}")

    return statcast_df


def build_feature_table(statcast_df):

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

    pitcher_games = add_opponent_features(pitcher_games, team_games,)

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

    return pitcher_games, team_games


def refresh_pitching_targets(feature_table):

    print("\nRefreshing official pitching targets...")

    game_pks = (feature_table["game_pk"].dropna().astype(int).unique())

    existing = pd.DataFrame(
        columns=[
            "game_pk",
            "pitcher",
            "outs",
            "hits",
            "earned_runs",
            "walks",
            "hit_batters",
            "strikeouts_official",
            "home_runs_allowed",
        ]
    )

    if PITCHING_TARGETS_PATH.exists():

        existing = pd.read_csv(PITCHING_TARGETS_PATH)

        if not existing.empty:
            existing["game_pk"] = pd.to_numeric(existing["game_pk"], errors="coerce",)
            existing["pitcher"] = pd.to_numeric(existing["pitcher"], errors="coerce")

    existing_game_pks = set(existing["game_pk"].dropna().astype(int))

    missing_game_pks = [int(game_pk) for game_pk in game_pks if int(game_pk) not in existing_game_pks]

    print(f"Existing target games: {len(existing_game_pks):,}")

    print(f"New target games required: {len(missing_game_pks):,}")

    if missing_game_pks:
        from src.retraining import get_pitching_targets

        new_targets = get_pitching_targets(missing_game_pks)

        if not new_targets.empty:
            existing = pd.concat([existing, new_targets], ignore_index=True)

    existing = existing.drop_duplicates(subset=["game_pk", "pitcher"], keep="last")

    existing.to_csv(PITCHING_TARGETS_PATH, index=False)

    print(f"Saved pitching targets: {PITCHING_TARGETS_PATH}")

    return existing


def save_feature_tables(pitcher_games, team_games, bullpen_games):

    pitcher_games.to_csv(FEATURE_TABLE_PATH, index=False)
    team_games.to_csv(TEAM_FEATURES_PATH, index=False)
    bullpen_games.to_csv(BULLPEN_FEATURES_PATH, index=False)

    print(f"Saved pitcher features: {FEATURE_TABLE_PATH}")
    print(f"Saved team features: {TEAM_FEATURES_PATH}")
    print(f"Saved bullpen features: {BULLPEN_FEATURES_PATH}")


def generate_predictions(feature_table, probable_pitchers, models, feature_cols, win_model, win_feature_cols, team_features, bullpen_features):
    """Generate all component and win predictions."""

    predictions = []

    for _, game in probable_pitchers.iterrows():
        game_date = pd.Timestamp(game["game_date"])

        away_pitcher = game["away_pitcher"]
        home_pitcher = game["home_pitcher"]

        away_features = None
        home_features = None

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

            except Exception as e:
                print(f"Could not build features for {away_pitcher}: {e}")

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

            except Exception as e:
                print(f"Could not build features for {home_pitcher}: {e}")

        away_component_predictions = {}
        home_component_predictions = {}

        if away_features is not None:
            for target, model in models.items():
                prediction = float(model.predict(away_features)[0])
                away_component_predictions[target] = max(0.0, prediction)

        if home_features is not None:
            for target, model in models.items():
                prediction = float(model.predict(home_features)[0])
                home_component_predictions[target] = max(0.0, prediction)

        home_win_probability = None

        if home_features is not None and away_features is not None:
            try:
                win_features = build_win_prediction_features(
                    home_pitcher_features=home_features.iloc[0],
                    away_pitcher_features=away_features.iloc[0],
                    home_team=game["home_team"],
                    away_team=game["away_team"],
                    game_date=game_date,
                    team_features=team_features,
                    bullpen_features=bullpen_features,
                    win_feature_cols=win_feature_cols,
                    home_component_predictions=home_component_predictions,
                    away_component_predictions=away_component_predictions,
                )

                print("\nWIN PREDICTION DEBUG")
                print("-" * 40)
                print(f"Game: {game['away_team']} @ {game['home_team']}")
                
                missing_win_features = win_features.columns[win_features.isna().any()].tolist()
                print(f"Missing win features: {len(missing_win_features)}")

                if missing_win_features:
                    print(f"  {missing_win_features}")

                if list(win_features.columns) != list(win_feature_cols):
                    raise ValueError("Win prediction feature order does not match trained win model.")

                if win_features.shape != (1, len(win_feature_cols)):
                    raise ValueError(f"Unexpected win feature shape: {win_features.shape}. Expected (1, {len(win_feature_cols)}).")

                home_win_probability = float(win_model.predict_proba(win_features)[0, 1])

                if not np.isfinite(home_win_probability):
                    raise ValueError("Win model returned a non-finite probability.")

                home_win_probability = float(np.clip(home_win_probability, 0.0, 1.0))

                print(f"Home win probability: {home_win_probability:.4f}")

            except Exception as e:
                print(f"Could not calculate win probability for {game['away_team']} @ {game['home_team']}: {type(e).__name__}: {e}")

        if (pd.notna(away_pitcher) and away_features is not None):
            row = {
                "game_date": game_date,
                "pitcher": away_pitcher,
                "team": game["away_team"],
                "opponent": game["home_team"],
                "location": "Away",
                "strikeouts": away_component_predictions.get("strikeouts"),
                "outs": away_component_predictions.get("outs"),
                "hits": away_component_predictions.get("hits"),
                "earned_runs": away_component_predictions.get("earned_runs"),
                "walks": away_component_predictions.get("walks"),
                "hit_batters": away_component_predictions.get("hit_batters"),
                "win_probability": (None if home_win_probability is None else 1.0 - home_win_probability),
            }

            row["projected_points"] = calculate_points(row)

            predictions.append(row)

        if pd.notna(home_pitcher) and home_features is not None:
            row = {
                "game_date": game_date,
                "pitcher": home_pitcher,
                "team": game["home_team"],
                "opponent": game["away_team"],
                "location": "Home",
                "strikeouts": home_component_predictions.get("strikeouts"),
                "outs": home_component_predictions.get("outs"),
                "hits": home_component_predictions.get("hits"),
                "earned_runs": home_component_predictions.get("earned_runs"),
                "walks": home_component_predictions.get("walks"),
                "hit_batters": home_component_predictions.get("hit_batters"),
                "win_probability": home_win_probability,
            }

            row["projected_points"] = calculate_points(row)

            predictions.append(row)

    return pd.DataFrame(predictions)


def main():
    today = (pd.Timestamp.now(tz="America/Toronto").normalize().tz_localize(None))
    tomorrow = (today + pd.Timedelta(days=1))

    print("=" * 60)
    print("Fantasy Pitcher Predictor — Daily Update")
    print("=" * 60)
    print(f"Today: {today.date()}")
    print(f"Tomorrow: {tomorrow.date()}")

    statcast_df = refresh_statcast()

    feature_table, team_games = (build_feature_table(statcast_df))

    print(f"Feature table shape: {feature_table.shape}")

    targets = refresh_pitching_targets(feature_table)

    print("\nBuilding bullpen features...")

    bullpen_games = add_bullpen_features(feature_table, targets)

    save_feature_tables(feature_table, team_games, bullpen_games)

    models, feature_cols = (load_component_models())

    win_model, win_feature_cols = (load_win_model())

    print("\nGetting today's MLB schedule...")

    today_schedule = get_schedule(today)

    print(f"Today's games: {len(today_schedule)}")

    today_predictions = generate_predictions(
        feature_table=feature_table,
        probable_pitchers=today_schedule,
        models=models,
        feature_cols=feature_cols,
        win_model=win_model,
        win_feature_cols=win_feature_cols,
        team_features=team_games,
        bullpen_features=bullpen_games,
    )

    if not today_predictions.empty:
        today_predictions["prediction_for"] = "today"

    print("\nGetting tomorrow's MLB schedule...")

    tomorrow_schedule = get_schedule(tomorrow)

    print(f"Tomorrow's games: {len(tomorrow_schedule)}")

    tomorrow_predictions = (
        generate_predictions(
            feature_table=feature_table,
            probable_pitchers=tomorrow_schedule,
            models=models,
            feature_cols=feature_cols,
            win_model=win_model,
            win_feature_cols=win_feature_cols,
            team_features=team_games,
            bullpen_features=bullpen_games,
        )
    )

    if not tomorrow_predictions.empty:
        tomorrow_predictions["prediction_for"] = "tomorrow"

    predictions = pd.concat([today_predictions, tomorrow_predictions], ignore_index=True)

    if not predictions.empty:
        predictions = predictions.sort_values(["game_date", "location", "pitcher"]).reset_index(drop=True)

    predictions.to_csv(PREDICTIONS_PATH, index=False)

    print(f"\nSaved {len(predictions)} predictions: {PREDICTIONS_PATH}")
    print("=" * 60)
    print("Daily update complete.")
    print("=" * 60)


if __name__ == "__main__":
   main()