from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

from src.data_pull import refresh_current_season_statcast
from src.features import build_pitcher_games_table, add_pitch_mix_features, build_team_game_stats, add_team_rolling_features, add_opponent_features, add_rolling_features, add_historical_features, identify_starters, add_park_factors


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA = PROJECT_ROOT / "data" / "processed"
MODELS = PROJECT_ROOT / "models"

MODEL_PATH = MODELS / "xgboost.pkl"
FEATURE_PATH = MODELS / "model_features.json"

FEATURE_TABLE_PATH = PROCESSED_DATA / "pitcher_games_features_base.csv"

SEASON_START = "2026-03-20"


def build_feature_table(statcast_df):
    """Build the complete leakage-safe 2026 modeling table."""

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


def get_model_features(df):
    """
    Return the 49 features used by the model.
    """

    feature_cols = [
        "rest_days",
        "max_times_through_order",
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
        "is_starter",
        "opp_runs_last14",
        "opp_k_rate_last14",
        "opp_bb_rate_last14",
        "opp_hr_rate_last14",
        "opp_iso_last14",
        "park_run_factor",
        "park_hr_factor",
        "last_avg_velocity",
        "rolling3_avg_velocity",
        "rolling5_avg_velocity",
        "season_avg_velocity",
        "last_avg_spin",
        "rolling3_avg_spin",
        "rolling5_avg_spin",
        "season_avg_spin",
        "last_avg_break_x",
        "rolling3_avg_break_x",
        "rolling5_avg_break_x",
        "season_avg_break_x",
        "last_avg_break_z",
        "rolling3_avg_break_z",
        "rolling5_avg_break_z",
        "season_avg_break_z",
        "last_csw",
        "rolling3_csw",
        "rolling5_csw",
        "season_csw",
        "last_whiffs",
        "rolling3_whiffs",
        "rolling5_whiffs",
        "season_whiffs",
        "last_called_strikes",
        "rolling3_called_strikes",
        "rolling5_called_strikes",
        "season_called_strikes",
    ]

    missing = [col for col in feature_cols if col not in df.columns]

    if missing:
        raise ValueError(f"Missing model features: {missing}")

    return feature_cols


def train_model(df):
    """Train the XGBoost strikeout model."""

    df = df.copy()

    df["game_date"] = pd.to_datetime(df["game_date"])

    df = df.sort_values("game_date")

    feature_cols = get_model_features(df)

    df = df[df["strikeouts"].notna()].copy()

    X = df[feature_cols]
    y = df["strikeouts"]

    split_date = df["game_date"].quantile(0.8)

    train_mask = df["game_date"] < split_date
    test_mask = df["game_date"] >= split_date

    X_train = X.loc[train_mask]
    X_test = X.loc[test_mask]

    y_train = y.loc[train_mask]
    y_test = y.loc[test_mask]

    print(f"Training rows: {len(X_train):,}")
    print(f"Test rows:     {len(X_test):,}")
    print(f"Split date:    {split_date.date()}")

    model = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "xgb",
                XGBRegressor(
                    n_estimators=100,
                    max_depth=3,
                    learning_rate=0.05,
                    colsample_bytree=0.8,
                    subsample=1.0,
                    objective="reg:squarederror",
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ]
    )

    print("Training XGBoost...")

    model.fit(X_train, y_train)

    return model, feature_cols


def save_model(model, feature_cols):
    """Save the trained model and feature list."""

    MODELS.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    with open(FEATURE_PATH, "w") as f:
        json.dump(feature_cols, f, indent=4)

    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved features: {FEATURE_PATH}")


def evaluate_model(df, model, feature_cols):
    """Evaluate the newly trained model on the time-based test set."""

    df = df.copy()

    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values("game_date")

    df = df[df["strikeouts"].notna()].copy()

    split_date = df["game_date"].quantile(0.8)

    test = df[df["game_date"] >= split_date].copy()

    X_test = test[feature_cols]
    y_test = test["strikeouts"]

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print("\nModel evaluation")
    print("-" * 30)
    print(f"MAE:  {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R²:   {r2:.3f}")

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }


def main():

    print("=" * 60)
    print("Fantasy Pitcher Predictor — Weekly Retraining")
    print("=" * 60)

    today = pd.Timestamp.today().normalize()

    print(f"Retraining date: {today.date()}")
    print(f"Season:          2026")

    print("\nRefreshing 2026 Statcast data...")

    statcast_df = refresh_current_season_statcast(SEASON_START, today - pd.Timedelta(days=1),)

    if statcast_df.empty:
        raise RuntimeError("No Statcast data available.")

    print(f"Statcast rows: {len(statcast_df):,}")

    print("\nBuilding feature table...")

    feature_table = build_feature_table(statcast_df)

    print(f"Feature table shape: {feature_table.shape}")

    PROCESSED_DATA.mkdir(parents=True, exist_ok=True)

    feature_table.to_csv(FEATURE_TABLE_PATH, index=False)

    print(f"Saved feature table: {FEATURE_TABLE_PATH}")

    print("\nTraining model...")

    model, feature_cols = train_model(feature_table)

    print("\nEvaluating model...")

    metrics = evaluate_model(feature_table, model, feature_cols)

    print("\nSaving model...")

    save_model(model, feature_cols)

    print("\n" + "=" * 60)
    print("Weekly retraining complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()