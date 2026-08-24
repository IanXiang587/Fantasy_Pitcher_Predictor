from pathlib import Path
import json
import joblib
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, log_loss, brier_score_loss, accuracy_score, roc_auc_score
import numpy as np

from src.data_pull import refresh_current_season_statcast
from src.features import build_pitcher_games_table, add_pitch_mix_features, build_team_game_stats, add_team_rolling_features, add_opponent_features, add_rolling_features, add_historical_features, identify_starters, add_park_factors, add_bullpen_features, add_win_probability_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA = PROJECT_ROOT / "data" / "processed"
MODELS = PROJECT_ROOT / "models"

MODEL_PATH = MODELS / "xgboost.pkl"
FEATURE_PATH = MODELS / "model_features.json"

FEATURE_TABLE_PATH = PROCESSED_DATA / "pitcher_games_features_base.csv"
TEAM_FEATURES_PATH = PROCESSED_DATA / "team_game_features.csv"
BULLPEN_FEATURES_PATH = PROCESSED_DATA / "bullpen_game_features.csv"

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
    """Return the 49 features used by the model."""

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

def train_component_model(df, target_column, model_filename, feature_cols):
    """
    Train one XGBoost regression model for a pitching component.

    Uses the same leakage-safe features and chronological
    80/20 time-based split as the strikeout model.
    """

    df = df.copy()

    df["game_date"] = pd.to_datetime(df["game_date"])

    df[target_column] = pd.to_numeric(df[target_column], errors="coerce")

    df = df.sort_values("game_date")

    df = df[df[target_column].notna()].copy()

    split_date = df["game_date"].quantile(0.8)

    train_mask = df["game_date"] < split_date
    test_mask = df["game_date"] >= split_date

    X_train = df.loc[train_mask, feature_cols]
    X_test = df.loc[test_mask, feature_cols]

    y_train = df.loc[train_mask, target_column]
    y_test = df.loc[test_mask, target_column]

    print()
    print("=" * 50)
    print(f"Training component: {target_column}")
    print("=" * 50)

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

    pred = model.predict(X_test)

    pred = np.maximum(pred, 0)

    mae = mean_absolute_error(y_test, pred)

    rmse = np.sqrt(mean_squared_error(y_test, pred))

    r2 = r2_score(y_test, pred)

    print(f"MAE: {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R²: {r2:.3f}")

    model_path = MODELS / model_filename

    joblib.dump(model, model_path)

    print(f"Saved model: {model_path}")

    return {
        "model": model,
        "metrics": {
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
        },
    }


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
    predictions = np.maximum(predictions, 0)

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

def get_one_game_pitching_targets(game_pk):
    """Download pitching stats for one MLB game."""

    url = f"https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore"

    response = requests.get(url, timeout=30)
    response.raise_for_status()

    boxscore = response.json()

    records = []

    for team_side in ["home", "away"]:
        players = boxscore["teams"][team_side]["players"]

        for player_id, player_data in players.items():

            pitching = player_data.get("stats", {}).get("pitching", {})

            if not pitching:
                continue

            records.append({
                "game_pk": int(game_pk),
                "pitcher": int(player_id.replace("ID", "")),
                "outs": pitching.get("outs"),
                "hits": pitching.get("hits"),
                "earned_runs": pitching.get("earnedRuns"),
                "walks": pitching.get("baseOnBalls"),
                "hit_batters": pitching.get("hitBatsmen"),
                "strikeouts_official": pitching.get("strikeOuts"),
                "home_runs_allowed": pitching.get("homeRuns"),
            })

    return records


def get_pitching_targets(game_pks, max_workers=8):
    """
    Download official pitcher-game outcomes from MLB Stats API.

    Uses concurrent requests to speed up the download.
    """

    all_records = []
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:

        futures = {
            executor.submit(
                get_one_game_pitching_targets,
                game_pk
            ): game_pk
            for game_pk in game_pks
        }

        for future in as_completed(futures):

            game_pk = futures[future]

            try:
                records = future.result()
                all_records.extend(records)

            except Exception as e:
                print(f"Failed game {game_pk}: {e}")

            completed += 1

            if completed % 100 == 0:
                print(
                    f"Processed {completed:,} / "
                    f"{len(game_pks):,} games"
                )

    return pd.DataFrame(all_records)

def merge_pitching_targets(feature_table, target_table):
    """
    Merge official game-level pitching outcomes onto the
    leakage-safe pregame feature table.
    """

    feature_table = feature_table.copy()
    target_table = target_table.copy()

    feature_table["game_pk"] = pd.to_numeric(feature_table["game_pk"], errors="coerce")

    feature_table["pitcher"] = pd.to_numeric(feature_table["pitcher"], errors="coerce")

    target_table["game_pk"] = pd.to_numeric(target_table["game_pk"], errors="coerce")

    target_table["pitcher"] = pd.to_numeric(target_table["pitcher"], errors="coerce")

    target_columns = ["game_pk", "pitcher", "outs", "hits", "earned_runs", "walks", "hit_batters", "strikeouts_official"]

    target_table = target_table[target_columns]

    merged = feature_table.merge(target_table, on=["game_pk", "pitcher"], how="inner")

    print(f"Feature rows: {len(feature_table):,}")

    print(f"Target rows: {len(target_table):,}")

    print(f"Matched rows: {len(merged):,}")

    return merged


def build_game_results(statcast_df):
    """
    Build one game-level win/loss result from Statcast.

    The result is the actual outcome and is used only as the
    classification target, never as a prediction feature.
    """

    df = statcast_df.copy()

    df["game_date"] = pd.to_datetime(df["game_date"])

    game_results = (df.groupby(["game_date", "game_pk", "home_team", "away_team"]).agg(home_score=("home_score", "max"), away_score=("away_score", "max"),).reset_index())

    return game_results


def add_win_target(pitcher_games, game_results):
    """Add game outcome information to each pitcher-game row."""

    pitcher_games = pitcher_games.copy()
    game_results = game_results.copy()

    pitcher_games = pitcher_games.merge(
        game_results[["game_pk", "home_score", "away_score"]], on="game_pk", how="left")

    pitcher_games["home_win"] = (pitcher_games["home_score"] > pitcher_games["away_score"]).astype(int)

    return pitcher_games


def build_game_level_win_data(training_data):
    """
    Convert pitcher-level training data into one row per game.

    Each row represents the game from the HOME team's perspective.

    Target:
        home_win = 1 if home team won, else 0

    Only information available before the game is used as features.
    """

    print("\nBuilding game-level win data...")

    df = training_data.copy()

    df["game_date"] = pd.to_datetime(df["game_date"])

    starters = df.loc[df["is_starter"] == 1].copy()

    if starters.empty:
        raise ValueError("No starting pitchers found in training data.")

    starters = (starters.sort_values(["game_pk", "game_date", "pitcher"]).drop_duplicates(["game_pk", "pitcher_team"], keep="first"))

    home = starters.loc[starters["pitcher_team"] == starters["home_team"]].copy()
    away = starters.loc[starters["pitcher_team"] == starters["away_team"]].copy()

    home = home.add_prefix("home_pitcher_")
    away = away.add_prefix("away_pitcher_")

    home = home.rename(columns={"home_pitcher_game_pk": "game_pk"})
    away = away.rename(columns={"away_pitcher_game_pk": "game_pk"})

    game_data = home.merge(away, on="game_pk", how="inner", validate="one_to_one")

    if game_data.empty:
        raise ValueError( "No valid game-level win data could be constructed." )

    result = pd.DataFrame({
        "game_pk": game_data["game_pk"],
        "game_date": game_data["home_pitcher_game_date"],
        "home_team": game_data["home_pitcher_home_team"],
        "away_team": game_data["away_pitcher_away_team"],
        "home_win": game_data["home_pitcher_home_win"].astype(int),
    })

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

    for col in difference_features:
        home_col = f"home_pitcher_{col}"
        away_col = f"away_pitcher_{col}"

        if home_col not in game_data.columns:
            result[f"diff_{col}"] = np.nan
            continue

        if away_col not in game_data.columns:
            result[f"diff_{col}"] = np.nan
            continue

        home_values = pd.to_numeric(game_data[home_col], errors="coerce")
        away_values = pd.to_numeric(game_data[away_col], errors="coerce")

        result[f"diff_{col}"] = home_values - away_values

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

    for col in matchup_features:
        home_col = f"home_pitcher_{col}"
        away_col = f"away_pitcher_{col}"

        if home_col in game_data.columns:
            result[f"home_{col}"] = pd.to_numeric(game_data[home_col], errors="coerce")
        else:
            result[f"home_{col}"] = np.nan

        if away_col in game_data.columns:
            result[f"away_{col}"] = pd.to_numeric(game_data[away_col], errors="coerce")
        else:
            result[f"away_{col}"] = np.nan

    result = (result .sort_values("game_date") .reset_index(drop=True))

    duplicate_games = result["game_pk"].duplicated().sum()

    print(f"Game-level rows: {len(result):,}")

    print("\nGame-level win data validation")
    print("-" * 40)

    print(f"Unique games: {result['game_pk'].nunique():,}")
    print(f"Home wins: {(result['home_win'] == 1).sum():,}")
    print(f"Home losses: {(result['home_win'] == 0).sum():,}")
    print(f"Home win rate: {result['home_win'].mean():.4f}")
    print(f"Duplicate games: {duplicate_games:,}")

    if duplicate_games > 0:
        raise ValueError(f"Game-level data contains {duplicate_games} duplicate games.")

    return result


def add_projected_win_features(game_data, component_models, component_feature_cols, training_data):
    """
    Add projected pitcher-component differences to game-level win data.

    All projections are generated from the same component models/features
    used during prediction.
    """

    print("\nAdding projected component features...")

    required_targets = ["strikeouts", "outs", "hits", "earned_runs", "walks", "hit_batters"]

    for target in required_targets:
        if target not in component_models:
            raise ValueError(f"Missing component model: {target}")

    starters = training_data[training_data["is_starter"] == 1].copy()
    starters = (starters.sort_values(["game_pk", "game_date", "pitcher"]).drop_duplicates(["game_pk", "pitcher_team"], keep="first"))

    home = starters[starters["pitcher_team"] == starters["home_team"]].copy()
    away = starters[starters["pitcher_team"] == starters["away_team"]].copy()

    home = home.drop_duplicates(subset=["game_pk"], keep="first")
    away = away.drop_duplicates(subset=["game_pk"], keep="first")

    home = home.set_index("game_pk")
    away = away.set_index("game_pk")

    valid_games = (game_data["game_pk"].isin(home.index) & game_data["game_pk"].isin(away.index))

    if not valid_games.all():
        game_data = game_data.loc[valid_games].copy()

    component_predictions = {}

    for target in required_targets:

        model = component_models[target]

        if isinstance(model, str):
            model_path = MODELS / model

            if not model_path.exists():
                raise FileNotFoundError(f"Component model not found: {model_path}")

            model = joblib.load(model_path)

        feature_cols = component_feature_cols[target]

        missing_home = [c for c in feature_cols if c not in home.columns]
        missing_away = [c for c in feature_cols if c not in away.columns]

        if missing_home:
            raise ValueError(f"Missing home features for {target}: {missing_home}")

        if missing_away:
            raise ValueError(f"Missing away features for {target}: {missing_away}")

        X_home = home[feature_cols].copy()
        X_away = away[feature_cols].copy()

        X_home = X_home.apply(pd.to_numeric, errors="coerce")
        X_away = X_away.apply(pd.to_numeric, errors="coerce")

        X_home = X_home.replace([np.inf, -np.inf], np.nan)
        X_away = X_away.replace([np.inf, -np.inf], np.nan)

        home_pred = model.predict(X_home)
        away_pred = model.predict(X_away)

        component_predictions[f"diff_projected_{target}"] = (pd.Series(home_pred, index=home.index) - pd.Series(away_pred, index=away.index))

    game_data = game_data.copy()

    for target in required_targets:
        feature_name = f"diff_projected_{target}"

        prediction_series = component_predictions[feature_name]

        game_data[feature_name] = (game_data["game_pk"].map(prediction_series))

    print("Projected component features added.")

    return game_data


def train_game_level_win_model(training_data, component_models, component_feature_cols):
    """Train a game-level XGBoost win classifier."""

    print("\n" + "=" * 50)
    print("Training game-level win-probability model")
    print("=" * 50)

    game_data = build_game_level_win_data(training_data)
    game_data = add_projected_win_features(game_data, component_models, component_feature_cols, training_data)

    print("\nWIN MODEL FEATURE CHECK")
    print("-" * 40)

    print(game_data[["game_date", "game_pk", "home_team", "away_team", "home_win"]].head())

    print("\nColumns:")
    print(game_data.columns.tolist())

    print("\nTarget distribution:")
    print(game_data["home_win"].value_counts(normalize=True))

    print("\nFeature missingness:")
    print(game_data[[c for c in game_data.columns if c.startswith("diff_") or c.startswith("home_") or c.startswith("away_")]].isna().mean().sort_values(ascending=False).head(20))

    if game_data.empty:
        raise ValueError("No game-level win data available.")

    print(f"Game-level rows: {len(game_data):,}")

    game_data = (game_data.sort_values("game_date").reset_index(drop=True))

    feature_cols = [
        "diff_team_runs_last14",
        "diff_opp_team_runs_last14",
        "diff_bullpen_era_last14",
        "diff_opp_bullpen_era_last14",
        "diff_bullpen_fip_last14",
        "diff_opp_bullpen_fip_last14",
        "diff_run_support_diff",
        "diff_bullpen_era_diff",
        "diff_bullpen_fip_diff",
        "diff_k_last3",
        "diff_velo_last3",
        "diff_spin_last3",
        "diff_csw_last3",
        "diff_pitches_last3",
        "diff_velo_last6",
        "diff_velo_trend",
        "diff_k_std_last5",
        "diff_pitches_last5",
        "diff_whiff_last3",
        "diff_csw_last5",
        "home_opp_runs_last14",
        "away_opp_runs_last14",
        "home_opp_k_rate_last14",
        "away_opp_k_rate_last14",
        "home_opp_bb_rate_last14",
        "away_opp_bb_rate_last14",
        "home_opp_hr_rate_last14",
        "away_opp_hr_rate_last14",
        "home_opp_iso_last14",
        "away_opp_iso_last14",
        "home_park_run_factor",
        "away_park_run_factor",
        "home_park_hr_factor",
        "away_park_hr_factor",
        "home_rest_days",
        "away_rest_days",
        "diff_projected_strikeouts",
        "diff_projected_outs",
        "diff_projected_hits",
        "diff_projected_earned_runs",
        "diff_projected_walks",
        "diff_projected_hit_batters",
    ]

    missing = [col for col in feature_cols if col not in game_data.columns]

    if missing:
        raise ValueError(f"Missing win-model features: {missing}")

    X = game_data[feature_cols].copy()

    #***
    print("\nWIN MODEL LEAKAGE CHECK")
    print("-" * 40)

    for col in feature_cols:
        if col in ["home_win", "home_score", "away_score", "runs", "home_runs", "away_runs", "score_diff", "win", "loss"]:
            raise ValueError(f"TARGET LEAKAGE: forbidden column '{col}' is being used by the win model.")

    print("Win-model features:")
    for col in feature_cols:
        print(f"  {col}")

    print("\nForbidden result columns present in game_data:")

    for col in ["home_win", "home_score", "away_score", "score_diff", "win", "loss"]:
        if col in game_data.columns:
            print(f"  FOUND: {col}")
    #***

    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    y = game_data["home_win"].astype(int)

    split_date = pd.Timestamp("2026-07-20")

    train_mask = game_data["game_date"] < split_date
    test_mask = game_data["game_date"] >= split_date

    X_train = X.loc[train_mask]
    X_test = X.loc[test_mask]

    y_train = y.loc[train_mask]
    y_test = y.loc[test_mask]

    #***
    print("\nSHUFFLED TARGET LEAKAGE TEST")
    print("-" * 40)

    y_shuffled = y_train.sample(frac=1, random_state=42).to_numpy()

    shuffle_model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "xgb",
                XGBClassifier(
                    n_estimators=100,
                    max_depth=3,
                    learning_rate=0.03,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    shuffle_model.fit(X_train, y_shuffled)

    shuffle_probs = shuffle_model.predict_proba(X_test)[:, 1]

    shuffle_auc = roc_auc_score(y_test, shuffle_probs)

    print(f"Shuffled-target ROC-AUC: {shuffle_auc:.4f}")
    #***

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
                XGBClassifier(
                    n_estimators=300,
                    max_depth=3,
                    learning_rate=0.03,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    #***
    print("\nFEATURE/TARGET CORRELATIONS")
    print("-" * 40)

    correlations = (
        game_data[feature_cols + ["home_win"]]
        .corr(numeric_only=True)["home_win"]
        .drop("home_win")
        .sort_values(key=abs, ascending=False)
    )

    print(correlations)
    #***

    print("Training XGBoost classifier...")

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]

    predictions = (probabilities >= 0.5).astype(int)

    logloss = log_loss(y_test, probabilities)
    brier = brier_score_loss(y_test, probabilities)
    accuracy = accuracy_score(y_test, predictions)

    try:
        roc_auc = roc_auc_score(y_test, probabilities)
    except ValueError:
        roc_auc = np.nan

    print("\nGame-level win model evaluation")
    print("-" * 30)

    print(f"Log loss: {logloss:.4f}")
    print(f"Brier: {brier:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Mean pred: {probabilities.mean():.4f}")
    print(f"Actual win: {y_test.mean():.4f}")

    baseline_prob = np.full(len(y_test), 0.5)
    baseline_logloss = log_loss(y_test, baseline_prob)
    baseline_brier = brier_score_loss(y_test, baseline_prob)

    print("\n50/50 baseline")
    print("-" * 30)

    print(f"Log loss: {baseline_logloss:.4f}")
    print(f"Brier: {baseline_brier:.4f}")

    print("\nImprovement")
    print("-" * 30)

    print(f"Log loss improvement: {baseline_logloss - logloss:.4f}")
    print(f"Brier improvement: {baseline_brier - brier:.4f}")
    print(f"Accuracy improvement: {accuracy - 0.5:.4f}")

    model_path = MODELS / "win_model.pkl"

    joblib.dump(model, model_path)

    print(f"\nSaved win model: {model_path}")

    return model, feature_cols


def evaluate_win_model_baseline(training_data, model, feature_cols, component_models, component_feature_cols):
    """
    Evaluate the win-probability model against a 50/50 baseline.

    Uses the exact same game-level transformation and chronological
    split as train_game_level_win_model().
    """

    print("\n" + "=" * 50)
    print("WIN MODEL VS 50/50 BASELINE")
    print("=" * 50)

    game_data = build_game_level_win_data(training_data)
    game_data = add_projected_win_features(game_data, component_models, component_feature_cols, training_data)

    print("\nPROJECTED FEATURE CHECK")
    print("-" * 40)

    projection_cols = [
        "diff_projected_strikeouts",
        "diff_projected_outs",
        "diff_projected_hits",
        "diff_projected_earned_runs",
        "diff_projected_walks",
        "diff_projected_hit_batters",
    ]

    for col in projection_cols:
        print(f"{col}: {col in game_data.columns}")

    if game_data.empty:
        raise ValueError("No game-level win data available for evaluation.")

    game_data = game_data.sort_values("game_date").reset_index(drop=True)

    split_date = pd.Timestamp("2026-07-20")

    test_mask = game_data["game_date"] >= split_date

    test = game_data.loc[test_mask].copy()

    if test.empty:
        raise ValueError("No test games available after the split date.")

    X_test = test[feature_cols].copy()
    X_test = X_test.apply(pd.to_numeric, errors="coerce")
    X_test = X_test.replace([np.inf, -np.inf], np.nan)

    predictions = model.predict_proba(X_test)[:, 1]

    y_test = test["home_win"].astype(int)

    predicted_classes = (predictions >= 0.5).astype(int)

    model_log_loss = log_loss(y_test, predictions)
    model_brier = brier_score_loss(y_test, predictions)
    model_accuracy = accuracy_score(y_test, predicted_classes)

    try:
        model_auc = roc_auc_score(y_test, predictions)
    except ValueError:
        model_auc = np.nan

    baseline_predictions = np.full(len(y_test), 0.5, dtype=float)
    baseline_log_loss = log_loss(y_test, baseline_predictions)
    baseline_brier = brier_score_loss(y_test, baseline_predictions,)
    baseline_accuracy = 0.5

    print(f"Split date: {split_date.date()}")
    print(f"Test games: {len(test):,}")
    print()

    print("XGBoost")
    print("-" * 30)
    print(f"Log loss: {model_log_loss:.4f}")
    print(f"Brier score: {model_brier:.4f}")
    print(f"Accuracy: {model_accuracy:.4f}")
    print(f"ROC-AUC: {model_auc:.4f}")
    print(f"Mean pred: {predictions.mean():.4f}")
    print(f"Actual win: {y_test.mean():.4f}")

    print()
    print("50/50 baseline")
    print("-" * 30)
    print(f"Log loss: {baseline_log_loss:.4f}")
    print(f"Brier score: {baseline_brier:.4f}")
    print(f"Accuracy: {baseline_accuracy:.4f}")

    print()
    print("Improvement")
    print("-" * 30)
    print(f"Log loss improvement: {baseline_log_loss - model_log_loss:.4f}")
    print(f"Brier improvement: {baseline_brier - model_brier:.4f}")
    print(f"Accuracy improvement: {model_accuracy - baseline_accuracy:.4f}")

    return {
        "split_date": str(split_date.date()),
        "test_rows": int(len(test)),
        "model_log_loss": float(model_log_loss),
        "model_brier": float(model_brier),
        "model_accuracy": float(model_accuracy),
        "model_roc_auc": float(model_auc),
        "baseline_log_loss": float(baseline_log_loss),
        "baseline_brier": float(baseline_brier),
        "baseline_accuracy": float(baseline_accuracy),
    }


def main():

    print("=" * 60)
    print("Fantasy Pitcher Predictor — Weekly Retraining")
    print("=" * 60)

    today = pd.Timestamp.today().normalize()

    print(f"Retraining date: {today.date()}")
    print(f"Season: 2026")

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

    print("\nGetting model features...")

    feature_cols = get_model_features(feature_table)

    print("\nDownloading official pitching targets...")

    game_pks = (feature_table["game_pk"].dropna().astype(int).unique())

    print(f"Games requiring targets: {len(game_pks):,}")

    targets = get_pitching_targets(game_pks)

    targets = targets.drop_duplicates(subset=["game_pk", "pitcher"], keep="last")

    targets.to_csv(PROCESSED_DATA / "pitching_targets.csv", index=False)

    print(f"Saved pitching targets: {PROCESSED_DATA / 'pitching_targets.csv'}")

    print(f"Target rows downloaded: {len(targets):,}")

    print("\nMerging features and targets...")

    training_data = merge_pitching_targets(feature_table, targets)

    print("\nMerging official pitching targets...")

    print(f"Training data shape: {training_data.shape}")


    print("\nBuilding team-level win features...")

    team_games = build_team_game_stats(statcast_df)

    team_games = add_team_rolling_features(team_games)

    bullpen_games = add_bullpen_features(feature_table, targets)

    team_games.to_csv(TEAM_FEATURES_PATH, index=False)
    bullpen_games.to_csv(BULLPEN_FEATURES_PATH, index=False)

    print(f"Saved team features: {TEAM_FEATURES_PATH}")
    print(f"Saved bullpen features: {BULLPEN_FEATURES_PATH}")

    training_data = add_win_probability_features(training_data, team_games, bullpen_games)

    game_results = build_game_results(statcast_df)

    training_data = add_win_target(training_data, game_results)

    print(f"Training data shape: {training_data.shape}")

    print("\nTraining strikeout model...")

    model, strikeout_feature_cols = train_model(training_data)

    print("\nEvaluating strikeout model...")

    metrics = evaluate_model(training_data, model, feature_cols)

    print("\nSaving strikeout model...")

    save_model(model, feature_cols)

    component_models = {
        "strikeouts": "xgboost.pkl",
        "outs": "outs_model.pkl",
        "hits": "hits_model.pkl",
        "earned_runs": "er_model.pkl",
        "walks": "bb_model.pkl",
        "hit_batters": "hbp_model.pkl",
    }

    component_feature_cols = {"strikeouts": strikeout_feature_cols}
    component_results = {}

    for target_column, model_filename in component_models.items():
        if target_column == "strikeouts":
            continue
        result = train_component_model(training_data, target_column, model_filename, feature_cols)

        component_results[target_column] = result["metrics"]
        component_feature_cols[target_column] = feature_cols

    print("\nTraining win-probability model...")

    win_model, win_feature_cols = train_game_level_win_model(training_data, component_models, component_feature_cols)

    win_metrics = evaluate_win_model_baseline(training_data, win_model, win_feature_cols, component_models, component_feature_cols)

    win_feature_path = MODELS / "win_model_features.json"

    with open(win_feature_path, "w") as f: 
        json.dump(win_feature_cols, f, indent=4)

    print(f"Saved win features: {win_feature_path}")

    metrics_path = PROCESSED_DATA / "component_model_metrics.json"

    component_results["win_probability"] = win_metrics

    with open(metrics_path, "w") as f:
        json.dump(component_results, f, indent=4)

    print(f"\nSaved component metrics: {metrics_path}")

    print("\n" + "=" * 60)
    print("Weekly retraining complete.")
    print("=" * 60)


if __name__ == "__main__":
   main()