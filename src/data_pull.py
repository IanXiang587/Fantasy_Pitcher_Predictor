from pathlib import Path
import pandas as pd
import requests
from pybaseball import statcast, schedule_and_record

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = PROJECT_ROOT / "data" / "raw"
RAW_DATA.mkdir(parents=True, exist_ok=True)

def download_statcast(start_date, end_date):
    """
    Download Statcast pitch-level data.
    """
    df = statcast(start_date, end_date)

    file_path = RAW_DATA / f"statcast_{start_date}_{end_date}.csv"

    print(f'Saving to {file_path}')

    df.to_csv(file_path, index=False)

    return df

def refresh_current_season_statcast(season_start, end_date=None, lookback_days=3):
    """
    Incrementally refresh current-season Statcast data.

    A small lookback window is used so recently completed
    games can be replaced if Statcast data is corrected.

    Returns the complete current-season Statcast DataFrame.
    """

    if end_date is None:
        end_date = (pd.Timestamp.today().normalize() - pd.Timedelta(days=1))

    end_date = pd.Timestamp(end_date).normalize()
    season_start = pd.Timestamp(season_start).normalize()

    output_path = RAW_DATA / f"statcast_{season_start.year}.csv"

    if not output_path.exists():

        print("No current-season Statcast file found.")

        print(f"Downloading {season_start.date()} → {end_date.date()}")

        df = statcast(
            season_start.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d"),
        )

    else:

        existing = pd.read_csv(output_path, low_memory=False)

        existing["game_date"] = pd.to_datetime(existing["game_date"])

        latest_date = existing["game_date"].max()

        start_date = max(season_start, latest_date - pd.Timedelta(days=lookback_days))

        print(f"Existing data through {latest_date.date()}")

        print(f"Refreshing {start_date.date()} → {end_date.date()}")

        new_data = statcast(
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d"),
        )

        if new_data.empty:
            print("No new Statcast data returned.")

            return existing

        new_data["game_date"] = pd.to_datetime(new_data["game_date"])

        existing = existing[existing["game_date"] < start_date]

        df = pd.concat([existing, new_data], ignore_index=True,)

    if "game_pk" in df.columns:

        dedup_columns = ["game_pk", "at_bat_number", "pitch_number"]

        available = [col for col in dedup_columns if col in df.columns]

        if available:
            df = df.drop_duplicates(subset=available)

    df = df.sort_values(["game_date","game_pk","at_bat_number","pitch_number",], na_position="last").reset_index(drop=True)

    df.to_csv(output_path, index=False)

    print(f"Saved {len(df):,} Statcast rows to {output_path}")

    return df

PROCESSED_DATA = PROJECT_ROOT / "data" / "processed"

def save_processed_data(df, filename="final_pitcher_modeling_table.csv"):
    PROCESSED_DATA.mkdir(parents=True, exist_ok=True)
    print(f'Saving as {filename}')
    df.to_csv(PROCESSED_DATA / filename, index=False)

def save_model_results(df, model):
    filename = f'{model}_results.csv'
    PROCESSED_DATA.mkdir(parents=True, exist_ok=True)
    print(f'Saving as {filename}')
    df.to_csv(PROCESSED_DATA / filename, index=False)

def get_tomorrow_schedule(date=None):
    """
    Get tomorrow's MLB schedule and probable pitchers.

    Returns one row per scheduled game with:
        game_date
        away_team
        home_team
        away_pitcher
        home_pitcher

    Uses the MLB Stats API directly because pybaseball's
    schedule_and_record() does not provide probable starters.
    """

    if date is None:
        date = pd.Timestamp.today().normalize()

    date = pd.Timestamp(date).normalize()
    tomorrow = date + pd.Timedelta(days=1)

    url = "https://statsapi.mlb.com/api/v1/schedule"

    params = {
        "sportId": 1,
        "date": tomorrow.strftime("%Y-%m-%d"),
        "hydrate": "probablePitcher,team"
    }

    response = requests.get(url, params=params, timeout=30)

    response.raise_for_status()

    data = response.json()

    games = []

    for date_data in data.get("dates", []):

        for game in date_data.get("games", []):

            away = game["teams"]["away"]
            home = game["teams"]["home"]

            away_team = away["team"]["abbreviation"]
            home_team = home["team"]["abbreviation"]

            away_pitcher = (away.get("probablePitcher", {}).get("fullName"))

            home_pitcher = (home.get("probablePitcher", {}).get("fullName"))

            games.append(
                {
                    "game_date": tomorrow,
                    "away_team": away_team,
                    "home_team": home_team,
                    "away_pitcher": away_pitcher,
                    "home_pitcher": home_pitcher,
                }
            )

    if not games:
        return pd.DataFrame(columns=["game_date", "away_team", "home_team", "away_pitcher", "home_pitcher"])

    return pd.DataFrame(games).sort_values(["game_date", "away_team", "home_team"]).reset_index(drop=True)
