from pathlib import Path
import pandas as pd
from pybaseball import statcast

RAW_DATA = Path('data/raw')
RAW_DATA.mkdir(parents=True, exist_ok=True)

def download_statcast(start_date, end_date):
    """
    Download Statcast pitch-level data.
    """
    df = statcast(start_date, end_date)

    file = RAW_DATA / "statcast_pitching_2024.csv"

    df.to_csv(file, index=False)

    return df