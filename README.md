# ⚾ Fantasy Pitcher Predictor

A machine-learning application that predicts MLB starting pitcher performance for upcoming games.

The project uses MLB Statcast data, official pitching results, team offensive performance, pitcher recent form, park factors, bullpen performance, and other matchup features to generate predictions for upcoming MLB games.

The application is designed to provide useful pitcher projections without requiring a specific fantasy league scoring system.

---

## Features

The application currently provides:

- Projected strikeouts
- Projected outs
- Projected hits allowed
- Projected earned runs
- Projected walks
- Projected hit batters
- Team win probability
- Projected fantasy points
- Start/Sit recommendations
- Predictions for today's games
- Predictions for tomorrow's games
- Manual pitcher matchup predictions
- Model evaluation metrics
- Automated daily data updates
- Automated weekly model retraining

---

## How It Works

The project follows an automated machine-learning pipeline:

```text
MLB Statcast Data
       |
       v
Data Cleaning & Aggregation
       |
       v
Pitcher Game-Level Features
       |
       v
Rolling / Historical Features
       |
       v
Opponent & Team Features
       |
       v
Park Factors
       |
       v
Bullpen Features
       |
       v
Training Targets
       |
       v
Machine Learning Models
       |
       v
Upcoming Game Predictions
       |
       v
Streamlit Application
```

---

## Data

The primary data source is MLB Statcast data accessed through `pybaseball`.

The project also uses MLB schedule and pitching information to identify upcoming games and probable pitchers.

The current pipeline is configured for the **2026 MLB season**.

### Main Data Components

- Pitch-level Statcast data
- Pitcher game logs
- Team offensive statistics
- Opponent offensive performance
- Pitcher rolling statistics
- Pitch mix characteristics
- Pitch velocity and spin
- Called strikes and whiffs
- Rest days
- Park factors
- Bullpen performance
- Official pitching results

---

## Feature Engineering

Features are designed to use information that would have been available before the predicted game.

### Pitcher Performance

Examples include:

- Recent strikeouts
- Recent pitch count
- Average velocity
- Average spin rate
- Pitch movement
- Called-strike rate
- Whiffs
- CSW
- Recent performance trends
- Historical averages
- Standard deviations
- Rest days

### Opponent Features

The model incorporates recent opponent offensive performance, including statistics such as:

- Runs
- Strikeout rate
- Walk rate
- Home run rate
- Isolated power
- wOBA and other offensive-strength metrics

### Team and Win-Probability Features

The win-probability model additionally uses:

- Team recent scoring
- Opponent recent scoring
- Bullpen ERA
- Bullpen FIP
- Pitcher matchup differences
- Recent pitcher performance differences
- Team and opponent matchup information

### Park Factors

Ballpark characteristics are incorporated to account for differences in offensive environments between stadiums.

---

## Machine Learning Models

The project uses separate machine-learning models for different pitching outcomes.

### Component Prediction Models

| Prediction | Model |
|---|---|
| Strikeouts | XGBoost |
| Outs | XGBoost |
| Hits allowed | XGBoost |
| Earned runs | XGBoost |
| Walks | XGBoost |
| Hit batters | XGBoost |

A separate classifier estimates the probability that the pitcher's team wins the game.

---

## Model Evaluation

The model is evaluated using a chronological train/test split rather than randomly mixing games from different points in the season.

This better simulates the real-world prediction problem:

```text
Past games
    |
    v
Training data
    |
    v
Future games
    |
    v
Test data
```

The current strikeout model has achieved approximately:

- **MAE:** 0.96 strikeouts
- **RMSE:** 1.34 strikeouts
- **R²:** 0.62

These values represent the model's current evaluation performance and may change after future retraining.

---

## Leakage Prevention

A major focus of the project is preventing future information from entering predictions.

Rolling and historical features are calculated using previous games rather than the current game's result.

For example:

```text
Game 1 ---> Game 2 ---> Game 3 ---> Game 4
                         |
                         v
                  Prediction uses
                  Games 1-3 only
```

This prevents the model from using information that would not have been available when making an actual prediction.

---

## Automated Pipeline

The project is designed to operate automatically.

### Daily Update

The daily pipeline:

1. Refreshes current-season Statcast data
2. Builds pitcher-game features
3. Updates team features
4. Updates opponent features
5. Updates bullpen features
6. Retrieves official pitching targets
7. Loads the trained models
8. Retrieves today's MLB schedule
9. Retrieves tomorrow's MLB schedule
10. Generates component predictions
11. Generates win probabilities
12. Calculates projected fantasy points
13. Saves the latest predictions

The resulting predictions are stored in:

```text
data/processed/latest_predictions.csv
```

---

## Weekly Retraining

The model is retrained automatically on a weekly schedule.

Weekly retraining:

1. Refreshes the current-season Statcast dataset
2. Rebuilds the feature table
3. Retrieves official pitching results
4. Rebuilds team and bullpen features
5. Trains the component prediction models
6. Trains the win-probability classifier
7. Evaluates the models on future data
8. Saves the updated models and feature definitions

The system intentionally **does not retrain every day**.

Daily updates use the latest trained models, while model retraining occurs weekly.

---

## Project Structure

```text
Fantasy_Pitcher_Predictor/
|
├── app.py
├── README.md
├── requirements.txt
|
├── .github/
│   └── workflows/
│       └── update.yml
|
├── data/
│   ├── raw/
│   │   └── statcast_2026.csv
│   │
│   └── processed/
│       ├── latest_predictions.csv
│       ├── pitcher_games_features_base.csv
│       ├── pitching_targets.csv
│       ├── team_game_features.csv
│       └── bullpen_game_features.csv
|
├── models/
│   ├── xgboost.pkl
│   ├── outs_model.pkl
│   ├── hits_model.pkl
│   ├── er_model.pkl
│   ├── bb_model.pkl
│   ├── hbp_model.pkl
│   ├── win_model.pkl
│   └── model_features.json
|
├── notebooks/
│   ├── exploration&feature_eng.ipynb
│   └── model_dev.ipynb
|
└── src/
    ├── data_pull.py
    ├── features.py
    ├── predict.py
    ├── retraining.py
    ├── daily_update.py
    ├── weekly_update.py
    ├── train.py
    └── evaluate.py
```

---

## Running the Application Locally

### 1. Clone the Repository

```bash
git clone https://github.com/IanXiang587/Fantasy_Pitcher_Predictor.git
cd Fantasy_Pitcher_Predictor
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate the Environment

Windows:

```bash
.venv\Scripts\activate
```

macOS / Linux:

```bash
source .venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in a browser.

---

## Updating Predictions Manually

The daily prediction pipeline can be run with:

```bash
python -m src.daily_update
```

This refreshes the current-season data and generates predictions for today's and tomorrow's games.

---

## Running Weekly Retraining Manually

The complete weekly pipeline can be run with:

```bash
python -m src.weekly_update
```

This retrains the models and updates the supporting feature tables.

---

## Streamlit Application

The Streamlit application provides two primary ways to use the model.

### Daily Predictions

The application displays automatically generated predictions for:

- Today's games
- Tomorrow's games

Each pitcher receives projections for the supported pitching components, win probability, and projected fantasy points.

### Manual Matchup

Users can select:

- Pitcher
- Opponent
- Home/Away location
- Game date

The application then generates a custom strikeout prediction for that matchup.

---

## Fantasy Recommendations

The application provides a simple recommendation based on projected strikeouts.

Current threshold:

```text
Projected strikeouts >= 5.0  ->  START
Projected strikeouts < 5.0   ->  SIT
```

This is intended as a simple general-purpose recommendation rather than a replacement for league-specific roster decisions.

---

## Project Goals

The long-term goal of the project is to build a reliable, automated fantasy baseball pitching prediction system that:

- Uses current MLB data
- Avoids data leakage
- Updates automatically
- Retrains periodically
- Provides interpretable component-level projections
- Supports different fantasy scoring systems
- Provides useful matchup information
- Can be deployed as a standalone web application

---

## Limitations

Predictions are estimates and should not be treated as guarantees.

Pitcher performance can be affected by factors that are difficult to model, including:

- Unexpected injuries
- Pitch-count restrictions
- Changes in starting pitcher status
- Weather
- Lineup changes
- Late scratches
- Bullpen usage
- Managerial decisions
- Small sample sizes
- Unpredictable game outcomes

The model also depends on the availability and accuracy of external MLB data.

---

## Future Improvements

Potential future improvements include:

- Improved probable-pitcher identification
- Weather and wind features
- Confirmed starting-lineup information
- Batter-level matchup features
- Pitch-type-specific opponent performance
- Improved calibration of win probabilities
- More sophisticated fantasy scoring customization
- Model comparison and ensemble methods
- Improved uncertainty estimates
- Expanded historical backtesting

---

## Technologies

The project uses:

- Python
- pandas
- NumPy
- scikit-learn
- XGBoost
- pybaseball
- MLB data
- Streamlit
- GitHub Actions

---

## Author

**Ian Xiang**

Fantasy Pitcher Predictor is a personal machine-learning project focused on applying data science and predictive modeling to fantasy baseball.

---

## Disclaimer

This project is for educational and analytical purposes.

Predictions are generated by machine-learning models and should not be considered guaranteed outcomes or financial advice.