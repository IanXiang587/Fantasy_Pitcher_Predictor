# Fantasy Pitcher Strikeout Predictor

An end-to-end machine learning project that predicts how many strikeouts an MLB pitcher is expected to record in an upcoming start.

The project combines MLB Statcast data, pitcher recent-form metrics, opponent offensive performance, rest, and park factors to produce strikeout predictions and a simple fantasy **START/SIT** recommendation.

The system is designed as a complete automated pipeline:

- Current-season MLB data ingestion
- Leakage-safe feature engineering
- XGBoost strikeout prediction
- Time-based model evaluation
- Historical backtesting
- Automated daily predictions
- Weekly model retraining
- GitHub Actions scheduling
- Streamlit deployment

---

## Project Goal

The goal is to answer a practical fantasy baseball question:

> **Given a pitcher and an upcoming matchup, how many strikeouts should we expect, and is the matchup worth starting the pitcher?**

Rather than stopping at model training, this project implements the complete workflow from raw MLB data through automated predictions and a deployed web application.

---

## System Architecture

```text
                    MLB / Statcast Data
                           |
                           v
                  Data Refresh Pipeline
                           |
                           v
                 Feature Engineering
                           |
                           v
              Leakage-Safe Feature Table
                           |
              +------------+------------+
              |                         |
              v                         v
       Daily Prediction             Weekly Retraining
              |                         |
              v                         v
     Today + Tomorrow             XGBoost Model
        Predictions                     |
              |                         |
              +------------+------------+
                           |
                           v
                latest_predictions.csv
                           |
                           v
                     Streamlit App
```

---

## Project Structure

```text
Fantasy_Pitcher_Predictor/
│
├── app.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── xgboost.pkl
│   └── model_features.json
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_baseline_models.ipynb
│   ├── 04_model_iteration.ipynb
│   └── 05_backtesting.ipynb
│
├── src/
│   ├── data_pull.py
│   ├── features.py
│   ├── predict.py
│   ├── daily_update.py
│   ├── weekly_update.py
│   └── retraining.py
│
├── .github/
│   └── workflows/
│       └── update.yml
│
├── requirements.txt
└── README.md
```

---

## Data

The project uses MLB Statcast data collected through `pybaseball`.

The original modeling and experimentation work was performed using historical MLB data. The deployed pipeline now operates on the **2026 MLB season**, beginning March 20, 2026.

The current automated system continuously refreshes current-season Statcast data and rebuilds the modeling feature table.

### Main feature groups

#### Pitcher performance

- Strikeouts
- Pitch count
- Average velocity
- Spin rate
- Horizontal movement
- Vertical movement
- Whiffs
- Called strikes
- CSW%

#### Recent form

Leakage-safe historical features include:

- Previous-start performance
- Rolling 3-game averages
- Rolling 5-game averages
- Season averages
- Velocity trend
- Strikeout variability
- Recent CSW%
- Recent whiffs
- Recent called strikes

#### Workload and usage

- Rest days
- Recent pitch volume
- Maximum times through the order
- Starting pitcher indicator

#### Opponent offense

Recent opponent metrics include:

- Runs
- Strikeout rate
- Walk rate
- Home-run rate
- ISO

#### Park factors

The model includes:

- Park run factor
- Park home-run factor

These allow the model to account for differences in offensive environments between ballparks.

---

## Feature Engineering

The feature engineering pipeline is designed around an important requirement:

> **A prediction must only use information that would have been available before the pitcher makes the start.**

Rolling features are calculated chronologically so future game results cannot be incorporated into earlier predictions.

The final feature table contains **84 columns**, while the deployed XGBoost model uses **49 features**.

The feature table is stored at:

```text
data/processed/pitcher_games_features_base.csv
```

The model feature list is stored at:

```text
models/model_features.json
```

---

## Model Development

Several models were evaluated during development:

- Naive rolling-average baseline
- Ridge regression
- Lasso regression
- XGBoost

Random train/test splitting was intentionally avoided because the model predicts future baseball games.

Instead, model evaluation follows chronological ordering:

```text
Earlier games
     |
     v
Training data
     |
     v
Later games
     |
     v
Validation / test data
```

### Initial baseline

```text
MAE:  1.954
RMSE: 2.449
R²:   0.003
```

The baseline established a reference point using recent historical strikeout performance.

### Initial model comparison

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Tuned Ridge | 1.752 | 2.210 | 0.188 |
| Lasso | 1.750 | 2.205 | 0.191 |
| **XGBoost** | **1.743** | **2.187** | **0.204** |

XGBoost produced the strongest performance during model iteration and was selected as the deployed model.

---

## Current 2026 Model

The deployed model is:

```text
XGBoost Regressor
```

with the trained pipeline stored at:

```text
models/xgboost.pkl
```

The model currently uses 49 input features.

The latest automated 2026 retraining produced approximately:

```text
MAE:  0.961
RMSE: 1.340
R²:   0.619
```

These metrics are generated during the weekly retraining process using a chronological train/test split.

Model performance can change as additional 2026 data becomes available and the model is retrained.

---

## Historical Backtesting

The model was also evaluated against a historical stretch of MLB games.

The backtest produced:

```text
MAE:  1.729
RMSE: 2.157
R²:   0.250
```

Predictions were converted into a simple fantasy decision:

```text
Predicted Ks >= 5.0 → START
Predicted Ks < 5.0  → SIT
```

### Backtest Results

| Decision | Games | Average Actual K | 5+ K Rate |
|---|---:|---:|---:|
| **START** | 750 | **5.835** | **69.5%** |
| SIT | 892 | 3.953 | 38.8% |
| Overall | 1,642 | 4.812 | — |

START recommendations averaged approximately **1.88 more actual strikeouts per game** than SIT recommendations.

The START group also averaged approximately **1.02 more strikeouts than the overall backtest average**.

These results indicate that the model's predictions can provide useful separation between stronger and weaker fantasy strikeout opportunities, although they should not be interpreted as guaranteed outcomes.

---

## Automated Daily Predictions

The project now refreshes predictions automatically rather than requiring the user to manually run the entire pipeline.

The daily pipeline:

1. Refreshes current-season Statcast data.
2. Rebuilds the pitcher-game feature table.
3. Loads the saved XGBoost model.
4. Retrieves MLB schedules and probable pitchers.
5. Generates predictions for **today**.
6. Generates predictions for **tomorrow**.
7. Labels each prediction as `today` or `tomorrow`.
8. Saves the results to:

```text
data/processed/latest_predictions.csv
```

The Streamlit application reads this saved prediction file rather than recomputing the entire prediction pipeline every time the application loads.

---

## Weekly Model Retraining

The model is **not retrained every day**.

A separate weekly process handles model retraining:

```text
Weekly GitHub Actions Job
        |
        v
Refresh 2026 Statcast
        |
        v
Rebuild feature table
        |
        v
Train XGBoost
        |
        v
Evaluate chronological test set
        |
        v
Save model + feature list
        |
        v
Generate updated predictions
```

The weekly orchestration is implemented in:

```text
src/weekly_update.py
```

The actual training process is implemented in:

```text
src/retraining.py
```

This separation keeps daily predictions fast while allowing the model to adapt to the current season on a slower cadence.

---

## GitHub Actions Automation

The pipeline is scheduled through GitHub Actions.

Workflow:

```text
.github/workflows/update.yml
```

### Daily job

The daily workflow runs automatically every day.

It:

- Checks out the repository
- Installs Python dependencies
- Runs `src.daily_update`
- Commits the updated prediction file
- Pushes the new predictions back to GitHub

The workflow can also be triggered manually through GitHub Actions.

### Weekly job

The weekly workflow runs the retraining process on a slower cadence.

It:

- Refreshes current-season data
- Rebuilds the feature table
- Retrains XGBoost
- Evaluates the new model
- Saves the model
- Generates updated predictions

This architecture avoids unnecessarily retraining the model every day.

---

## Streamlit Application

The Streamlit application provides two ways to use the model.

### Automated daily predictions

The application displays:

#### Today's Predictions

Current-day probable pitchers are ranked by projected strikeouts.

#### Tomorrow's Predictions

Tomorrow's probable pitchers are ranked separately.

Each prediction includes:

- Pitcher
- Team
- Opponent
- Home/Away location
- Projected strikeouts
- START/SIT recommendation

### Manual matchup predictor

The application also allows the user to select:

- Pitcher
- Opponent
- Home/Away location
- Game date

The application then constructs the appropriate matchup feature vector and sends it through the saved XGBoost model.

Run locally with:

```bash
streamlit run app.py
```

The deployed Streamlit application uses the same saved prediction and model artifacts stored in the repository.

---

## Prediction Architecture

```text
MLB schedule
     |
     v
Probable pitchers
     |
     v
Historical feature table
     |
     +--> Recent pitcher performance
     +--> Velocity / spin / movement
     +--> Opponent offense
     +--> Rest / workload
     +--> Park factors
     |
     v
49 model features
     |
     v
XGBoost
     |
     v
Projected strikeouts
     |
     v
START / SIT
```

The prediction logic is implemented in:

```text
src/predict.py
```

---

## Technologies

- Python
- pandas
- NumPy
- scikit-learn
- XGBoost
- pybaseball
- Streamlit
- Jupyter Notebook
- Git
- GitHub Actions

---

## Limitations

This remains a portfolio project rather than a production fantasy baseball system.

### Probable pitchers

MLB probable pitcher information can change before game time. Predictions therefore depend on the most recently available schedule information.

### Player and team changes

Mid-season trades, roster changes, injuries, and changes in pitcher roles can affect model accuracy.

### Park factors

Park factors are treated as environmental adjustments rather than dynamically estimated for every individual game.

### Fantasy scoring

The START/SIT recommendation uses a simple 5-strikeout threshold. It does not model a specific league's scoring system, roster construction, replacement level, or category needs.

### Model uncertainty

Strikeout totals remain noisy. Even a strong model cannot account for every factor affecting a single baseball game.

### Current-season performance

The 2026 model will continue to change as additional games become available and weekly retraining incorporates new information.

---

## Future Improvements

Potential future work includes:

- Batter-level matchup features
- Confirmed starting lineup information
- Batter handedness splits
- Pitcher-vs-team history
- Bullpen usage
- Weather and temperature
- Betting strikeout lines
- Prediction confidence intervals
- Fantasy scoring projections
- League-specific START/SIT thresholds
- Improved model calibration
- Additional time-based cross-validation
- More advanced ensemble models

---

## Conclusion

This project demonstrates an end-to-end machine learning workflow applied to a practical fantasy baseball problem.

The system goes beyond model training by implementing:

- MLB Statcast data ingestion
- Leakage-safe feature engineering
- Time-based model evaluation
- Historical backtesting
- XGBoost prediction
- Automated daily predictions
- Weekly model retraining
- GitHub Actions scheduling
- Streamlit deployment

The current system produces strikeout predictions for both **today and tomorrow**, while the underlying model is periodically retrained as the 2026 MLB season progresses.

The project therefore demonstrates not only predictive modeling, but also the engineering required to turn a machine-learning experiment into an automated application.