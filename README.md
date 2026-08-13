# Fantasy Pitcher Strikeout Predictor

An end-to-end machine learning project that predicts how many strikeouts an MLB pitcher is expected to record in an upcoming start.

The project combines MLB Statcast data, pitcher recent-form metrics, opponent offensive performance, rest, and park factors to produce a strikeout prediction and a simple fantasy **START/SIT** recommendation.

## Project Goal

The goal is to answer a practical fantasy baseball question:

> **Given a pitcher and an upcoming matchup, how many strikeouts should we expect, and is the matchup worth starting the pitcher?**

The project was designed as a complete machine learning pipeline rather than simply a model-training exercise. It covers data collection, feature engineering, model development, time-based evaluation, historical backtesting, and deployment through a Streamlit application.

---

## Project Structure

```text
Fantasy/
│
├── app.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── xgb_strikeout_model.pkl
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
│   └── predict.py
│
└── README.md
```

---

## Data

The project uses MLB Statcast data collected with `pybaseball`.

The primary modeling dataset covers the **2024 MLB regular season**, from March 20 through September 30, 2024.

The final modeling table contains one row per pitcher game and includes pitcher performance, recent-form, opponent, and park-related features.

### Main feature groups

#### Pitcher performance

* Strikeouts
* Pitch count
* Average velocity
* Spin rate
* Horizontal and vertical movement
* Whiffs
* Called strikes
* CSW%

#### Recent form

Leakage-safe historical features include:

* Last-start performance
* Rolling 3-game averages
* Rolling 5-game averages
* Season averages
* Velocity trend
* Strikeout variability
* Recent CSW%
* Recent whiffs
* Recent called strikes

#### Workload and usage

* Rest days
* Pitches in recent starts
* Maximum times through the order
* Starter indicator

#### Opponent offense

Opponent rolling metrics include:

* Runs
* Strikeout rate
* Walk rate
* Home-run rate
* ISO

These metrics are based on the opponent's recent performance prior to the pitcher start.

#### Park factors

The model includes:

* Park run factor
* Park home-run factor

These allow the model to account for differences in offensive environments between ballparks.

Historical pitch-mix features were considered but were not included in the final model iteration.

---

# Machine Learning Pipeline

The project follows a chronological, leakage-conscious modeling workflow.

```text
Raw MLB Data
     ↓
Data Cleaning
     ↓
Game-Level Aggregation
     ↓
Feature Engineering
     ↓
Time-Based Train/Test Split
     ↓
Baseline Model
     ↓
Ridge / Lasso / XGBoost
     ↓
Time-Based Model Evaluation
     ↓
Season Backtest
     ↓
Streamlit Application
```

Random train/test splits were intentionally avoided because the model is intended to predict future baseball games using information available before those games occur.

---

# Stage 1 — Data Pipeline

Raw Statcast and game-level information was collected and stored locally.

The pipeline produced a pitcher-game dataset covering the 2024 season.

The data was organized into raw and processed directories so that the modeling workflow could be reproduced without repeatedly downloading the raw data.

---

# Stage 2 — Feature Engineering

The second stage transformed historical game data into a modeling dataset.

A major requirement was avoiding future information leakage.

Rolling features were calculated using only information available before each start.

The final dataset contained **84 columns**, including the target and supporting matchup information.

The modeling feature set ultimately contained **49 features**.

---

# Stage 3 — Baseline and Ridge Regression

A naive rolling-average baseline was established before evaluating machine learning models.

### Baseline

```text
MAE:  1.954
RMSE: 2.449
R²:   0.003
```

A Ridge regression model was then developed as the first machine-learning benchmark.

The Ridge model substantially improved upon the naive baseline during the initial model evaluation.

The baseline established an important reference point: a useful machine-learning model needed to demonstrate improvement over simply using recent historical strikeout performance.

---

# Stage 4 — Model Iteration

Three model approaches were evaluated:

* Tuned Ridge regression
* Lasso regression
* XGBoost

Model selection used time-aware evaluation rather than random data splitting.

## Model Comparison

| Model       |       MAE |      RMSE |        R² |
| ----------- | --------: | --------: | --------: |
| Tuned Ridge |     1.752 |     2.210 |     0.188 |
| Lasso       |     1.750 |     2.205 |     0.191 |
| **XGBoost** | **1.743** | **2.187** | **0.204** |

XGBoost produced the strongest overall performance and was selected as the final model.

The trained model is stored in:

```text
models/xgb_strikeout_model.pkl
```

The corresponding model feature list is stored in:

```text
models/model_features.json
```

---

# Stage 5 — Historical Backtesting

The final model was evaluated against a real stretch of the 2024 MLB season rather than relying solely on a conventional test-set metric.

The backtest produced:

```text
MAE:  1.729
RMSE: 2.157
R²:   0.250
```

More importantly, the predictions were converted into practical fantasy decisions.

A pitcher was classified as a **START** when the predicted strikeout total reached the project's 5-strikeout decision threshold.

## Backtest Results

| Decision  | Games | Average Actual K | 5+ K Rate |
| --------- | ----: | ---------------: | --------: |
| **START** |   750 |        **5.835** | **69.5%** |
| SIT       |   892 |            3.953 |     38.8% |
| Overall   | 1,642 |            4.812 |         — |

The pitchers recommended as STARTs averaged **5.835 actual strikeouts**, compared with **3.953** among pitchers classified as SIT.

The START group therefore averaged approximately **1.88 more strikeouts per game** than the SIT group.

Compared with the overall backtest average of 4.812 strikeouts, START recommendations provided an advantage of approximately **1.022 strikeouts per pitcher**.

The results suggest that the model was able to identify substantially better strikeout opportunities than simply treating every pitcher equally.

The backtest output is stored in:

```text
data/processed/stage5_backtest.parquet
```

and the summary results are stored in:

```text
data/processed/stage5_results.csv
```

---

# Stage 6 — Streamlit Application

The final model was deployed through a Streamlit application.

The application allows the user to select:

* Pitcher
* Opponent
* Home/Away location
* Game date

The application then constructs the relevant pregame feature row and sends it to the trained XGBoost model.

The output includes:

* Projected strikeouts
* START/SIT recommendation
* Matchup information
* Model backtest performance

The application can be launched with:

```bash
streamlit run app.py
```

---

# Prediction Architecture

The deployed prediction workflow is:

```text
User selects matchup
        ↓
Retrieve pitcher's historical data
        ↓
Construct recent pitcher features
        ↓
Calculate rest
        ↓
Retrieve opponent recent offensive metrics
        ↓
Determine home team
        ↓
Retrieve park factors
        ↓
Construct 49 model features
        ↓
XGBoost model
        ↓
Predicted strikeouts
        ↓
START / SIT recommendation
```

The prediction logic is implemented in:

```text
src/predict.py
```

---

# Limitations

This project is a portfolio prototype rather than a production fantasy baseball system.

### Historical data

The model was trained using 2024 MLB data. Player performance and team strength can change significantly between seasons.

### Future feature availability

The deployed prediction function uses the latest available historical pitcher features and updates matchup-specific variables such as opponent performance, rest, and park factors.

A production system would automatically refresh these features from current MLB data before every game.

### Team changes

Pitcher team assignment is inferred from the pitcher's most recent historical game. Mid-season trades or roster changes could therefore require additional handling.

### Park factors

Park factors are treated as historical environmental adjustments rather than dynamically updated values.

### START/SIT threshold

The 5-strikeout threshold is a simple decision rule rather than a complete fantasy scoring model. A more advanced system could optimize the decision threshold based on league scoring settings, roster construction, betting lines, and replacement-level alternatives.

### Model performance

An R² of 0.25 on the historical backtest means substantial variation in pitcher strikeouts remains unexplained. Strikeouts are inherently noisy and depend on factors that are difficult to capture before a game.

---

# Future Improvements

Potential future iterations include:

* Current-season data updates
* Automated daily Statcast ingestion
* Batter-level matchup features
* Batter handedness splits
* Expected lineup information
* Starting lineup confirmation
* Bullpen usage
* Pitch-mix matchup features
* Pitcher-vs-team historical performance
* Opponent wOBA and wRC+
* Weather and temperature
* Betting strikeout lines
* Confidence intervals
* Fantasy scoring projections
* Automated daily recommendations
* Cloud deployment

---

# Technologies

* Python
* pandas
* NumPy
* scikit-learn
* XGBoost
* pybaseball
* Streamlit
* Jupyter Notebook
* Git

---

# Conclusion

This project demonstrates an end-to-end machine learning workflow for a practical fantasy baseball problem.

The final XGBoost model achieved a backtest MAE of **1.729 strikeouts** and an R² of **0.250**.

More importantly, converting predictions into fantasy decisions produced a meaningful separation between recommended START and SIT pitchers:

> START recommendations averaged **5.835 actual strikeouts**, compared with **3.953** for SIT recommendations.

The project therefore moves beyond simply predicting a statistic and demonstrates how a predictive model could be incorporated into an actual fantasy baseball decision-making workflow.
