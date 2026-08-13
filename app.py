from pathlib import Path
import pandas as pd
import streamlit as st

from src.predict import predict_pitcher

PROJECT_ROOT = Path(__file__).resolve().parent

PREDICTIONS_PATH = PROJECT_ROOT / "data" / "processed" / "latest_predictions.csv"

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "pitcher_games_features_base.csv"

historical_data = pd.read_csv(DATA_PATH)

historical_data["game_date"] = pd.to_datetime(historical_data["game_date"])

def load_latest_predictions():
    """Load the most recent daily predictions."""

    if not PREDICTIONS_PATH.exists():
        return pd.DataFrame()

    predictions = pd.read_csv(PREDICTIONS_PATH)

    if predictions.empty:
        return predictions

    predictions["game_date"] = pd.to_datetime(predictions["game_date"])

    return predictions

st.set_page_config(
    page_title="Fantasy Pitcher Predictor",
    page_icon="⚾",
    layout="centered",
)


st.title("⚾ Fantasy Pitcher Predictor")

st.write("Predict a starting pitcher's expected strikeouts for an upcoming matchup.")

st.caption("Powered by an XGBoost model trained on 2026 MLB data.")

predictions = load_latest_predictions()

if not predictions.empty:
    prediction_date = predictions["game_date"].iloc[0]

    st.subheader(f"Predictions for {prediction_date.strftime('%B %d, %Y')}")

if predictions.empty:
    st.info("No daily predictions are currently available.")

else:
    predictions = predictions.sort_values("projected_strikeouts", ascending=False)

    display_predictions = predictions[["pitcher", "team", "opponent", "location", "projected_strikeouts"]].copy()

    display_predictions["projected_strikeouts"] = (display_predictions["projected_strikeouts"].round(1))

    display_predictions["Recommendation"] = (display_predictions["projected_strikeouts"].apply(lambda x: "START" if x >= 5.0 else "SIT"))

    st.dataframe(display_predictions, use_container_width=True, hide_index=True)


with st.expander("About the model"):

    st.write(
        "The model uses pitcher performance, recent form, opponent "
        "offense, rest, and park factors to predict strikeouts."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Test MAE", "0.96 Ks")

    with col2:
        st.metric("Test RMSE", "1.34 Ks")

    with col3:
        st.metric("Test R²", "0.62")


st.subheader("Matchup")

pitchers = sorted(historical_data["player_name"].dropna().unique())

teams = sorted(historical_data["opponent"].dropna().unique())


pitcher = st.selectbox("Pitcher", pitchers)

opponent = st.selectbox("Opponent", teams)

location = st.radio("Location", ["Home", "Away"], horizontal=True)

game_date = st.date_input("Game Date")


if st.button("Predict Strikeouts", type="primary", use_container_width=True,):
    try:
        prediction = predict_pitcher(
        historical_data=historical_data,
        pitcher_name=pitcher,
        opponent=opponent,
        game_date=game_date,
        location=location,
        )

        prediction = max(0, prediction)

        st.divider()

        st.subheader("Prediction")

        st.metric("Projected Strikeouts", f"{prediction:.1f}",)


        if prediction >= 5.0:
            st.success(f"🟢 START — projected {prediction:.1f} Ks")

        else:
            st.warning(f"🟡 SIT — projected {prediction:.1f} Ks")


        st.subheader("Matchup")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.write("**Pitcher**")
            st.write(pitcher)

        with col2:
            st.write("**Opponent**")
            st.write(opponent)

        with col3:
            st.write("**Location**")
            st.write(location)

        st.caption("Recommendation threshold: 5.0 projected strikeouts.")

    except ValueError as e:
        st.error(str(e))


st.divider()

st.caption("Fantasy Pitcher Predictor • 2026 MLB Statcast-based model")