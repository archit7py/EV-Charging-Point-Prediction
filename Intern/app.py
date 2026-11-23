from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import joblib


st.set_page_config(page_title="EV Forecast", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "forecasting_ev_model.pkl"
DATA_PATH = BASE_DIR / "preprocessed_ev_data.csv"
IMAGE_PATH = BASE_DIR / "ev-car-factory.jpg"

FEATURE_COLUMNS = [
    "months_since_start",
    "county_encoded",
    "ev_total_lag1",
    "ev_total_lag2",
    "ev_total_lag3",
    "ev_total_roll_mean_3",
    "ev_total_pct_change_1",
    "ev_total_pct_change_3",
    "ev_growth_slope",
]


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    required_columns = {
        "Date",
        "County",
        "county_encoded",
        "months_since_start",
        "Electric Vehicle (EV) Total",
    }
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.dropna(subset=["Date", "County", "Electric Vehicle (EV) Total"]).copy()
    df["County"] = df["County"].astype(str)
    return df.reset_index(drop=True)


def generate_forecast(model, county_df, forecast_horizon=36):
    county_df = county_df.sort_values("Date").copy()
    recent_ev = list(county_df["Electric Vehicle (EV) Total"].dropna().astype(float).tail(6))
    if len(recent_ev) < 3:
        raise ValueError(f"Not enough historical data for {county_df['County'].iloc[0]}.")

    cumulative_ev = list(np.cumsum(recent_ev))
    months_since_start = int(county_df["months_since_start"].max())
    latest_date = county_df["Date"].max()
    county_code = int(county_df["county_encoded"].iloc[0])

    future_rows = []
    for i in range(1, forecast_horizon + 1):
        forecast_date = latest_date + pd.DateOffset(months=i)
        months_since_start += 1

        lag1, lag2, lag3 = recent_ev[-1], recent_ev[-2], recent_ev[-3]
        roll_mean = float(np.mean([lag1, lag2, lag3]))
        pct_change_1 = (lag1 - lag2) / lag2 if lag2 != 0 else 0.0
        pct_change_3 = (lag1 - lag3) / lag3 if lag3 != 0 else 0.0

        recent_cum = cumulative_ev[-6:]
        slope = float(np.polyfit(np.arange(len(recent_cum)), recent_cum, 1)[0]) if len(recent_cum) >= 2 else 0.0

        feature_row = {
            "months_since_start": months_since_start,
            "county_encoded": county_code,
            "ev_total_lag1": lag1,
            "ev_total_lag2": lag2,
            "ev_total_lag3": lag3,
            "ev_total_roll_mean_3": roll_mean,
            "ev_total_pct_change_1": pct_change_1,
            "ev_total_pct_change_3": pct_change_3,
            "ev_growth_slope": slope,
        }

        pred = float(model.predict(pd.DataFrame([feature_row], columns=FEATURE_COLUMNS))[0])
        pred = max(pred, 0.0)
        future_rows.append({"Date": forecast_date, "Predicted EV Total": round(pred, 2)})

        recent_ev.append(pred)
        if len(recent_ev) > 6:
            recent_ev.pop(0)

        cumulative_ev.append(float(cumulative_ev[-1]) + pred)
        if len(cumulative_ev) > 6:
            cumulative_ev.pop(0)

    return pd.DataFrame(future_rows)


st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(to right, #050b14, #16263d, #050b14);
        }
        div[data-testid="stAppViewContainer"] {
            color: #eaf2ff;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style='text-align: center; font-family: Algerian; font-size: 46px; font-weight: bold; color: #FFFFFF; margin-top: 20px;'>
        🔮 EV Adoption Forecaster for Washington Counties
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style='text-align: center; font-family: Algerian; font-size: 22px; font-weight: bold; padding-top: 10px; margin-bottom: 25px; color: #FFFFFF;'>
        Welcome to the Electric Vehicle (EV) Adoption Forecast tool.
    </div>
    """,
    unsafe_allow_html=True,
)

if IMAGE_PATH.exists():
    st.image(str(IMAGE_PATH), use_container_width=True)
else:
    st.info("Image asset not found. The app will still run without it.")

st.markdown(
    """
    <div style='text-align: left; font-size: 28px; padding-top: 10px; color: #FFFFFF;'>
        Select a county and see the forecasted EV adoption trend for the next 3 years.
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    model = load_model()
    df = load_data()
except (FileNotFoundError, ValueError) as exc:
    st.error(f"Unable to start the EV forecasting app: {exc}")
    st.stop()

county_list = sorted(df["County"].dropna().unique().tolist())
county = st.selectbox("Select a County", county_list)

county_df = df[df["County"] == county].sort_values("Date")
if county_df.empty:
    st.warning(f"County '{county}' was not found in the dataset.")
    st.stop()

forecast_horizon = 36
forecast_df = generate_forecast(model, county_df, forecast_horizon)

historical_cum = county_df[["Date", "Electric Vehicle (EV) Total"]].copy()
historical_cum["Source"] = "Historical"
historical_cum["Cumulative EV"] = historical_cum["Electric Vehicle (EV) Total"].cumsum()

forecast_df = forecast_df.copy()
forecast_df["Source"] = "Forecast"
forecast_df["Cumulative EV"] = forecast_df["Predicted EV Total"].cumsum() + historical_cum["Cumulative EV"].iloc[-1]

combined = pd.concat(
    [
        historical_cum[["Date", "Cumulative EV", "Source"]],
        forecast_df[["Date", "Cumulative EV", "Source"]],
    ],
    ignore_index=True,
)

st.subheader(f"📊 Cumulative EV Forecast for {county} County")
fig, ax = plt.subplots(figsize=(12, 6))
for label, data in combined.groupby("Source"):
    ax.plot(data["Date"], data["Cumulative EV"], label=label, marker="o")
ax.set_title(f"Cumulative EV Trend - {county} (3 Years Forecast)", fontsize=14, color="white")
ax.set_xlabel("Date", color="white")
ax.set_ylabel("Cumulative EV Count", color="white")
ax.grid(True, alpha=0.3)
ax.set_facecolor("#1c1c1c")
fig.patch.set_facecolor("#1c1c1c")
ax.tick_params(colors="white")
ax.legend()
st.pyplot(fig)

historical_total = historical_cum["Cumulative EV"].iloc[-1]
forecasted_total = forecast_df["Cumulative EV"].iloc[-1]

if historical_total > 0:
    forecast_growth_pct = ((forecasted_total - historical_total) / historical_total) * 100
    trend = "increase 📈" if forecast_growth_pct > 0 else "decrease 📉"
    st.success(
        f"EV adoption in **{county}** is expected to **{trend} by {abs(forecast_growth_pct):.2f}%** over the next 3 years."
    )
else:
    st.warning("Historical EV total is zero, so the percentage change cannot be computed.")

st.markdown("---")
st.header("Compare EV Adoption Trends for up to 3 Counties")
selected_counties = st.multiselect("Select up to 3 counties to compare", county_list, max_selections=3)

if selected_counties:
    comparison_data = []
    for cty in selected_counties:
        cty_df = df[df["County"] == cty].sort_values("Date")
        if cty_df.empty:
            continue

        cty_forecast = generate_forecast(model, cty_df, forecast_horizon)
        hist_cum = cty_df[["Date", "Electric Vehicle (EV) Total"]].copy()
        hist_cum["Cumulative EV"] = hist_cum["Electric Vehicle (EV) Total"].cumsum()
        cty_forecast["Cumulative EV"] = cty_forecast["Predicted EV Total"].cumsum() + hist_cum["Cumulative EV"].iloc[-1]

        combined_cty = pd.concat(
            [
                hist_cum[["Date", "Cumulative EV"]],
                cty_forecast[["Date", "Cumulative EV"]],
            ],
            ignore_index=True,
        )
        combined_cty["County"] = cty
        comparison_data.append(combined_cty)

    if comparison_data:
        comp_df = pd.concat(comparison_data, ignore_index=True)

        st.subheader("📈 Comparison of Cumulative EV Adoption Trends")
        fig, ax = plt.subplots(figsize=(14, 7))
        for cty, group in comp_df.groupby("County"):
            ax.plot(group["Date"], group["Cumulative EV"], marker="o", label=cty)
        ax.set_title("EV Adoption Trends: Historical + 3-Year Forecast", fontsize=16, color="white")
        ax.set_xlabel("Date", color="white")
        ax.set_ylabel("Cumulative EV Count", color="white")
        ax.grid(True, alpha=0.3)
        ax.set_facecolor("#1c1c1c")
        fig.patch.set_facecolor("#1c1c1c")
        ax.tick_params(colors="white")
        ax.legend(title="County")
        st.pyplot(fig)

        growth_summaries = []
        for cty in selected_counties:
            cty_rows = comp_df[comp_df["County"] == cty].reset_index(drop=True)
            historical_total = cty_rows["Cumulative EV"].iloc[len(cty_rows) - forecast_horizon - 1]
            forecasted_total = cty_rows["Cumulative EV"].iloc[-1]

            if historical_total > 0:
                growth_pct = ((forecasted_total - historical_total) / historical_total) * 100
                growth_summaries.append(f"{cty}: {growth_pct:.2f}%")
            else:
                growth_summaries.append(f"{cty}: N/A")

        st.success(f"Forecasted EV adoption growth over next 3 years — {' | '.join(growth_summaries)}")

st.success("Forecast complete")
st.markdown("Prepared for the **AICTE Internship Cycle 2 by S4F**")
