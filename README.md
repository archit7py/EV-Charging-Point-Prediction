# ⚡ EV Adoption Forecasting Dashboard

A Streamlit app that forecasts **Electric Vehicle (EV) adoption for counties in Washington State** over the next 36 months, using a Random Forest model trained on engineered time-series features.

> **Scope note:** this is an EV adoption *forecasting* tool. It estimates how fast EV counts will grow per county, which can inform charging infrastructure planning. It does not recommend exact charger locations.

---

## 🚀 Features

- **County selector:** view the historical cumulative EV trend for any county
- **36-month forecast:** recursive, month-by-month predictions from the trained model
- **Growth summary:** percentage change over the forecast horizon vs. the last historical value
- **County comparison:** overlay up to 3 counties on one chart

---

## 🧠 How It Works

1. Load the cleaned dataset (`preprocessed_ev_data.csv`) and the saved model (`forecasting_ev_model.pkl`).
2. For the selected county, take the most recent EV totals.
3. Build lag, rolling-mean, percent-change, and growth-slope features.
4. Predict the next month, append the prediction to the history, and repeat for 36 months.
5. Plot history + forecast and compute the growth percentage.

### Model

- **Algorithm:** `RandomForestRegressor` (scikit-learn)
- **Target:** EV total per county
- **Features:** `months_since_start`, `county_encoded`, `ev_total_lag1`, `ev_total_lag2`, `ev_total_lag3`, `ev_total_roll_mean_3`, `ev_total_pct_change_1`, `ev_total_pct_change_3`, `ev_growth_slope`

### Data

- **Source:** Electric Vehicle Population By County (Washington State)
- **Processing:** dropped rows with missing `Date`, `County`, or EV total; engineered lag, rolling, and trend features; encoded counties
- Raw file: `Electric_Vehicle_Population_By_County.csv` → cleaned file: `preprocessed_ev_data.csv`

---

## 📁 Project Structure

```text
EV-Charging-Point-Prediction/
├── README.md
├── PROJECT_CONTEXT.md
└── Intern/
    ├── app.py                                  # Streamlit app
    ├── EV_Adoption_Forecasting.ipynb           # EDA, feature engineering, model training
    ├── forecasting_ev_model.pkl                # Saved RandomForestRegressor
    ├── preprocessed_ev_data.csv                # Cleaned dataset used by the app
    ├── Electric_Vehicle_Population_By_County.csv  # Raw dataset
    ├── ev-car-factory.jpg                      # Hero image
    └── requirements.txt
```

`app.py`, the model, the dataset, and the image must stay in the same folder.

---

## ▶️ Run Locally

```bash
cd Intern
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

Then open the URL shown in the terminal (usually `http://localhost:8501`).

---

## 📈 Model Evaluation

This is a **regression** task, so it is evaluated with MAE, RMSE, and R².

The current model was checked on a held-out test set using the same engineered feature set as the app:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Random Forest Regressor | 0.0222 | 0.1425 | 0.9939 |

These values reflect the saved model that powers the Streamlit app in `Intern/forecasting_ev_model.pkl`.

---

## 🛠️ Tech Stack

Python · Pandas · NumPy · scikit-learn · Streamlit · Matplotlib · joblib

---

## ⚠️ Limitations

- Prototype for demonstration, not a production planning system.
- Recursive forecasting compounds error over long horizons, so later months are less reliable than early ones.
- Forecasts depend only on historical EV counts; policy changes, incentives, and charger availability are not modeled.
- The saved model is tied to the scikit-learn version in `requirements.txt`.

---

## 🔮 Future Work

- Export the forecast table as CSV
- Add prediction intervals
- Combine adoption forecasts with charger-location data for placement recommendations

---

## 👤 Author

**Archit**: [GitHub](https://github.com/archit7py) · [LinkedIn](https://www.linkedin.com/in/archit-prasad-0379b8352)