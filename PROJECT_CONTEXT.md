# EV Charging Point Prediction App — Project Context

## 1) Project purpose

This project is an EV adoption forecasting dashboard for counties in Washington state. It combines a processed historical dataset with a trained machine learning model to estimate how the cumulative number of electric vehicles will grow over time.

The app is designed as a user-friendly Streamlit interface where a user can:
- choose a county,
- view historical EV trends,
- forecast the next 36 months,
- compare up to 3 counties side by side.

This is not a generic charging-station placement optimizer. It is a forecasting app whose output is the expected EV adoption trend, not directly the optimal charger placement.

---

## 2) Repository structure

```text
EV-Charging-Point-Prediction/
├── README.md
├── PROJECT_CONTEXT.md
├── Intern/
│   ├── app.py
│   ├── ev-car-factory.jpg
│   ├── forecasting_ev_model.pkl
│   ├── preprocessed_ev_data.csv
│   ├── Electric_Vehicle_Population_By_County.csv
│   ├── EV_Adoption_Forecasting.ipynb
│   └── requirements.txt
└── .git/
```

Key files:
- `Intern/app.py` — main Streamlit app
- `Intern/preprocessed_ev_data.csv` — cleaned historical EV dataset used by the app
- `Intern/forecasting_ev_model.pkl` — saved trained RandomForestRegressor model
- `Intern/EV_Adoption_Forecasting.ipynb` — exploratory notebook used to build the pipeline
- `Intern/requirements.txt` — Python dependencies
- `README.md` — general project summary

---

## 3) What the app does

The app does the following:

1. Loads the processed dataset from `preprocessed_ev_data.csv`
2. Loads the saved ML model from `forecasting_ev_model.pkl`
3. Displays a county selector
4. For the selected county:
   - loads historical EV totals,
   - creates forecast features based on recent values,
   - predicts the next 36 months,
   - plots historical and forecast cumulative EV counts,
   - shows percentage change over the forecast horizon
5. Allows comparison for up to 3 counties on the same graph

The output is a cumulative EV adoption forecast visualized as a chart and summary text.

---

## 4) Dataset details

The dataset is named:
- `Intern/preprocessed_ev_data.csv`

It contains monthly or periodic county-level EV data, including fields like:
- `Date`
- `County`
- `State`
- `Vehicle Primary Use`
- `Battery Electric Vehicles (BEVs)`
- `Plug-In Hybrid Electric Vehicles (PHEVs)`
- `Electric Vehicle (EV) Total`
- `Non-Electric Vehicle Total`
- `Total Vehicles`
- `Percent Electric Vehicles`
- `year`
- `month`
- `numeric_date`
- `county_encoded`
- `months_since_start`
- `ev_total_lag1`
- `ev_total_lag2`
- `ev_total_lag3`
- `ev_total_roll_mean_3`
- `ev_total_pct_change_1`
- `ev_total_pct_change_3`
- `cumulative_ev`
- `ev_growth_slope`

The app expects these columns to exist and uses them for forecasting.

---

## 5) Model type and training logic

The saved model is a `RandomForestRegressor` trained to predict EV total values.

The model was trained using engineered time-series features such as:
- `months_since_start`
- `county_encoded`
- `ev_total_lag1`
- `ev_total_lag2`
- `ev_total_lag3`
- `ev_total_roll_mean_3`
- `ev_total_pct_change_1`
- `ev_total_pct_change_3`
- `ev_growth_slope`

This means the app is not simply using raw values; it is creating lag-based and trend-based features to forecast future EV totals.

The app uses the model in a custom recursive forecasting loop:
- take the last 6 historical EV values,
- derive lag and slope features,
- call the model for the next month,
- append the prediction,
- continue for 36 months.

---

## 6) Runtime environment

The app depends on Python packages such as:
- streamlit
- pandas
- numpy
- scikit-learn
- matplotlib
- joblib
- scipy
- statsmodels

Current project requirements are listed in:
- `Intern/requirements.txt`

The project was standardized to avoid unnecessary or incompatible heavy packages for this specific app. TensorFlow was removed from the effective app dependency list because it was not required for the model execution or dashboard itself.

---

## 7) How the app works internally

### Main flow in `app.py`

1. Set page configuration:
   - `st.set_page_config(page_title="EV Forecast", layout="wide")`
2. Load the model:
   - `load_model()`
3. Load the data:
   - `load_data()`
4. Build the UI:
   - title,
   - image,
   - county selector,
   - forecast section,
   - comparison section
5. Generate forecast data for selected county
6. Plot cumulative history + forecast values
7. Compute growth percentage relative to the last historical value
8. Allow comparison among up to 3 counties

### Forecast generation helper

The app contains a helper function called `generate_forecast(model, county_df, forecast_horizon=36)`.

This function:
- sorts by date,
- gets the last few EV totals,
- calculates lag and trend features,
- creates a DataFrame with the required model input shape,
- predicts one month at a time,
- appends the predicted value to the rolling features,
- repeats for the specified horizon.

This is the core prediction logic.

---

## 8) Important implementation notes

### A. Model file and data file must stay together

The app assumes that:
- `forecasting_ev_model.pkl`
- `preprocessed_ev_data.csv`
- `ev-car-factory.jpg`

are all located in the same folder as `app.py`.

If they are moved, the app may fail unless paths are updated.

### B. Data cleaning matters

The app drops invalid rows where:
- `Date` is missing,
- `County` is missing,
- `Electric Vehicle (EV) Total` is missing.

This prevents model and chart failures.

### C. Forecasting uses recursive updates

The model predicts the next month, then that prediction is pushed into the rolling history so the next month can be predicted with updated state.

This is a valid short-term forecasting approach for a time-series trend model.

### D. Model compatibility

The saved model had to be regenerated under the version of scikit-learn installed in the environment to avoid pickle version mismatch warnings and runtime errors.

That means the project is intentionally tied to the installed package versions in the requirements file.

---

## 9) How to run the project

From the project folder:

```bash
cd "C:\Users\archi\Desktop\ev\EV-Charging-Point-Prediction\Intern"
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL shown in the terminal, usually:
- `http://localhost:8501` or another assigned port.

---

## 10) What the UI shows

The dashboard includes:
- a title banner,
- a hero image,
- county dropdown,
- cumulative EV trend chart,
- forecast summary text,
- county comparison chart,
- growth summary for selected counties.

The app is designed to be polished and presentation-friendly, which is useful for demos and interviews.

---

## 11) Known status and quality notes

This project is working as a forecasting app and has been repaired to handle runtime issues gracefully. It is suitable for demonstration and local use.

That said, the project is still a prototype-level forecasting dashboard and is not a full production-ready EV infrastructure planning system.

---

## 12) Quick summary for future LLMs or coding agents

If an LLM is asked to work on this repository, this is the most important contextual summary:

- This project is a Streamlit EV adoption forecasting app.
- It uses processed county-level EV data and a saved RandomForestRegressor.
- The app forecasts future EV totals for the next 36 months.
- The main code is in `Intern/app.py`.
- The model and data must remain in the same folder as the app.
- The app is designed to run successfully in a local environment after dependency installation.
- The project is not a deployment-ready charging-station recommendation engine; it is a forecasting dashboard for EV adoption trends.

---

## 13) Suggested prompts for future AI assistants

If a future model needs to continue from here, these prompts are highly effective:

```text
Review the EV forecasting app and explain how the forecast pipeline works.
```

```text
Trace the data flow from CSV to Streamlit chart in this project.
```

```text
Fix any model/data compatibility issues in the EV forecast app.
```

```text
Add a feature to export the forecast table as CSV.
```

```text
Improve the Streamlit UI for this EV adoption dashboard.
```

---

## 14) Final takeaway

This project is an EV adoption forecasting dashboard built around a cleaned historical dataset and a trained regression model. The real work is in the combination of:
- dataset processing,
- feature engineering,
- recursive forecasting logic,
- Streamlit visualization,
- county comparison UI.

It is a good demonstration project for machine learning, time-series forecasting, and dashboard building.
