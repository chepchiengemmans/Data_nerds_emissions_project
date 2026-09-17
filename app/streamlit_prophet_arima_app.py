from pathlib import Path

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

APP_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = APP_DIR / "src" / "owid-co2-data.csv"

RAW_FUTURE_FEATURES = [
    "population",
    "gdp",
    "primary_energy_consumption",
    "energy_per_capita",
    "energy_per_gdp",
    "coal_co2",
    "oil_co2",
    "gas_co2",
    "cement_co2",
    "flaring_co2",
    "land_use_change_co2",
    "total_ghg",
    "total_ghg_excluding_lucf",
]

st.set_page_config(
    page_title="ARIMA CO2 Forecast",
    page_icon="",
    layout="wide",
)

# Helper function to get existing columns from the dataset
def get_existing_columns(data, columns):
    return [column for column in columns if column in data.columns]


@st.cache_data(show_spinner=False)
def load_dataset():
    return pd.read_csv(DATA_PATH)

# Load and clean the OWID CO2 dataset
@st.cache_data(show_spinner="Cleaning the OWID dataset...")
def clean_data(df_raw):
    df = df_raw[df_raw["iso_code"].notna()].copy()
    df = df.drop_duplicates(subset=["iso_code", "year"], keep="first")

    columns_to_keep = [
        "country", "year", "iso_code", "gdp", "population", "co2",
        "co2_per_capita", "flaring_co2", "cement_co2", "coal_co2", "oil_co2",
        "gas_co2", "land_use_change_co2", "total_ghg", "other_industry_co2",
        "primary_energy_consumption", "energy_per_capita", "co2_including_luc",
        "energy_per_gdp", "total_ghg_excluding_lucf",
    ]
    df = df[get_existing_columns(df, columns_to_keep)].copy()
    # Ensure the 'year' column is of datetime type
    df["year"] = pd.to_datetime(df["year"], format='%Y').dt.year

    # Keep years from 1950 onwards
    df = df[df["year"] >= 1950].copy()
    df = df.replace([np.inf, -np.inf], np.nan).sort_values(["country", "year"])

    # Filter out countries with more than 30% missing CO2 data
    co2_missing_ratio = df.groupby("country")["co2"].transform(lambda values: values.isna().mean())
    df = df[co2_missing_ratio <= 0.30].copy()
    df = df.groupby("country", group_keys=False).filter(lambda values: values["co2"].notna().any())

    return df.reset_index(drop=True)

# Build the model frame with lagged and calculated features for ARIMA modeling
def build_model_frame(data):
    df = data.copy()
    df = df.sort_values(["country", "year"]).reset_index(drop=True)
    grouped = df.groupby("country", group_keys=False)

    df["target_co2"] = df["co2"]

    df["population_t"] = df.get("population")
    df["gdp_t"] = df.get("gdp")
    df["primary_energy_consumption_t"] = df.get("primary_energy_consumption")
    df["energy_per_capita_t"] = df.get("energy_per_capita")
    df["energy_per_gdp_t"] = df.get("energy_per_gdp", np.nan)

    for col in [
        "coal_co2", "oil_co2", "gas_co2", "cement_co2", "flaring_co2",
        "land_use_change_co2", "total_ghg", "total_ghg_excluding_lucf",
    ]:
        if col in df.columns:
            df[f"{col}_t"] = df[col]

    df["gdp_per_capita_calc"] = df["gdp_t"] / df["population_t"]
    df["energy_per_person_calc"] = df["primary_energy_consumption_t"] / df["population_t"]

    df["co2_lag1"] = grouped["co2"].shift(1)
    df["co2_lag2"] = grouped["co2"].shift(2)
    df["co2_lag3"] = grouped["co2"].shift(3)

    shifted_co2 = df.groupby("country")["co2"].shift(1)
    df["co2_roll3"] = (
        shifted_co2.groupby(df["country"], group_keys=False).rolling(3, min_periods=3).mean()
        .reset_index(level=0, drop=True).sort_index()
    )
    df["co2_roll5"] = (
        shifted_co2.groupby(df["country"], group_keys=False).rolling(5, min_periods=5).mean()
        .reset_index(level=0, drop=True).sort_index()
    )

    co2_growth = df.groupby("country")["co2"].pct_change(fill_method=None)
    df["co2_growth_lag1"] = co2_growth.groupby(df["country"], group_keys=False).shift(1)

    df["gdp_lag1"] = grouped["gdp"].shift(1)
    df["population_lag1"] = grouped["population"].shift(1)
    df["energy_lag1"] = grouped["primary_energy_consumption"].shift(1)

    return df.replace([np.inf, -np.inf], np.nan)

# Get the list of feature columns for the model
def get_feature_columns(model_df):
    feature_cols = [
        "year", "population_t", "gdp_t", "primary_energy_consumption_t",
        "energy_per_capita_t", "energy_per_gdp_t", "coal_co2_t", "oil_co2_t",
        "gas_co2_t", "cement_co2_t", "flaring_co2_t", "land_use_change_co2_t",
        "total_ghg_t", "total_ghg_excluding_lucf_t", "gdp_per_capita_calc",
        "energy_per_person_calc", "co2_lag1", "co2_lag2", "co2_lag3",
        "co2_roll3", "co2_roll5", "co2_growth_lag1", "gdp_lag1",
        "population_lag1", "energy_lag1",
    ]
    feature_cols = [c for c in feature_cols if c in model_df.columns]
    return ["country"] + feature_cols

# One-hot encoding for categorical features
def make_one_hot_encoder(sparse_output=True):
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=sparse_output)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=sparse_output)

# Build the preprocessing pipelines for numeric and categorical features
def build_preprocessors(feature_cols):
    categorical_features = ["country"]
    numeric_features = [c for c in feature_cols if c not in categorical_features]

    numeric_transformer = Pipeline(
        steps=[("imputer", IterativeImputer(random_state=0)), ("scaler", StandardScaler())]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            (
                "cat",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", make_one_hot_encoder())]),
                categorical_features,
            ),
        ]
    )
    hgb_preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            (
                "cat",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", make_one_hot_encoder(sparse_output=False))]),
                categorical_features,
            ),
        ]
    )
    return preprocessor, hgb_preprocessor


def regression_metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        #"R2": r2_score(y_true, y_pred),
    }

# Train CO2 prediction models using candidate regressors and evaluate their performance on validation data
@st.cache_resource(show_spinner="Training candidate CO2 models...")
def train_co2_model():
    df_raw = load_dataset()
    df_cleaned = clean_data(df_raw)
    model_df = build_model_frame(df_cleaned)
    model_df = model_df[model_df["target_co2"].notna()].copy()
    feature_cols = get_feature_columns(model_df)

    train_df = model_df[model_df["year"] <= 2013].copy()
    valid_df = model_df[(model_df["year"] >= 2014) & (model_df["year"] <= 2018)].copy()
    test_df = model_df[(model_df["year"] >= 2019) & (model_df["year"] <= 2024)].copy()

    X_train, y_train = train_df[feature_cols], train_df["target_co2"]
    X_valid, y_valid = valid_df[feature_cols], valid_df["target_co2"]
    X_test, y_test = test_df[feature_cols], test_df["target_co2"]

    preprocessor, hgb_preprocessor = build_preprocessors(feature_cols)

    candidate_models = {
        "Ridge": Pipeline([("preprocessor", preprocessor), ("model", Ridge(alpha=10.0))]),
        "Random Forest": Pipeline([
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor(n_estimators=200, max_depth=12, min_samples_leaf=2, random_state=42, n_jobs=-1)),
        ]),
        "Extra Trees": Pipeline([
            ("preprocessor", preprocessor),
            ("model", ExtraTreesRegressor(n_estimators=200, max_depth=12, min_samples_leaf=2, random_state=42, n_jobs=-1)),
        ]),
        "HistGradientBoosting": Pipeline([
            ("preprocessor", hgb_preprocessor),
            ("model", HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0, random_state=42)),
        ]),
    }

    rows = []
    baseline_valid_metrics = regression_metrics(y_valid, valid_df["co2_lag1"])
    rows.append({"Model": "Naive persistence", **{f"Validation {k}": v for k, v in baseline_valid_metrics.items()}})

    for name, model in candidate_models.items():
        model.fit(X_train, y_train)
        valid_pred = np.maximum(model.predict(X_valid), 0)
        rows.append({"Model": name, **{f"Validation {k}": v for k, v in regression_metrics(y_valid, valid_pred).items()}})

    comparison = pd.DataFrame(rows).sort_values("Validation RMSE", na_position="last").reset_index(drop=True)
    best_model_name = comparison.loc[comparison["Model"] != "Naive persistence", "Model"].iloc[0]
    best_model = candidate_models[best_model_name]

    test_pred = np.maximum(best_model.predict(X_test), 0)
    test_metrics = regression_metrics(y_test, test_pred)

    # Refit the selected model on all observed labelled data for production use.
    production_model = Pipeline(best_model.steps)
    production_model.fit(model_df[feature_cols], model_df["target_co2"])

    return {
        "df_cleaned": df_cleaned,
        "model_df": model_df,
        "feature_cols": feature_cols,
        "comparison": comparison,
        "best_model_name": best_model_name,
        "test_metrics": test_metrics,
        "production_model": production_model,
        "latest_year": int(df_cleaned["year"].max()),
    }

# Stage 1: Prophet / ARIMA future-feature forecasting
def prepare_feature_series(country_df, feature, cutoff_year):
    series = country_df[["year", feature]].dropna().drop_duplicates("year").sort_values("year")
    series = series[series["year"] <= cutoff_year].copy()
    if len(series) < 8:
        return None

    idx = pd.RangeIndex(int(series["year"].min()), cutoff_year + 1)
    s = series.set_index("year")[feature].reindex(idx)
    s = s.interpolate(limit_area="inside").dropna()
    if len(s) < 8:
        return None
    return s.astype(float)

# Stage 1a: ARIMA-based feature forecasting
def forecast_feature_arima(country_df, feature, cutoff_year, future_years, order=(1, 1, 1)):
    series = prepare_feature_series(country_df, feature, cutoff_year)
    if series is None or len(future_years) == 0:
        return np.full(len(future_years), np.nan)

    transformed = np.log1p(np.clip(series.values, 0, None))
    try:
        model = ARIMA(transformed, order=order, enforce_stationarity=False, enforce_invertibility=False).fit()
        pred_log = np.asarray(model.forecast(steps=len(future_years)), dtype=float)
        return np.maximum(np.expm1(pred_log), 0.0)
    except Exception:
        x = np.arange(len(transformed))
        degree = 1 if len(transformed) >= 2 else 0
        coef = np.polyfit(x, transformed, degree)
        future_x = np.arange(len(transformed), len(transformed) + len(future_years))
        pred = np.polyval(coef, future_x)
        return np.maximum(np.expm1(pred), 0.0)

# Stage 1b: Prophet-based feature forecasting
def forecast_feature_prophet(country_df, feature, cutoff_year, future_years):
    series = prepare_feature_series(country_df, feature, cutoff_year)
    if series is None or len(future_years) == 0:
        return np.full(len(future_years), np.nan)

    try:
        from prophet import Prophet
    except ImportError:
        return forecast_feature_arima(country_df, feature, cutoff_year, future_years)

    prophet_df = pd.DataFrame({
        "ds": pd.to_datetime(series.index.astype(str), format="%Y"),
        "y": np.log1p(np.clip(series.values, 0, None)),
    })

    try:
        model = Prophet(growth="linear", yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
        model.fit(prophet_df)
        future_dates = pd.DataFrame({"ds": pd.to_datetime([str(y) for y in future_years], format="%Y")})
        forecast = model.predict(future_dates)
        pred_log = forecast["yhat"].to_numpy(dtype=float)
        return np.maximum(np.expm1(pred_log), 0.0)
    except Exception:
        x = np.arange(len(prophet_df))
        coef = np.polyfit(x, prophet_df["y"].to_numpy(), 1)
        future_x = np.arange(len(prophet_df), len(prophet_df) + len(future_years))
        pred = np.polyval(coef, future_x)
        return np.maximum(np.expm1(pred), 0.0)

# Stage 2: Forecast all features for a given country using the selected methods
def forecast_country_features(country_df, target_year, method_map, future_feature_cols):
    country_df = country_df.sort_values("year").copy()
    cutoff_year = int(country_df["year"].max())
    future_years = list(range(cutoff_year + 1, target_year + 1))

    out = country_df[["country", "year"] + future_feature_cols].drop_duplicates("year").copy()
    if not future_years:
        return out

    future_rows = pd.DataFrame({"country": country_df["country"].iloc[0], "year": future_years})
    for feature in future_feature_cols:
        method = method_map.get(feature, "ARIMA")
        if method == "Prophet":
            pred = forecast_feature_prophet(country_df, feature, cutoff_year, future_years)
        else:
            pred = forecast_feature_arima(country_df, feature, cutoff_year, future_years)
        future_rows[feature] = pred

    return pd.concat([out, future_rows], ignore_index=True).sort_values("year").reset_index(drop=True)

# Stage 3: Select the best forecasting method for each feature based on backtesting
@st.cache_data(show_spinner="Backtesting Prophet vs ARIMA for each feature...")
def select_feature_forecast_methods(_df_cleaned, use_prophet, sample_size=15):
    future_feature_cols = [c for c in RAW_FUTURE_FEATURES if c in _df_cleaned.columns]

    if not use_prophet:
        return {feature: "ARIMA" for feature in future_feature_cols}, future_feature_cols

    latest_country_co2 = (
        _df_cleaned.dropna(subset=["co2"]).sort_values("year").groupby("country").tail(1).nlargest(sample_size, "co2")
    )
    backtest_countries = latest_country_co2["country"].tolist()
    cutoff, backtest_year = 2013, 2014

    rows = []
    for feature in future_feature_cols:
        for method in ["ARIMA", "Prophet"]:
            actual_values, predicted_values = [], []
            for country in backtest_countries:
                country_df = _df_cleaned[_df_cleaned["country"] == country]
                actual = country_df.loc[country_df["year"] == backtest_year, feature]
                if actual.empty or pd.isna(actual.iloc[0]):
                    continue
                fn = forecast_feature_prophet if method == "Prophet" else forecast_feature_arima
                pred = fn(country_df, feature, cutoff, [backtest_year])[0]
                if np.isfinite(pred):
                    actual_values.append(float(actual.iloc[0]))
                    predicted_values.append(float(pred))

            if actual_values:
                rows.append({
                    "Feature": feature, "Method": method,
                    "RMSE": np.sqrt(mean_squared_error(actual_values, predicted_values)),
                })

    scores = pd.DataFrame(rows)
    if scores.empty:
        return {feature: "ARIMA" for feature in future_feature_cols}, future_feature_cols

    best = scores.sort_values(["Feature", "RMSE"]).groupby("Feature", as_index=False).first()
    method_map = {row["Feature"]: row["Method"] for _, row in best.iterrows()}
    return method_map, future_feature_cols


def build_future_model_row(country, year, feature_history, co2_history, feature_cols):
    current = feature_history.loc[feature_history["year"] == year].iloc[0]
    prior_co2 = pd.Series(co2_history, dtype=float).sort_index()

    row = {
        "country": country, "year": year,
        "population_t": current.get("population", np.nan),
        "gdp_t": current.get("gdp", np.nan),
        "primary_energy_consumption_t": current.get("primary_energy_consumption", np.nan),
        "energy_per_capita_t": current.get("energy_per_capita", np.nan),
        "energy_per_gdp_t": current.get("energy_per_gdp", np.nan),
        "coal_co2_t": current.get("coal_co2", np.nan),
        "oil_co2_t": current.get("oil_co2", np.nan),
        "gas_co2_t": current.get("gas_co2", np.nan),
        "cement_co2_t": current.get("cement_co2", np.nan),
        "flaring_co2_t": current.get("flaring_co2", np.nan),
        "land_use_change_co2_t": current.get("land_use_change_co2", np.nan),
        "total_ghg_t": current.get("total_ghg", np.nan),
        "total_ghg_excluding_lucf_t": current.get("total_ghg_excluding_lucf", np.nan),
    }

    population, gdp, energy = row["population_t"], row["gdp_t"], row["primary_energy_consumption_t"]
    row["gdp_per_capita_calc"] = gdp / population if pd.notna(gdp) and pd.notna(population) and population != 0 else np.nan
    row["energy_per_person_calc"] = energy / population if pd.notna(energy) and pd.notna(population) and population != 0 else np.nan

    row["co2_lag1"] = prior_co2.get(year - 1, np.nan)
    row["co2_lag2"] = prior_co2.get(year - 2, np.nan)
    row["co2_lag3"] = prior_co2.get(year - 3, np.nan)

    prior_values = prior_co2.loc[prior_co2.index < year]
    row["co2_roll3"] = prior_values.tail(3).mean() if len(prior_values) >= 3 else np.nan
    row["co2_roll5"] = prior_values.tail(5).mean() if len(prior_values) >= 5 else np.nan

    if len(prior_values) >= 2 and prior_values.iloc[-2] != 0:
        row["co2_growth_lag1"] = prior_values.iloc[-1] / prior_values.iloc[-2] - 1
    else:
        row["co2_growth_lag1"] = np.nan

    previous_feature = feature_history.loc[feature_history["year"] == year - 1]
    if not previous_feature.empty:
        previous_feature = previous_feature.iloc[0]
        row["gdp_lag1"] = previous_feature.get("gdp", np.nan)
        row["population_lag1"] = previous_feature.get("population", np.nan)
        row["energy_lag1"] = previous_feature.get("primary_energy_consumption", np.nan)
    else:
        row["gdp_lag1"] = row["population_lag1"] = row["energy_lag1"] = np.nan

    return pd.DataFrame([{col: row.get(col, np.nan) for col in feature_cols}])

# Stage 4: Recursive future CO2 forecasting for a single country simply put using the best feature-based CO2 model
def recursive_country_co2_forecast(
    country_df,
    target_year,
    model,
    feature_cols,
    method_map,
    future_feature_cols,
    manual_future_features=None,
):
    country = country_df["country"].iloc[0]
    observed_co2 = country_df[["year", "co2"]].dropna().drop_duplicates("year").sort_values("year")
    co2_history = {int(r["year"]): float(r["co2"]) for _, r in observed_co2.iterrows()}

    feature_history = forecast_country_features(country_df, target_year, method_map, future_feature_cols)
    if manual_future_features and target_year in feature_history["year"].values:
        for feature, value in manual_future_features.items():
            if feature in feature_history.columns:
                feature_history.loc[feature_history["year"] == target_year, feature] = value
    last_observed = int(country_df["year"].max())

    predictions = []
    for year in range(last_observed + 1, target_year + 1):
        row = build_future_model_row(country, year, feature_history, co2_history, feature_cols)
        prediction = float(np.maximum(model.predict(row)[0], 0))
        co2_history[year] = prediction
        predictions.append({"country": country, "year": year, "predicted_co2 (million tons)": prediction})

    return pd.DataFrame(predictions), feature_history


# App

def main():
    st.title("CO2 Forecasting Dashboard")
    view = st.sidebar.radio(
        "Navigate",
        ["Forecast dashboard", "About"],# "Future feature inputs",
        index=0,
    )

    # if view == "Future feature inputs":
    #     st.header("Future feature inputs")
    #     st.write(
    #         "Enter values for the selected country and target year. Saved values override "
    #         "the automatically forecast features for that target year."
    #     )

    #     input_data = clean_data(load_dataset())
    #     input_countries = sorted(input_data["country"].unique())
    #     input_country = st.selectbox("Country", input_countries, key="input_country")
    #     input_latest_year = int(input_data["year"].max())
    #     input_year = st.number_input(
    #         "Target year",
    #         min_value=input_latest_year + 1,
    #         max_value=input_latest_year + 30,
    #         value=input_latest_year + 5,
    #         step=1,
    #         format="%d",
    #         key="input_year",
    #     )

    #     latest_country_row = (
    #         input_data[input_data["country"] == input_country]
    #         .sort_values("year")
    #         .iloc[-1]
    #     )
    #     st.caption(
    #         "Fields are prefilled with the latest available value for this country. "
    #         "Edit them before saving."
    #     )

    #     feature_values = {}
    #     input_columns = st.columns(2)
    #     for index, feature in enumerate(RAW_FUTURE_FEATURES):
    #         latest_value = latest_country_row.get(feature, np.nan)
    #         default_value = float(latest_value) if pd.notna(latest_value) else 0.0
    #         feature_values[feature] = input_columns[index % 2].number_input(
    #             feature.replace("_", " ").title(),
    #             value=default_value,
    #             format="%.6f",
    #             key=f"future_input_{input_country}_{int(input_year)}_{feature}",
    #         )

    #     if st.button("Save future feature inputs", type="primary"):
    #         overrides = st.session_state.setdefault("future_feature_overrides", {})
    #         overrides[(input_country, int(input_year))] = feature_values
    #         st.success(f"Future feature inputs saved for {input_country} ({int(input_year)}).")
    #     return

    if view == "About":
        st.header("About the CO2 Forecasting Dashboard")
        st.write(
            "This dashboard forecasts country-level CO2 emissions using annual data from "
            "Our World in Data."
        )
        st.subheader("How it works")
        st.markdown(
            """
            1. Historical economic, population, energy, and emissions data are cleaned and transformed.
            2. Prophet or ARIMA forecasts future input features.
            3. A machine-learning model produces the recursive CO2 forecast for the selected country.
            """
        )
        st.subheader("Models")
        st.write(
            "The app compares Ridge Regression, Random Forest, Extra Trees, and "
            "HistGradientBoosting models using chronological validation data."
        )
        st.subheader("Data and limitations")
        st.markdown(
            """
            - Source: Our World in Data CO2 dataset.
            - Coverage: country-level annual observations from the locally bundled CSV file.
            - Forecasts are planning estimates, not guarantees.
            - Results depend on the source data and future-feature projections.
            """
        )
        return

    st.caption(
        "Two-stage country-level forecast based on "
        "Final_Notebook_Prophet_ARIMA_Feature_CO2_Forecast.ipynb: "
        "future predictors are forecast with Prophet/ARIMA, then fed into the best feature-based CO2 model."
    )

    try:
        results = train_co2_model()
    except FileNotFoundError:
        st.error(f"Could not find {DATA_PATH}. Keep the CSV in the same folder as this app.")
        st.stop()

    df_cleaned = results["df_cleaned"]
    latest_year = results["latest_year"]

    metric_cols = st.columns(4)
    metric_cols[0].metric("Countries", f"{df_cleaned['country'].nunique():,}")
    metric_cols[1].metric("Model rows", f"{len(results['model_df']):,}")
    metric_cols[2].metric("Latest observed year", latest_year)
    metric_cols[3].metric("Selected CO2 model", results["best_model_name"])

    st.subheader("Feature-Based Model Comparison (Validation)")
    st.dataframe(results["comparison"].round(3), use_container_width=True, hide_index=True)
    st.caption(
        f"Test metrics for {results['best_model_name']}: "
        f"MAE={results['test_metrics']['MAE']:.3f}, "
        f"RMSE={results['test_metrics']['RMSE']:.3f} "
        #f"R2={results['test_metrics']['R2']:.3f}"
    )

    st.sidebar.header("Forecast settings")
    countries = sorted(df_cleaned["country"].unique())
    default_country = "United States" if "United States" in countries else countries[0]
    country = st.sidebar.selectbox("Country", countries, index=countries.index(default_country))
    forecast_year = st.sidebar.number_input(
        "Forecast year", min_value=latest_year + 1, max_value=latest_year + 30, value=latest_year + 5, step=1
    )
    use_prophet = st.sidebar.checkbox(
        "Compare Prophet vs ARIMA per feature (slower)",
        value=False,
        help="If unchecked, ARIMA is used for every future predictor to keep the app responsive.",
    )
    run_forecast = st.sidebar.button("Run forecast", type="primary")

    if not run_forecast:
        st.info("Choose a country and forecast year in the sidebar, then click **Run forecast**.")
        return

    method_map, future_feature_cols = select_feature_forecast_methods(df_cleaned, use_prophet)

    st.subheader("Selected Future-Feature Forecasting Method")
    st.dataframe(
        pd.DataFrame(
            [{"Feature": feature, "Method": method_map.get(feature, "ARIMA")} for feature in future_feature_cols]
        ),
        use_container_width=True,
        hide_index=True,
    )

    country_df = df_cleaned[df_cleaned["country"] == country].copy()
    manual_overrides = st.session_state.get("future_feature_overrides", {}).get(
        (country, int(forecast_year))
    )

    with st.spinner(f"Forecasting {country} through {forecast_year}..."):
        predictions, feature_history = recursive_country_co2_forecast(
            country_df=country_df,
            target_year=forecast_year,
            model=results["production_model"],
            feature_cols=results["feature_cols"],
            method_map=method_map,
            future_feature_cols=future_feature_cols,
            manual_future_features=manual_overrides,
        )

    if predictions.empty:
        st.warning(f"Not enough history for {country} to build a forecast.")
        return

    st.subheader(f"{country} CO2 Forecast Through {forecast_year}")

    history = country_df[["year", "co2"]].dropna().rename(columns={"co2": "predicted_co2 (million tons)"})
    history["Type"] = "Observed"
    future = predictions[["year", "predicted_co2 (million tons)"]].copy()
    future["Type"] = "Forecast"
    combined = pd.concat([history, future], ignore_index=True).sort_values("year")

    chart_data = combined.pivot(index="year", columns="Type", values="predicted_co2 (million tons)")
    chart = (
        alt.Chart(chart_data.reset_index().melt(id_vars="year", var_name="Type", value_name="co2"))
        .mark_line()
        .encode(
            x=alt.X("year:Q", title="Year", axis=alt.Axis(format="d")),
            y=alt.Y("co2:Q", title="CO2 emissions (million tons)"),
            color=alt.Color("Type:N", title="Series"),
            tooltip=[
                alt.Tooltip("year:Q", title="Year", format="d"),
                alt.Tooltip("Type:N", title="Series"),
                alt.Tooltip("co2:Q", title="CO2 emissions (million tons)", format=",.2f"),
            ],
        )
        .properties(height=420)
    )
    st.altair_chart(chart, use_container_width=True)

    st.dataframe(
        predictions.round(3),
        column_config={"year": st.column_config.NumberColumn("Year", format="%d")},
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download country forecast CSV",
        predictions.to_csv(index=False).encode("utf-8"),
        file_name=f"{country.replace(' ', '_').lower()}_co2_forecast_{forecast_year}.csv",
        mime="text/csv",
    )

    with st.expander("Forecasted future predictors used as model inputs"):
        st.dataframe(feature_history.round(3), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
