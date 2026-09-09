# CRISP-DM: Country-Level CO2 Emissions Forecasting

## Project Overview

This project uses the Our World in Data (OWID) CO2 and greenhouse-gas emissions dataset to forecast annual country-level CO2 emissions. The current notebook implementation builds a one-year-ahead forecasting pipeline and evaluates several machine-learning models against a naive persistence baseline.

- **Unit of analysis:** country-year
- **Target:** `co2` (million tonnes)
- **Country identifier:** `iso_code`
- **Historical period:** 1950 onward
- **Current modeling scope:** African countries
- **Forecast horizon:** one year ahead
- **Latest forecast generated:** 2025, using 2024 observations

## 1. Business Understanding

### Objective

Develop an accurate and interpretable forecasting process that can estimate a country's future CO2 emissions and help identify whether emissions are likely to increase or decrease.

### Questions

1. How accurately can historical emissions predict the following year's emissions?
2. Which economic, population, energy, and historical-emissions variables are useful predictors?
3. How do emissions trends differ between countries?
4. Which modeling approach performs best on unseen years?

### Success Criteria

A successful solution should:

- outperform a naive persistence baseline;
- preserve chronological order and avoid temporal leakage;
- achieve low MAE and RMSE with strong R2 where appropriate;
- perform reasonably across individual countries;
- provide interpretable information about important predictors; and
- produce non-negative country-level forecasts.

## 2. Data Understanding

### Data Source

The project uses the OWID CO2 dataset stored in `owid-co2-data.csv`. The source contains country and regional observations, annual emissions measures, and economic, demographic, and energy indicators.

The raw dataset used by the notebook contains 50,411 rows and 79 columns. Coverage and missingness vary substantially by country and year.

### Main Variables

| Group | Variables | Role |
|---|---|---|
| Identification | `country`, `iso_code`, `year` | Country and time index |
| Target | `co2` | Total CO2 emissions |
| Economic | `gdp`, derived GDP per capita | Economic activity |
| Demographic | `population`, lagged population | Country scale and growth |
| Energy | `primary_energy_consumption`, `energy_per_capita`, `energy_per_gdp` | Energy use and intensity |
| Emissions composition | `coal_co2`, `oil_co2`, `gas_co2`, `cement_co2`, `flaring_co2`, `land_use_change_co2` | Emissions sources |
| Greenhouse gases | `total_ghg`, `total_ghg_excluding_lucf` | Broader emissions context |

### Initial Findings

- CO2 is strongly associated with primary energy consumption and GDP.
- Coal, oil, and gas emissions are also strongly related to total CO2.
- Missing values are unevenly distributed and are more common in earlier or less-complete country records.
- Emissions vary greatly in scale between countries, so country and time effects must be handled carefully.

## 3. Data Preparation

The notebook prepares the data using the following process:

1. Keep rows with a valid `iso_code` so regional aggregates can be filtered out.
2. Retain observations from 1950 onward.
3. Keep variables relevant to CO2 forecasting.
4. Sort observations by country and year.
5. Remove duplicate country-year observations.
6. Replace infinite values with missing values.
7. Remove countries without usable CO2 observations.
8. Create a supervised learning target by shifting CO2 one year forward.
9. Remove rows where the next year's target is unavailable.

### Missing-Value Handling

- Numeric features use median imputation followed by standardization.
- The country feature uses most-frequent imputation followed by one-hot encoding.
- Unknown countries are ignored by the encoder so the preprocessing pipeline can handle new categories.

## 4. Feature Engineering

The model frame is created separately for each country after sorting chronologically. It includes:

- CO2 lags: `co2_lag1`, `co2_lag2`, and `co2_lag3`;
- rolling CO2 means: `co2_roll3` and `co2_roll5`;
- lagged GDP, population, and energy consumption;
- calculated GDP per capita and energy per person;
- current-year energy, economic, population, and emissions-composition variables; and
- `target_co2`, defined as the following year's CO2 emissions.

This creates the forecasting relationship:

> Historical CO2 + economic activity + energy use + population information -> next year's CO2 emissions

All lag and rolling features must be calculated using past observations only. This is the main protection against temporal leakage.

## 5. Modeling

The notebook compares five approaches:

### Naive Persistence Baseline

Predict the next year's emissions as equal to the current year's emissions. This establishes the minimum useful performance standard for a time-series model.

### Ridge Regression

A regularized linear model that provides an interpretable benchmark while reducing instability caused by correlated economic, energy, and emissions variables.

### Random Forest Regressor

A nonlinear ensemble of decision trees that can capture interactions between country characteristics, energy use, and emissions history.

### Extra Trees Regressor

A randomized tree ensemble used to test whether additional split randomization improves generalization over Random Forest.

### HistGradientBoosting Regressor

A sequential tree-boosting model designed to capture more complex nonlinear relationships. It uses dense one-hot encoded country features.

All machine-learning models are implemented in preprocessing-and-model pipelines so transformations are fitted only on training data.

## 6. Evaluation

### Temporal Split

The current notebook uses a chronological split:

- **Training:** years through 2014
- **Validation:** 2015-2018
- **Test:** 2019 onward

The validation set is used to compare models during development. The test set is reserved for the final unbiased evaluation.

### Metrics

- **MAE:** average absolute prediction error, in million tonnes of CO2.
- **RMSE:** penalizes larger errors more heavily than MAE.
- **R2:** proportion of target variation explained by the model.

Lower MAE and RMSE are better. Higher R2 is better, but it should be interpreted alongside absolute error and country-level results.

### Model Selection Rule

Select the model with the lowest validation RMSE, then confirm its performance on the untouched test set. The selected model should also be checked for stability across countries rather than relying only on one aggregate score.

## 7. Deployment and Forecast Use

The notebook creates 2025 country-level forecasts from the latest observed 2024 data for countries represented in the training data. Predictions are clipped at zero because negative CO2 emissions are not meaningful for this target definition.

A practical deployment workflow should:

1. update the OWID dataset;
2. rerun the cleaning and feature-engineering steps;
3. retrain or refit the selected pipeline using the updated historical data;
4. generate the next-year country forecasts; and
5. export forecasts with the source year, model version, and data version.

## 8. Monitoring and Maintenance

After deployment, monitor:

- forecast MAE and RMSE once actual emissions become available;
- performance by country and by emissions scale;
- changes in missingness and dataset coverage;
- feature drift in GDP, population, energy, and emissions variables;
- countries entering or leaving the training population; and
- whether the selected model continues to beat naive persistence.

Retraining should occur when new annual OWID data is released or when monitoring shows meaningful performance degradation.

## 9. Limitations and Next Steps

- The current notebook focuses on African countries in its modeling section, while the project introduction describes a broader country-level goal.
- The data is annual, so the model cannot describe within-year changes.
- Correlated emissions-source variables may make feature importance difficult to interpret causally.
- Missing-value imputation can affect countries with sparse histories.
- A single global model may not perform equally well for small and large emitters.
- Multi-year forecasts would accumulate uncertainty and require recursive or direct multi-step validation.

Recommended next steps are to report country-level error distributions, compare against country-specific baselines, test rolling-origin validation, document the final model choice with recorded metrics, and decide whether the final product should cover Africa or all countries with valid data.
