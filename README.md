# Country-Level CO2 Emissions Forecasting

This project forecasts next-year CO2 emissions for individual countries using historical emissions, economic, population, and energy data from Our World in Data (OWID).

## Project goals

- Prepare a clean country-year emissions dataset.
- Engineer lagged and rolling historical features.
- Forecast one year ahead without using future information.
- Compare a naive persistence baseline with machine-learning models.
- Generate country-level forecasts and assess generalisation on unseen years.

## Dataset

The project uses [`owid-co2-data.csv`](owid-co2-data.csv), containing annual country and regional observations from OWID. The main target is `co2`, measured in million tonnes.

Important fields include:

- `country`, `iso_code`, and `year`
- `co2` and fuel-specific emissions
- `gdp` and population measures
- primary energy consumption and energy-intensity measures
- greenhouse-gas totals

Rows without a usable country code are excluded so regional aggregates do not mix with country-level forecasts.

## Notebook

The main analysis is in [`forecast_cleaning_updated_with_future_table.ipynb`](forecast_cleaning_updated_with_future_table.ipynb). It covers:

1. Data loading and quality checks
2. Cleaning, duplicate removal, and missing-value handling
3. Lag and rolling-feature engineering
4. Creation of the next-year target, `target_co2`
5. Chronological train, validation, and test evaluation
6. Baseline and candidate-model comparison
7. Hyperparameter tuning
8. Error analysis and future-feature availability
9. Next-year forecasting

## Forecasting approach

The supervised-learning target is the following year's CO2 value:

```text
prediction inputs from year t -> target CO2 for year t + 1
```

Historical features include `co2_lag1`, `co2_lag2`, `co2_lag3`, rolling CO2 means, and lagged economic, population, and energy variables. Numeric features are imputed and scaled; the country feature is one-hot encoded inside scikit-learn pipelines.

### Naive persistence baseline

The baseline predicts that next year's emissions will equal the current year's emissions:

```text
forecast(t + 1) = observed CO2(t)
```

This is a meaningful benchmark because annual country-level emissions are highly persistent. A more complex model should outperform it on unseen data before it is adopted.

### Candidate models

- Ridge Regression
- Random Forest Regressor
- Extra Trees Regressor
- HistGradientBoosting Regressor

Models are ranked using validation RMSE. The test period is kept separate for final generalisation reporting.

## Evaluation

The notebook uses a chronological split so future observations do not enter model development:

- Training: years before 2014
- Validation: 2014-2018
- Test: 2019-2023

Metrics are:

- **MAE:** average absolute error in million tonnes of CO2
- **RMSE:** error metric that penalises large mistakes more heavily
- **R2:** proportion of target variation explained by the predictions

Lower MAE and RMSE are better. R2 should be considered alongside the absolute-error metrics.

## Running the project

1. Open the project folder in VS Code.
2. Select a Python kernel with Jupyter, pandas, NumPy, scikit-learn, matplotlib, and seaborn installed.
3. Open the main notebook.
4. Run the cells from top to bottom.

The CSV file must remain in the same folder as the notebook unless the data-loading path is changed.

## Outputs

The notebook produces:

- cleaned modeling data;
- validation and test comparison tables;
- tuned-model results;
- country and year-level error analysis;
- a future-feature availability audit; and
- next-year country-level forecasts.

## Limitations

- Forecast quality depends on the completeness and timing of OWID updates.
- Multi-year forecasts require future economic, population, and energy inputs or additional forecasts for those variables.
- A strong aggregate score can hide poor performance for individual countries.
- Forecasts are planning estimates, not exact future emissions values.

## Conclusion
- Ridge remains the strongest feature-based model
- ARIMA dominates feature forecasting
- Forecasts are decision support, not certainty

See [`CRISP-DM.md`](CRISP-DM.md) for the fuller business-understanding, data-preparation, modeling, deployment, and maintenance documentation.
