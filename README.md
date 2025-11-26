# kaggle-learnings

## Overview
This repository contains Kaggle learning projects and datasets, including time series analysis of accidental deaths in the USA.

## Time Series Analysis Workflow

### 1. Exploratory Data Analysis (EDA)
- Visualize time series data to identify trends, seasonality, and outliers.
- Check for missing values and handle gaps.
- Calculate summary statistics: mean, median, min, max, standard deviation.
- Perform autocorrelation analysis to detect dependencies.
- Compute rolling statistics (moving averages, rolling std).
- Decompose the series into trend, seasonality, and residuals using `statsmodels`.
- Check stationarity using statistical tests (e.g., Augmented Dickey-Fuller).
- Group by time units (month, year) for seasonal analysis.
- Correlation analysis with other variables or lagged values.

### 2. Modeling Approaches
- **Linear Regression:** For simple trend analysis (does not capture seasonality).
- **Polynomial Regression:** For non-linear trends (limited for seasonal data).
- **ARIMA/SARIMA:** For time series forecasting, capturing trend and seasonality.
- **Exponential Smoothing (ETS):** For seasonal data.
- **Prophet:** For robust forecasting with seasonality and holidays.

### 3. Model Evaluation
- Use R² score to evaluate regression and time series models.
- Visualize actual vs. predicted/forecasted values.

## Notebooks
- `accidental-deaths-in-usa-monthly/accidental-deaths-in-usa-monthly.ipynb`: Time series analysis and forecasting notebook.

## Datasets
- Various datasets for practice and analysis, including accidental deaths, students performance, and more.