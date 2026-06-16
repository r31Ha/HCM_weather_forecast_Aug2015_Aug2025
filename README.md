# HCMC Weather Forecasting Model

Predicting tomorrow's average temperature in Ho Chi Minh City using historical weather data.

## Results
| Model | MAE |
|---|---|
| Linear Regression | 0.64°C |
| XGBoost | 0.65°C |
| Random Forest | 0.66°C |

## Features Used
- Lag temperatures (1, 2, 3, 7, 14, 365 days)
- Rolling averages (7-day, 30-day)
- Rolling std deviation (7-day)
- Month + day of year

## Data
Daily average temperature from Tan Son Hoa station (HCMC), sourced from NOAA.

## Tools
Python, Pandas, Scikit-learn, XGBoost, Matplotlib

## Key Insight
Linear Regression matched or beat more complex models — the temp-to-temp
relationship in HCMC is essentially linear. Remaining error (~0.64°C) comes
from unpredictable weather events that temperature history alone can't capture.