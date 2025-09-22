
---

## 🚀 Workflow Overview

1. **Data Acquisition**
   - Downloads 3 months of TSLA daily price data using `yfinance`.

2. **Feature Engineering**
   - Computes:
     - `HL_PCT` = (High - Close) / Close × 100
     - `PCT_change` = (Close - Open) / Open × 100
   - Keeps `Close`, `HL_PCT`, `PCT_change`, `Volume` for modeling.

3. **Data Scaling**
   - Applies `MinMaxScaler` to features for normalized LSTM input.

4. **Sequence Generation**
   - Transforms time series into:
     - Sequences of length `look_back=25`
     - Forecasts for the next `forecast_out=14` days

5. **LSTM Model Construction**
   - Custom LSTM network with:
     - Input dim: number of features (`input_size`)
     - Hidden size: 50
     - Output: 14-day price forecast

6. **Training**
   - Loss: `MSELoss`
   - Optimizer: `Adam`
   - Epochs: 20

7. **Forecasting**
   - Feeds last available sequence to model for the next 14 predicted close prices
   - Inverse transforms forecasts to real price scale

8. **Visualization**
   - Plots both historical and forecasted prices for visualization.

## 📝 Notes

- Modify `ticker`, `look_back`, and `forecast_out` as needed.
- Works with any stock ticker available on Yahoo Finance.
- Can be adapted to other LSTM/sequence models.

---

## ⚠️ Disclaimer

This project is for **educational purposes**


