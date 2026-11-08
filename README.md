# Cryptocurrency Analysis: Ethereum price modelling

A short Jupyter notebook that models the daily price of **Ethereum (ETH-USD)** with two approaches:

1. **XGBoost regression**: predicts a day's closing price from that same day's `Open`, `High`, `Low` and `Volume`.
2. **ARIMA(2, 1, 1)**: a univariate time-series model fitted on the closing-price history and used to forecast the last 30 days.

## Dataset

- **Source:** Yahoo Finance daily history for the `ETH-USD` ticker (columns `Date, Open, High, Low, Close, Adj Close, Volume`). The copy used in the notebook starts on 2017-11-09.
- **Not included in the repo.** To get it:
  - download it from the [Yahoo Finance ETH-USD history page](https://finance.yahoo.com/quote/ETH-USD/history/) (set the range to "Max", then download), or
  - use the `yfinance` package: `yfinance.download("ETH-USD", start="2017-11-09").to_csv("ETH-USD.csv")` (check that the column names match the list above).
- Save it as `ETH-USD.csv` next to the notebook, or change `DATA_PATH` in the first cell. On Google Colab with Drive mounted, set `DATA_PATH = '/content/drive/MyDrive/ETH-USD.csv'`.

The original Colab notebook (with the data files the author used) is linked here: <https://colab.research.google.com/drive/1zmP-78Zqvv1JReXpsPDJK9PZUFHw6PJ_?usp=share_link>

## Approach

| Step | XGBoost | ARIMA |
|---|---|---|
| Input | `Open`, `High`, `Low`, `Volume` | `Close` series |
| Split | random 80/20 (`train_test_split`, `random_state=42`) | all but the last 30 days for training, last 30 days for testing |
| Model | `XGBRegressor(colsample_bytree=0.3, learning_rate=0.1, max_depth=5, alpha=10, n_estimators=100)` | `ARIMA(order=(2, 1, 1))` from `statsmodels` |
| Metric | RMSE | MAE |

## Results

| Model | Metric | Value |
|---|---|---|
| XGBoost | RMSE on the 20% hold-out | **105.19 USD** (from the saved notebook output) |
| ARIMA(2,1,1) | MAE over the last 30 days | not available; see below |

**Notes on these numbers**

- The XGBoost model uses same-day `High`/`Low`, which are only known once the day is over, and the split is random rather than chronological. The RMSE therefore measures how well the model reconstructs a day's close from that day's own range. It is **not** a forecast of future prices.
- The ARIMA cell originally printed an MAE of 336.93, but that value was wrong: the code kept only the first forecast step (`forecast(...)[0]`) and compared that single number with all 30 test days. The cell is fixed (see below) and its old output has been cleared. Re-run it with the dataset to get the correct MAE.

## How to run

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# put ETH-USD.csv next to the notebook (see Dataset)
jupyter notebook Crypto_Analysis.ipynb
```

On macOS, XGBoost needs the OpenMP runtime (`brew install libomp`).

## Changes made for reproducibility

- Replaced the hard-coded Colab path `/content/drive/MyDrive/ETH-USD.csv` with a `DATA_PATH` variable.
- Fixed the ARIMA evaluation. `forecast(...)[0]` raises `KeyError` on a date-indexed Series in pandas 3, and it also discarded 29 of the 30 forecast steps. The cell now compares all 30 forecasts against the 30 test days and declares the daily frequency explicitly.
- Added `requirements.txt` and `.gitignore`.

## Repository structure

```
Crypto_Analysis.ipynb   # the analysis notebook
requirements.txt        # Python dependencies
LICENSE                 # MIT
```

## Tech stack

Python, pandas, NumPy, XGBoost, scikit-learn, statsmodels, Jupyter.
