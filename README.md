# Quantitative AI Trading System

A machine-learning trading research project that predicts next-day SPY behavior and evaluates model-driven strategies under transaction costs. The project compares an LSTM return regressor, an LSTM direction classifier, logistic regression, an always-up baseline, and SPY buy-and-hold.

## Pipeline

`SPY OHLCV → feature engineering → chronological split → train-only scaling → 60-day sequences → model training → signal generation → transaction-cost backtest → risk evaluation`

The feature set contains daily return, 5/20/50-day moving averages, 5/20-day momentum, 10/20-day rolling volatility, volume change, and normalized intraday high-low range.

## Models

- **LSTM regression:** predicts the next-day return using MSE loss.
- **LSTM classification:** predicts the probability that the next-day return is positive using binary cross-entropy with logits.
- **Logistic regression:** simple non-sequential classification baseline using the same engineered features.
- **Always-Up / Buy & Hold:** sanity-check baselines that expose whether a model is learning genuine timing information or mostly the market's upward bias.

Both LSTMs use two recurrent layers with 64 hidden units, dropout, a 32-unit dense layer, and early stopping on chronological validation loss.

## Notebook results

The committed exploration notebook produced the following test-period results:

| Model | Direction Accuracy |
|---|---:|
| Classification LSTM | 56.94% |
| Logistic Regression | 57.12% |
| Always-Up Baseline | 56.94% |

| Strategy | Total Return | Sharpe | Max Drawdown | Trades |
|---|---:|---:|---:|---:|
| Classification LSTM | 49.18% | 1.18 | -18.76% | 1 |
| Logistic Regression | 35.58% | 0.92 | -25.05% | 11 |
| SPY Buy & Hold | 49.25% | 1.18 | -18.76% | 1 |

The classification LSTM assigned test probabilities in a narrow 53.33%–55.05% range and generated a long signal on all 562 aligned test days. The earlier regression LSTM also generated a long signal on all 562 test days and matched the 56.94% always-up directional baseline. These diagnostics show why predictive accuracy alone is not evidence of a useful trading signal.

## Repository structure

```text
quant-ai-trading/
├── notebooks/exploration.ipynb
├── src/
│   ├── data.py
│   ├── features.py
│   ├── sequences.py
│   ├── signals.py
│   ├── backtest.py
│   ├── metrics.py
│   └── models/
│       ├── lstm.py
│       └── baseline.py
├── models/
├── configs/config.json
├── outputs/
├── train.py
├── run_backtest.py
└── requirements.txt
```

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python train.py
python run_backtest.py
```

`train.py` downloads the data, recreates the features and chronological split, fits the scaler on training data only, trains both LSTMs with early stopping, fits logistic regression, and writes model artifacts to `models/`.

`run_backtest.py` loads the trained artifacts, evaluates the classification models on aligned dates, runs the cost-aware backtests, and writes comparison CSVs and plots to `outputs/`.

## Methodology notes

The notebook uses a 70%/15%/15% chronological train/validation/test split and a 0.05% transaction cost per unit of position turnover. The classification trading thresholds (long above 0.52, short below 0.48) were selected from a small predetermined set using validation return rather than test return.

The test period was inspected repeatedly during development, so its results should be treated as exploratory rather than a pristine final estimate of future performance. A stronger next version would freeze the pipeline and evaluate once on a new holdout period or use walk-forward evaluation.

## Technologies

Python, PyTorch, scikit-learn, pandas, NumPy, yfinance, Matplotlib, joblib.
