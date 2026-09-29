# Quantitative AI Trading System

An end-to-end machine learning pipeline for predicting next-day SPY market behavior using LSTM neural networks and statistical baselines. The system uses technical features from historical market data, generates model-driven trading signals, and evaluates them through transaction-cost-aware backtesting.

## Pipeline

```
SPY OHLCV Data
      ↓
Feature Engineering
      ↓
Chronological Split & Scaling
      ↓
60-Day Sequences
      ↓
Model Training
      ↓
Trading Signals
      ↓
Backtesting
      ↓
Performance Evaluation
```

## 1. Market Data

Historical daily SPY OHLCV data is collected using `yfinance`, including open, high, low, close, and volume.

Daily return is defined as:

$$
R_t = \frac{P_t-P_{t-1}}{P_{t-1}}
$$

where $P_t$ is the closing price on day $t$.

## 2. Feature Engineering

Ten features are generated from the raw market data:

| Feature | Description |
|---|---|
| **Return** | Daily percentage change in closing price |
| **SMA 5** | 5-day average closing price; short-term trend |
| **SMA 20** | 20-day average closing price; medium-term trend |
| **SMA 50** | 50-day average closing price; longer-term trend |
| **Momentum 5** | Price change over the previous 5 trading days |
| **Momentum 20** | Price change over the previous 20 trading days |
| **Volatility 10** | 10-day rolling standard deviation of returns |
| **Volatility 20** | 20-day rolling standard deviation of returns |
| **Volume Change** | Daily percentage change in trading volume |
| **High-Low Range** | Intraday price range relative to closing price |

The primary feature formulas are:

$$
SMA_{n,t} = \frac{1}{n}\sum_{i=0}^{n-1}P_{t-i}
$$

$$
Momentum_{n,t} = \frac{P_t}{P_{t-n}}-1
$$

$$
Volatility_{n,t} = Std(R_t,\ldots,R_{t-n+1})
$$

$$
VolumeChange_t = \frac{V_t-V_{t-1}}{V_{t-1}}
$$

$$
HL_t = \frac{High_t-Low_t}{Close_t}
$$

## 3. Prediction Targets

Two LSTM formulations are evaluated.

**Return Regression** predicts the numerical next-day return:

$$
y_t = R_{t+1}
$$

**Direction Classification** predicts whether the next-day return is positive:

$$
y_t =
\begin{cases}
1, & R_{t+1}>0 \\
0, & R_{t+1}\leq0
\end{cases}
$$

This allows comparison between predicting exact return magnitude and directly predicting market direction.

## 4. Data Preparation

Data is split chronologically into:

- 70% training
- 15% validation
- 15% testing

Features are standardized using:

$$
z=\frac{x-\mu}{\sigma}
$$

The scaler is fit **only on training data** and then applied to validation and test data to prevent future information leakage.

## 5. LSTM Training Observations

Each trading day is represented by a 10-feature vector:

$$
x_t =
\begin{bmatrix}
R_t &
SMA_{5,t} &
SMA_{20,t} &
SMA_{50,t} &
MOM_{5,t} &
MOM_{20,t} &
\sigma_{10,t} &
\sigma_{20,t} &
\Delta V_t &
HL_t
\end{bmatrix}
$$

The LSTM receives the previous **60 trading days** for each prediction. Therefore, one training observation is:

$$
X_t =
\begin{bmatrix}
x_{t-59}\\
x_{t-58}\\
\vdots\\
x_t
\end{bmatrix}
\in \mathbb{R}^{60\times10}
$$

The full model input has shape:

$$
N\times60\times10
$$

where $N$ is the number of sequences. Each $60\times10$ matrix produces one prediction for day $t+1$.

## 6. LSTM Architecture

Both LSTMs use:

```
60 Days × 10 Features
        ↓
2-Layer LSTM
64 Hidden Units
        ↓
Linear (64 → 32)
        ↓
ReLU
        ↓
Linear (32 → 1)
        ↓
Prediction
```

The **regression LSTM** minimizes mean squared error:

$$
MSE=\frac{1}{N}\sum_{i=1}^{N}(y_i-\hat y_i)^2
$$

The **classification LSTM** is trained with binary cross-entropy. Its output logit is converted into an upward-movement probability using:

$$
p_t=\frac{1}{1+e^{-z_t}}
$$

Early stopping based on validation loss is used to limit overfitting.

## 7. Logistic Regression Baseline

A Logistic Regression classifier provides a simpler benchmark.

Unlike the LSTM's $60\times10$ sequence, Logistic Regression receives only the current day's feature vector:

$$
x_t\in\mathbb{R}^{10}
$$

This tests whether the LSTM's temporal complexity provides useful predictive information beyond a simple linear classifier.

## 8. Trading Signals

The classification model outputs a probability $p_t$ that SPY will have a positive return on the next trading day.

Predictions are converted into three possible positions using two confidence thresholds:

- **Long (+1):** $p_t > 0.52$
- **Cash (0):** $0.48 \leq p_t \leq 0.52$
- **Short (-1):** $p_t < 0.48$

The thresholds were selected using validation data rather than test performance.

For example, a predicted probability of 0.56 generates a long position, while a probability of 0.44 generates a short position.

## 9. Backtesting

Strategy return is calculated as:

$$
R_t^{strategy}=S_tR_t
$$

Position changes incur transaction costs:

$$
T_t=|S_t-S_{t-1}|
$$

$$
R_t^{net}=R_t^{strategy}-cT_t
$$

Portfolio value then compounds from an initial $10,000:

$$
V_t=V_{t-1}(1+R_t^{net})
$$

## 10. Evaluation

Models are evaluated using both **ML metrics** and **trading metrics**.

Prediction metrics include accuracy, precision, recall, F1 score, and confusion matrices.

Trading performance includes:

**Total Return**

$$
R_{total}=\frac{V_T}{V_0}-1
$$

**Annualized Sharpe Ratio**

$$
Sharpe=\frac{\bar R}{\sigma_R}\sqrt{252}
$$

**Maximum Drawdown**

$$
DD_t=\frac{V_t-\max_{s\leq t}V_s}{\max_{s\leq t}V_s}
$$

## 11. Results & Findings

The experiments compared:

- LSTM return regression
- LSTM direction classification
- Logistic Regression
- Always-up baseline
- SPY buy-and-hold

A key finding was that **prediction accuracy did not necessarily translate into market-timing ability**. However, this model needs to be tested on other stocks to find more prediction findings.

The classification LSTM produced probabilities within a narrow range and remained long throughout the evaluated test period, causing its portfolio to closely track buy-and-hold. Logistic Regression similarly predicted upward movement on nearly every observation.

The experiments demonstrate the importance of comparing financial ML models against simple baselines and evaluating predictions through realistic backtesting rather than accuracy alone.

## 12. Project Structure

```text
quant-ai-trading/
├── notebooks/
│   └── exploration.ipynb
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
├── configs/
│   └── config.json
├── outputs/
├── train.py
├── run_backtest.py
├── requirements.txt
└── README.md
```

## 13. Usage

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the models:

```bash
python train.py
```

Run the backtest:

```bash
python run_backtest.py
```

## Technologies

Python · PyTorch · scikit-learn · pandas · NumPy · Matplotlib · yfinance
