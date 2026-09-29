import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score

from src.backtest import run_backtest
from src.data import download_data
from src.features import FEATURES, add_targets, create_features
from src.metrics import drawdown_series, max_drawdown, number_of_trades, sharpe_ratio, total_return
from src.models.lstm import LSTMClassifier
from src.sequences import create_sequences
from src.signals import probabilities_to_signals

ROOT = Path(__file__).resolve().parent


def split_dataframe(df, train_fraction, validation_fraction):
    train_size = int(len(df) * train_fraction)
    val_size = int(len(df) * validation_fraction)
    return (
        df.iloc[:train_size].copy(),
        df.iloc[train_size : train_size + val_size].copy(),
        df.iloc[train_size + val_size :].copy(),
    )


def strategy_metrics(backtest, initial_capital):
    returns = backtest["Strategy_Return_Net"]
    return {
        "Total Return": total_return(backtest["Portfolio"], initial_capital),
        "Sharpe Ratio": sharpe_ratio(returns),
        "Max Drawdown": max_drawdown(returns),
        "Trades": number_of_trades(backtest),
    }


def main():
    with open(ROOT / "configs" / "config.json") as f:
        cfg = json.load(f)

    required = ["lstm_classifier.pth", "logistic_model.pkl", "scaler.pkl"]
    missing = [name for name in required if not (ROOT / "models" / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing {missing}. Run `python train.py` first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = add_targets(create_features(download_data(cfg["ticker"], cfg["start_date"], cfg["end_date"])))
    _, _, test_df = split_dataframe(df, cfg["train_fraction"], cfg["validation_fraction"])

    scaler = joblib.load(ROOT / "models" / "scaler.pkl")
    logistic = joblib.load(ROOT / "models" / "logistic_model.pkl")
    X_test = scaler.transform(test_df[FEATURES].values)
    y_test_return = test_df["Target"].values
    y_test_direction = test_df["Direction_Target"].values

    seq = cfg["sequence_length"]
    X_test_seq, y_test_return_seq = create_sequences(X_test, y_test_return, seq)
    _, y_test_direction_seq = create_sequences(X_test, y_test_direction, seq)
    test_dates = test_df.index[seq - 1 :]

    classifier = LSTMClassifier(len(FEATURES), cfg["hidden_size"], cfg["num_layers"], cfg["dropout"]).to(device)
    classifier.load_state_dict(torch.load(ROOT / "models" / "lstm_classifier.pth", map_location=device))
    classifier.eval()
    with torch.no_grad():
        logits = classifier(torch.tensor(X_test_seq, dtype=torch.float32, device=device))
        lstm_probs = torch.sigmoid(logits).cpu().numpy().ravel()
    lstm_preds = (lstm_probs >= 0.5).astype(int)

    logistic_probs = logistic.predict_proba(X_test)[:, 1][seq - 1 :]
    logistic_preds = (logistic_probs >= 0.5).astype(int)

    lstm_signals = probabilities_to_signals(lstm_probs, cfg["long_threshold"], cfg["short_threshold"])
    logistic_signals = probabilities_to_signals(logistic_probs, cfg["long_threshold"], cfg["short_threshold"])

    lstm_bt = run_backtest(y_test_return_seq, lstm_signals, cfg["initial_capital"], cfg["transaction_cost"], test_dates)
    logistic_bt = run_backtest(y_test_return_seq, logistic_signals, cfg["initial_capital"], cfg["transaction_cost"], test_dates)

    accuracy = pd.DataFrame({
        "Model": ["Classification LSTM", "Logistic Regression", "Always-Up"],
        "Accuracy": [
            accuracy_score(y_test_direction_seq, lstm_preds),
            accuracy_score(y_test_direction_seq, logistic_preds),
            np.mean(y_test_direction_seq == 1),
        ],
    })

    lstm_metrics = strategy_metrics(lstm_bt, cfg["initial_capital"])
    logistic_metrics = strategy_metrics(logistic_bt, cfg["initial_capital"])
    buy_hold_returns = lstm_bt["Actual_Return"]
    buy_hold_metrics = {
        "Total Return": total_return(lstm_bt["Buy_Hold"], cfg["initial_capital"]),
        "Sharpe Ratio": sharpe_ratio(buy_hold_returns),
        "Max Drawdown": max_drawdown(buy_hold_returns),
        "Trades": 1,
    }
    comparison = pd.DataFrame([
        {"Strategy": "Classification LSTM", **lstm_metrics},
        {"Strategy": "Logistic Regression", **logistic_metrics},
        {"Strategy": "SPY Buy & Hold", **buy_hold_metrics},
    ])

    outputs = ROOT / "outputs"
    outputs.mkdir(exist_ok=True)
    accuracy.to_csv(outputs / "accuracy_comparison.csv", index=False)
    comparison.to_csv(outputs / "strategy_comparison.csv", index=False)
    print("\nPrediction comparison\n", accuracy.to_string(index=False))
    print("\nStrategy comparison\n", comparison.to_string(index=False))

    plt.figure(figsize=(14, 6))
    plt.plot(test_dates, lstm_bt["Portfolio"], label="Classification LSTM")
    plt.plot(test_dates, logistic_bt["Portfolio"], label="Logistic Regression")
    plt.plot(test_dates, lstm_bt["Buy_Hold"], label="SPY Buy & Hold")
    plt.title("Trading Strategy Performance")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outputs / "strategy_performance.png", dpi=160)
    plt.close()

    plt.figure(figsize=(14, 6))
    plt.plot(test_dates, drawdown_series(lstm_bt["Portfolio"]), label="Classification LSTM")
    plt.plot(test_dates, drawdown_series(logistic_bt["Portfolio"]), label="Logistic Regression")
    plt.plot(test_dates, drawdown_series(lstm_bt["Buy_Hold"]), label="SPY Buy & Hold")
    plt.title("Strategy Drawdowns")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outputs / "drawdowns.png", dpi=160)
    plt.close()


if __name__ == "__main__":
    main()
