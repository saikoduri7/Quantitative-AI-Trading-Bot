import json
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.data import download_data
from src.features import FEATURES, add_targets, create_features
from src.models.baseline import build_logistic_regression
from src.models.lstm import LSTMClassifier, LSTMModel
from src.sequences import create_sequences

ROOT = Path(__file__).resolve().parent


def split_dataframe(df, train_fraction, validation_fraction):
    train_size = int(len(df) * train_fraction)
    val_size = int(len(df) * validation_fraction)
    train_df = df.iloc[:train_size].copy()
    val_df = df.iloc[train_size : train_size + val_size].copy()
    test_df = df.iloc[train_size + val_size :].copy()
    return train_df, val_df, test_df


def make_loader(X, y, batch_size):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    return DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=batch_size, shuffle=False)


def train_with_early_stopping(model, train_loader, val_loader, criterion, optimizer, device, epochs, patience):
    best_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                val_loss += criterion(model(X_batch), y_batch).item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        print(f"Epoch {epoch + 1:02d} | train={train_loss:.6f} | val={val_loss:.6f}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping")
                break

    model.load_state_dict(best_state)
    return model.to(device), best_loss


def main():
    with open(ROOT / "configs" / "config.json") as f:
        cfg = json.load(f)

    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    df = download_data(cfg["ticker"], cfg["start_date"], cfg["end_date"])
    df = add_targets(create_features(df))
    train_df, val_df, _ = split_dataframe(df, cfg["train_fraction"], cfg["validation_fraction"])

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[FEATURES].values)
    X_val = scaler.transform(val_df[FEATURES].values)
    y_train_return = train_df["Target"].values
    y_val_return = val_df["Target"].values
    y_train_direction = train_df["Direction_Target"].values
    y_val_direction = val_df["Direction_Target"].values

    seq = cfg["sequence_length"]
    X_train_seq, y_train_return_seq = create_sequences(X_train, y_train_return, seq)
    X_val_seq, y_val_return_seq = create_sequences(X_val, y_val_return, seq)
    _, y_train_direction_seq = create_sequences(X_train, y_train_direction, seq)
    _, y_val_direction_seq = create_sequences(X_val, y_val_direction, seq)

    regression = LSTMModel(len(FEATURES), cfg["hidden_size"], cfg["num_layers"], cfg["dropout"]).to(device)
    regression, reg_loss = train_with_early_stopping(
        regression,
        make_loader(X_train_seq, y_train_return_seq, cfg["batch_size"]),
        make_loader(X_val_seq, y_val_return_seq, cfg["batch_size"]),
        nn.MSELoss(),
        torch.optim.Adam(regression.parameters(), lr=cfg["learning_rate"]),
        device, cfg["epochs"], cfg["patience"],
    )

    classifier = LSTMClassifier(len(FEATURES), cfg["hidden_size"], cfg["num_layers"], cfg["dropout"]).to(device)
    classifier, cls_loss = train_with_early_stopping(
        classifier,
        make_loader(X_train_seq, y_train_direction_seq, cfg["batch_size"]),
        make_loader(X_val_seq, y_val_direction_seq, cfg["batch_size"]),
        nn.BCEWithLogitsLoss(),
        torch.optim.Adam(classifier.parameters(), lr=cfg["learning_rate"]),
        device, cfg["epochs"], cfg["patience"],
    )

    logistic = build_logistic_regression()
    logistic.fit(X_train, y_train_direction)

    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    torch.save(regression.state_dict(), models_dir / "lstm_regression.pth")
    torch.save(classifier.state_dict(), models_dir / "lstm_classifier.pth")
    joblib.dump(logistic, models_dir / "logistic_model.pkl")
    joblib.dump(scaler, models_dir / "scaler.pkl")

    print(f"Saved models. Best regression val loss: {reg_loss:.6f}; classifier val loss: {cls_loss:.6f}")


if __name__ == "__main__":
    main()
