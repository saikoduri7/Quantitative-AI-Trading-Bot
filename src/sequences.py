import numpy as np


def create_sequences(X, y, sequence_length: int = 60):
    """Convert tabular time-series data into rolling LSTM sequences."""
    X_sequences = []
    y_sequences = []

    for i in range(sequence_length - 1, len(X)):
        X_sequences.append(X[i - sequence_length + 1 : i + 1])
        y_sequences.append(y[i])

    return np.asarray(X_sequences), np.asarray(y_sequences)
