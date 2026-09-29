import numpy as np


def probabilities_to_signals(probabilities, long_threshold=0.55, short_threshold=0.45):
    """Map P(up) to short (-1), cash (0), or long (+1)."""
    probabilities = np.asarray(probabilities)
    signals = np.zeros(len(probabilities), dtype=float)
    signals[probabilities > long_threshold] = 1
    signals[probabilities < short_threshold] = -1
    return signals


def returns_to_signals(predicted_returns, threshold=0.001):
    """Map predicted returns to short/cash/long positions."""
    predicted_returns = np.asarray(predicted_returns)
    signals = np.zeros(len(predicted_returns), dtype=float)
    signals[predicted_returns > threshold] = 1
    signals[predicted_returns < -threshold] = -1
    return signals
