import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


class LSTMForecaster:
    def __init__(self, n_lags=14, n_units=50, epochs=10, batch_size=32):
        """
        LSTM forecaster using lagged time windows.

        Args:
            n_lags (int): Number of past time steps to use as input features.
            n_units (int): Number of LSTM units.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
        """
        self.n_lags = n_lags
        self.n_units = n_units
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.last_window = None

    def _make_supervised(self, series: pd.Series):
        """Transform time series into (X, y) for supervised learning."""
        values = series.values
        X, y = [], []
        for i in range(self.n_lags, len(values)):
            X.append(values[i - self.n_lags:i])
            y.append(values[i])
        X = np.array(X)
        y = np.array(y)
        return X[..., np.newaxis], y  # shape: (samples, timesteps, features)

    def _build_model(self):
        """Construct the LSTM model."""
        model = Sequential([
            LSTM(self.n_units, input_shape=(self.n_lags, 1)),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def fit(self, y_train: pd.Series):
        """Fit LSTM on training data."""
        X, y = self._make_supervised(y_train)
        self.model = self._build_model()
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        self.last_window = y_train.values[-self.n_lags:]

    def predict(self, steps: int) -> np.ndarray:
        """Generate multi-step forecasts recursively."""
        if self.model is None:
            raise ValueError("Model must be fit before prediction.")

        window = self.last_window.copy()
        preds = []

        for _ in range(steps):
            x_input = window[-self.n_lags:].reshape(1, self.n_lags, 1)
            pred = self.model.predict(x_input, verbose=0)[0, 0]
            preds.append(pred)
            window = np.append(window, pred)

        return np.array(preds)

    def evaluate(self, y_train: pd.Series, y_test: pd.Series) -> dict:
        """Fit model and evaluate."""
        self.fit(y_train)
        preds = self.predict(len(y_test))
        return {
            'MAE': mean_absolute_error(y_test, preds),
            'RMSE': root_mean_squared_error(y_test, preds),
            'y_test': y_test.values,
            'y_pred': preds
        }
