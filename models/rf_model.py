import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


class RFForecaster:
    def __init__(self, n_lags=14, n_estimators=100, random_state=42):
        """
        Random Forest forecaster using lagged features.

        Args:
            n_lags (int): Number of past time steps to use as features.
            n_estimators (int): Number of trees in the forest.
            random_state (int): Random seed for reproducibility.
        """
        self.n_lags = n_lags
        self.model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        self.last_window = None

    def _make_supervised(self, series: pd.Series) -> tuple:
        """Convert time series to supervised learning format."""
        X, y = [], []
        values = series.values
        for i in range(self.n_lags, len(values)):
            X.append(values[i - self.n_lags:i])
            y.append(values[i])
        return np.array(X), np.array(y)

    def fit(self, y_train: pd.Series):
        """Fit the model on training data."""
        X, y = self._make_supervised(y_train)
        self.model.fit(X, y)
        self.last_window = y_train.values[-self.n_lags:]

    def predict(self, steps: int) -> np.ndarray:
        """Predict multiple steps ahead using recursive strategy."""
        if self.last_window is None:
            raise ValueError("Model must be fit before prediction.")

        window = self.last_window.copy()
        preds = []

        for _ in range(steps):
            x_input = window[-self.n_lags:]
            pred = self.model.predict([x_input])[0]
            preds.append(pred)
            window = np.append(window, pred)

        return np.array(preds)

    def evaluate(self, y_train: pd.Series, y_test: pd.Series) -> dict:
        """Fit and evaluate the Random Forest model."""
        self.fit(y_train)
        preds = self.predict(len(y_test))
        return {
            'MAE': mean_absolute_error(y_test, preds),
            'RMSE': root_mean_squared_error(y_test, preds),
            'y_test': y_test.values,
            'y_pred': preds
        }
