import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


class ARForecaster:
    def __init__(self):
        """
        AR forecaster (random walk model).
        Always forecasts the last observed value forward.
        """
        self.last_value = None

    def fit(self, y_train: pd.Series):
        """Memorize the last observed value."""
        self.last_value = y_train.iloc[-1]

    def predict(self, steps: int) -> np.ndarray:
        """Return a flat forecast using the last value."""
        return np.full(steps, self.last_value)

    def evaluate(self, y_train: pd.Series, y_test: pd.Series) -> dict:
        """Fit and evaluate against test data."""
        self.fit(y_train)
        preds = self.predict(len(y_test))
        return {
            'MAE': mean_absolute_error(y_test, preds),
            'RMSE': root_mean_squared_error(y_test, preds),
            'y_test': y_test.values,
            'y_pred': preds
        }
