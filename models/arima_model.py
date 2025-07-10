import numpy as np
import pandas as pd
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


class ARIMAForecaster:
    def __init__(self, seasonal=True, m=1, max_order=5, suppress_warnings=True):
        """
        Initializes the ARIMA forecaster.

        Args:
            seasonal (bool): Whether to fit a seasonal ARIMA.
            m (int): Seasonal periodicity (e.g. 12 for monthly data with yearly seasonality).
            max_order (int): Max order to try for AR, I, MA.
            suppress_warnings (bool): Suppress convergence warnings.
        """
        self.seasonal = seasonal
        self.m = m
        self.max_order = max_order
        self.suppress_warnings = suppress_warnings
        self.model = None

    def fit(self, y_train: pd.Series):
        """Fits the ARIMA model on the training data."""
        self.model = auto_arima(
            y_train,
            seasonal=self.seasonal,
            m=self.m,
            max_order=self.max_order,
            stepwise=True,
            suppress_warnings=self.suppress_warnings,
            error_action='ignore'
        )

    def predict(self, steps: int) -> np.ndarray:
        """Forecasts the next `steps` points."""
        return self.model.predict(n_periods=steps)

    def evaluate(self, y_train: pd.Series, y_test: pd.Series) -> dict:
        """Fits model on y_train and evaluates on y_test."""
        self.fit(y_train)
        preds = self.predict(len(y_test))
        return {
            'MAE': mean_absolute_error(y_test, preds),
            'RMSE': root_mean_squared_error(y_test, preds),
            'y_test': y_test.values,
            'y_pred': preds
        }
