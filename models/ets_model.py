import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


class ETSForecaster:
    def __init__(self, seasonal='add', seasonal_periods=7):
        """
        Exponential Smoothing (ETS) forecaster.
        
        Args:
            seasonal (str): 'add' or 'mul' for additive or multiplicative seasonality.
            seasonal_periods (int): Number of periods in a seasonal cycle (e.g., 7 for weekly).
        """
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.model = None
        self.fitted_model = None

    def fit(self, y_train: pd.Series):
        """Fit ETS model on training data."""
        self.model = ExponentialSmoothing(
            y_train,
            seasonal=self.seasonal,
            seasonal_periods=self.seasonal_periods,
            trend='add',
            initialization_method="estimated"
        )
        self.fitted_model = self.model.fit()

    def predict(self, steps: int) -> np.ndarray:
        """Forecast future values."""
        return self.fitted_model.forecast(steps)

    def evaluate(self, y_train: pd.Series, y_test: pd.Series) -> dict:
        """Fit and evaluate the ETS model."""
        self.fit(y_train)
        preds = self.predict(len(y_test))
        return {
            'MAE': mean_absolute_error(y_test, preds),
            'RMSE': root_mean_squared_error(y_test, preds),
            'y_test': y_test.values,
            'y_pred': preds
        }
