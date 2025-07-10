import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


class ProphetForecaster:
    def __init__(self, yearly_seasonality='auto', weekly_seasonality='auto', daily_seasonality=False):
        """
        Prophet forecaster from Meta.
        
        Args:
            yearly_seasonality (str/bool): 'auto', True, or False.
            weekly_seasonality (str/bool): 'auto', True, or False.
            daily_seasonality (bool): Whether to include daily seasonality.
        """
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.model = None
        self.fitted = False

    def fit(self, y_train: pd.Series):
        """Fit the Prophet model on the training data."""
        df = y_train.reset_index()
        df.columns = ['ds', 'y']

        self.model = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality
        )
        self.model.fit(df)
        self.fitted = True

    def predict(self, steps: int) -> np.ndarray:
        """Predict the next `steps` time points."""
        if not self.fitted:
            raise ValueError("Model must be fit before prediction.")
        
        future = self.model.make_future_dataframe(periods=steps)
        forecast = self.model.predict(future)
        return forecast['yhat'].iloc[-steps:].values

    def evaluate(self, y_train: pd.Series, y_test: pd.Series) -> dict:
        """Fit and evaluate the Prophet model."""
        self.fit(y_train)
        preds = self.predict(len(y_test))
        return {
            'MAE': mean_absolute_error(y_test, preds),
            'RMSE': root_mean_squared_error(y_test, preds),
            'y_test': y_test.values,
            'y_pred': preds
        }
