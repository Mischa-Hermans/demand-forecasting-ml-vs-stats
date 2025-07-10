import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


class ARForecaster:
    def __init__(self, max_lag: int = 20, ic: str = 'aic'):
        """
        AutoRegressive (AR) model using statsmodels.

        Args:
            max_lag (int): Maximum number of lags to consider.
            ic (str): Information criterion for lag selection ('aic' or 'bic').
        """
        self.max_lag = max_lag
        self.ic = ic
        self.model = None
        self.fitted_model = None

    def fit(self, y_train: pd.Series):
        """Select optimal lag and fit AR model on training data."""
        selected_order = ar_select_order(y_train, maxlag=self.max_lag, ic=self.ic).ar_lags
        self.model = AutoReg(y_train, lags=selected_order, old_names=False)
        self.fitted_model = self.model.fit()

    def predict(self, steps: int) -> np.ndarray:
        """Forecast the next `steps` values using fitted AR model."""
        return self.fitted_model.predict(start=len(self.fitted_model.model.endog), 
                                         end=len(self.fitted_model.model.endog) + steps - 1)

    def evaluate(self, y_train: pd.Series, y_test: pd.Series) -> dict:
        """Fit model and evaluate on test data."""
        self.fit(y_train)
        preds = self.predict(len(y_test))
        return {
            'MAE': mean_absolute_error(y_test, preds),
            'RMSE': root_mean_squared_error(y_test, preds),
            'y_test': y_test.values,
            'y_pred': preds
        }
