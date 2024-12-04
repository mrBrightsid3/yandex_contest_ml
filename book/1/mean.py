import numpy as np
import pandas as pd

from sklearn.base import RegressorMixin


class MeanRegressor(RegressorMixin):
    # Predicts the mean of y_train
    def fit(self, X=None, y=None):
        """
        Parameters
        ----------
        X : array like, shape = (n_samples, n_features)
        Training data features
        y : array like, shape = (_samples,)
        Training data targets
        """
        self.res = y.mean()
        return None

    def predict(self, X=None):
        """
        Parameters
        ----------
        X : array like, shape = (n_samples, n_features)
        Data to predict
        """
        return self.res
