from sklearn.base import RegressorMixin
import pandas as pd
import numpy as np


import pandas as pd
import numpy as np
from sklearn.base import RegressorMixin, BaseEstimator


import pandas as pd
import numpy as np
from sklearn.base import RegressorMixin, BaseEstimator


class CityMeanRegressor(BaseEstimator, RegressorMixin):
    def fit(self, X, y):
        # Преобразуем X в DataFrame, если это ndarray
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=["city"])

        if not isinstance(X, pd.DataFrame):
            raise ValueError("X должен быть pandas DataFrame или numpy.ndarray")

        if "city" not in X.columns:
            raise ValueError("В X должна быть колонка 'city'")

        # Вычисляем среднее значение целевой переменной для каждого города
        self.city_means_ = pd.Series(y).groupby(X["city"]).mean()
        self.global_mean_ = np.mean(y)  # На случай, если город неизвестен

        return self

    def predict(self, X):
        # Преобразуем X в DataFrame, если это ndarray
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=["city"])

        if not isinstance(X, pd.DataFrame):
            raise ValueError("X должен быть pandas DataFrame или numpy.ndarray")

        if "city" not in X.columns:
            raise ValueError("В X должна быть колонка 'city'")

        # Прогнозируем среднее значение по городу, если город известен
        predictions = X["city"].map(self.city_means_).fillna(self.global_mean_)
        return predictions.values
