from sklearn.base import ClassifierMixin
import numpy as np
from scipy.stats import mode


class MostFrequentClassifier(ClassifierMixin):
    def fit(self, X=None, y=None):
        """
        Parameters
        ----------
        X : array like, shape = (n_samples, n_features)
            Training data features (ignored in this classifier)
        y : array like, shape = (n_samples,)
            Training data targets
        """
        self.most_frequent_ = mode(y).mode[0]
        return None

    def predict(self, X=None):
        """
        Parameters
        ----------
        X : array like, shape = (n_samples, n_features)
            Data to predict (features are ignored)
        """
        return self.most_frequent_
