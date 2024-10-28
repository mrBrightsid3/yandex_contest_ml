import numpy as np


class LaplaceDistribution:
    @staticmethod
    def mean_abs_deviation_from_median(x: np.ndarray):
        """
        Args:
        - x: A numpy array of shape (n_objects, n_features) containing the data
          consisting of num_train samples each of dimension D.
        """
        ####
        # Do not change the class outside of this block
        # Your code here
        ####

    def __init__(self, features):
        """
        Args:
            feature: A numpy array of shape (n_objects, n_features). Every column represents all available values for the selected feature.
        """
        ####
        # Do not change the class outside of this block
        if len(features.shape) > 1:
            self.loc = np.array(
                [np.median(features[:, i : i + 1]) for i in range(features.shape[1])]
            )
            self.scale = np.array(
                [
                    np.mean(np.abs(features[:, i : i + 1] - self.loc[i]))
                    for i in range(features.shape[1])
                ]
            )
        else:
            self.loc = np.median(features)  # YOUR CODE HERE
            self.scale = np.mean(np.abs(features - self.loc))  # YOUR CODE HERE
        ####

    def logpdf(self, values):
        """
        Returns logarithm of probability density at every input value.
        Args:
            values: A numpy array of shape (n_objects, n_features). Every column represents all available values for the selected feature.
        """
        ####
        # Do not change the class outside of this block
        return -(np.abs(values - self.loc) / self.scale) + np.log(1 / (2 * self.scale))
        ####

    def pdf(self, values):
        """
        Returns probability density at every input value.
        Args:
            values: A numpy array of shape (n_objects, n_features). Every column represents all available values for the selected feature.
        """
        return np.exp(self.logpdf(values))
