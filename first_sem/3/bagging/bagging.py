import numpy as np


class SimplifiedBaggingRegressor:
    def __init__(self, num_bags, oob=False):
        self.num_bags = num_bags
        self.oob = oob

    def _generate_splits(self, data: np.ndarray):
        self.indices_list = []
        data_length = len(data)
        for bag in range(self.num_bags):
            indeces = np.random.choice(data_length, size=data_length)
            self.indices_list.append(indeces)

    def fit(self, model_constructor, data, target):
        """
        Fit model on every bag.
        Model constructor with no parameters (and with no ()) is passed to this function.

        example:

        bagging_regressor = SimplifiedBaggingRegressor(num_bags=10, oob=True)
        bagging_regressor.fit(LinearRegression, X, y)
        """
        self.data = None
        self.target = None
        self._generate_splits(data)
        assert (
            len(set(list(map(len, self.indices_list)))) == 1
        ), "All bags should be of the same length!"
        assert list(map(len, self.indices_list))[0] == len(
            data
        ), "All bags should contain `len(data)` number of elements!"
        self.models_list = []
        for bag in range(self.num_bags):
            model = model_constructor()
            data_bag, target_bag = (
                data[self.indices_list[bag]],
                target[self.indices_list[bag]],
            )
            self.models_list.append(
                model.fit(data_bag, target_bag)
            )  # store fitted models here
        if self.oob:
            self.data = data
            self.target = target

    def predict(self, data):
        """
        Get average prediction for every object from passed dataset
        """
        all_predictions = []
        for model in self.models_list:
            predicted = model.predict(data)
            all_predictions.append(predicted)
        all_predictions = np.vstack(all_predictions)
        return all_predictions.sum(axis=0) / all_predictions.shape[0]

    def _get_oob_predictions_from_every_model(self):
        """
        Generates list of lists, where list i contains predictions for self.data[i] object
        from all models, which have not seen this object during training phase
        """
        list_of_predictions_lists = [[] for _ in range(len(self.data))]
        for i in range(len(list_of_predictions_lists)):
            for j in range(len(self.indices_list)):
                if i not in self.indices_list[j]:
                    list_of_predictions_lists[i].append(
                        self.models_list[j].predict(self.data[i].reshape(1, -1))
                    )

        self.list_of_predictions_lists = np.array(
            list_of_predictions_lists, dtype=object
        )

    def _get_averaged_oob_predictions(self):
        """
        Compute average prediction for every object from training set.
        If object has been used in all bags on training phase, return None instead of prediction
        """
        self._get_oob_predictions_from_every_model()
        self.oob_predictions = np.array(
            [
                None if len(pred) == 0 else np.mean(pred)
                for pred in self.list_of_predictions_lists
            ]
        )

    def OOB_score(self):
        """
        Compute mean square error for all objects, which have at least one prediction
        """
        self._get_averaged_oob_predictions()
        valid_indexes = [
            i
            for i in range(len(self.oob_predictions))
            if self.oob_predictions[i] is not None
        ]
        return np.mean(
            (self.target[valid_indexes] - self.oob_predictions[valid_indexes]) ** 2
        )
