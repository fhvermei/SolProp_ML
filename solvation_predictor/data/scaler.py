import numpy as np
from solvation_predictor.data.data import DatapointList


class Scaler:
    """
    Class used to scale targets and features, either standard scaling or minmax scaling. It also allows for inverse
    operation of the scaling. The scaled targets and features are saved in the Datapoint object.
    """

    def __init__(self, data: DatapointList = None, scale_features: bool = True, mean=0.0, mean_f=0.0, std=0.0, std_f=0.0):
        self.scale_features = scale_features
        if data is not None:
            targets = data.get_targets()
            self.mean = 0
            self.std = 0
            self.mean = np.nanmean(targets, axis=0)
            self.std = np.nanstd(targets, axis=0)
            if self.scale_features:
                features = data.get_features()
                self.mean_features = 0
                self.std_features = 0
                self.mean_features = np.mean(features, axis=0)
                self.std_features = np.std(features, axis=0)
            else:
                self.mean_features = 0
                self.std_features = 0
            self.type = "None"
        else:
            self.mean = mean
            self.std = std
            self.mean_features = mean_f
            self.std_features = std_f

    def transform_standard(self, data):
        targets = data.get_targets()
        scaled_targets = (targets-self.mean)/self.std
        data.set_scaled_targets(scaled_targets)
        if self.scale_features:
            features = data.get_features()
            scaled_features = (features-self.mean_features)/self.std_features
            data.set_scaled_features(scaled_features)
        else:
            data.set_scaled_features(data.get_features())
        self.type = "standard"

    def inverse_transform_standard(self, preds):
        scaled_predictions = preds
        predictions = (self.std*scaled_predictions+self.mean)
        return predictions

    def inverse_transform(self, data):
        self.type="standard"
        if self.type == "standard":
            return self.inverse_transform_standard(data)
        elif self.type == "None":
            print("no scaler transformation")
        else:
            raise ValueError(f'Type of scaler transformation"{self.type}" not supported.')

