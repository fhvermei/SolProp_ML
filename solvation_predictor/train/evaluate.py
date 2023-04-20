import math
from typing import List

import numpy as np
import torch
from sklearn.metrics import mean_squared_error
from torch import nn
from tqdm import trange

from solvation_predictor.data.scaler import Scaler
from solvation_predictor.data.data import DatapointList


def evaluate(model: nn.Module,
             data: DatapointList,
             metric_func: str,
             scaler: Scaler = None
             ):
    """
    Evaluates an ensemble of models on a dataset.

    :param model: A model.
    :param data: A MoleculeDataset.
    :param num_tasks: Number of tasks.
    :param metric_func: Metric function which takes in a list of targets and a list of predictions.
    :param batch_size: Batch size.
    :param dataset_type: Dataset type.
    :param scaler: A StandardScaler object fit on the training targets.
    :param logger: Logger.
    :return: A list with the score for each task based on `metric_func`.
    """
    preds = predict(
        model=model,
        data=data,
        scaler=scaler
    )

    targets = [d.targets for d in data.get_data()]

    results = evaluate_predictions(
        preds=preds,
        targets=targets,
        metric_func=metric_func
    )

    return results


def evaluate_predictions(preds: List[List[float]],
                         targets: List[List[float]],
                         metric_func: str
                         ) -> List[float]:
    """
    Evaluates predictions using a metric function and filtering out invalid targets.

    :param preds: A list of lists of shape (data_size, num_tasks) with model predictions.
    :param targets: A list of lists of shape (data_size, num_tasks) with targets.
    :param num_tasks: Number of tasks.
    :param metric_func: Metric function which takes in a list of targets and a list of predictions.
    :param dataset_type: Dataset type.
    :param logger: Logger.
    :return: A list with the score for each task based on `metric_func`.
    """
    num_tasks = len(targets[0])
    if len(preds) == 0:
        return [float('nan')] * num_tasks

    # Filter out empty targets
    # valid_preds and valid_targets have shape (num_tasks, data_size)
    valid_preds = [[] for _ in range(num_tasks)]
    valid_targets = [[] for _ in range(num_tasks)]
    for i in range(num_tasks):
        for j in range(len(preds)):
            if not np.isnan(targets[j][i]):  # Skip those without targets
                valid_preds[i].append(preds[j][i])
                valid_targets[i].append(targets[j][i])

    # Compute metric
    results = []
    for i in range(num_tasks):
        if metric_func == "rmse":
            results.append(rmse(valid_targets[i], valid_preds[i]))
        elif metric_func == "mse":
            results.append(mse(valid_targets[i], valid_preds[i]))

    return results


def predict(model: nn.Module,
            data: DatapointList,
            scaler: Scaler = None):
    """
    Makes predictions on a dataset using an ensemble of models.

    :param model: A model.
    :param data: A MoleculeDataset.
    :param batch_size: Batch size.
    :param scaler: A StandardScaler object fit on the training targets.
    :return: A list of lists of predictions. The outer list is examples
    while the inner list is tasks.
    """
    model.eval()
    preds = []

    batch_size = 1
    num_iters = len(data.get_data()) // batch_size * batch_size
    iter_size = batch_size

    for i in trange(0, num_iters, iter_size):
        if (i + iter_size) > len(data.get_data()):
            break
        batch = DatapointList(data.get_data()[i:i+batch_size])
        with torch.no_grad():
            pred = model(batch)
        pred = pred.data.cpu().numpy().tolist()
        preds.extend(pred)

    if num_iters == 0:
        preds = []
    for i in range(0, len(preds)):
        data.get_data()[i].scaled_predictions = preds[i]

    if scaler is not None:
        preds = scaler.inverse_transform(preds)

    for i in range(0, len(preds)):
        data.get_data()[i].predictions = preds[i]
    preds = preds.tolist()
    return preds


def rmse(targets: List[float], preds: List[float]) -> float:
    """
    Computes the root mean squared error.

    :param targets: A list of targets.
    :param preds: A list of predictions.
    :return: The computed rmse.
    """
    return math.sqrt(mean_squared_error(targets, preds))


def mse(targets: List[float], preds: List[float]) -> float:
    """
    Computes the mean squared error.

    :param targets: A list of targets.
    :param preds: A list of predictions.
    :return: The computed mse.
    """
    return mean_squared_error(targets, preds)

