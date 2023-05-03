import csv
import math

from solvation_predictor.inp import InputArguments
from solvation_predictor.data.scaler import Scaler
from solvation_predictor.data.data import DatapointList
from solvation_predictor.data.splitter import Splitter
from solvation_predictor.models.model import Model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging
from typing import Callable, List, Union

from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import _LRScheduler, ExponentialLR, StepLR
from tqdm import trange
import numpy as np
from solvation_predictor.train.evaluate import evaluate, predict
import matplotlib.pyplot as plt
from logging import Logger
import pandas as pd
import os
import io, pkgutil


def train(
    model: nn.Module,
    data: DatapointList,
    loss_func: Callable,
    optimizer: Optimizer,
    scheduler: _LRScheduler,
    n_iter: int = 0,
    logger: logging.Logger = None,
    writer: SummaryWriter = None,
    inp: InputArguments = None,
) -> (int, float):
    """
    Trains a model for an epoch.

    :param model: A class containing the model used for training.
    :param data: A class containing a list of data points on which operations can be executed.
    :param loss_func: Loss function.
    :param optimizer: An optimizer.
    :param scheduler: A learning rate scheduler.
    :param n_iter: The number of iterations (training examples) trained on so far.
    :param logger: A logger for recording output.
    :param writer: A tensorboardX SummaryWriter.
    :param inp: A class containing all input arguments for the training procedure.
    :return: The total number of iterations (training examples) trained on so far and the sum of loss.
    """
    debug = logger.debug if logger is not None else print

    model.train()
    data.shuffle()
    loss_sum, iter_count = 0, 0
    batch_size = inp.batch_size
    cuda = inp.cuda

    num_iters = len(data.get_data()) // batch_size * batch_size
    iter_size = batch_size

    for i in trange(0, num_iters, iter_size):
        if (i + iter_size) > len(data.get_data()):
            break
        batch = DatapointList(data.get_data()[i : i + batch_size])

        # make mol graphs and mol vectors now if saving memory
        # training will take longer because encoders have to be constructed again every time

        mask = torch.Tensor(
            [[not np.isnan(x) for x in tb] for tb in batch.get_scaled_targets()]
        )
        targets = torch.Tensor(
            [[0 if np.isnan(x) else x for x in tb] for tb in batch.get_scaled_targets()]
        )

        if next(model.parameters()).is_cuda or inp.cuda:
            device = torch.device('mps')
            model = model.to(device)
            targets = targets.to(device)
            mask = mask.to(device)

        # Run model
        model.zero_grad()
        preds = model(batch)
        loss = loss_func(preds, targets) * mask  # tensor with loss of all datapoints
        loss = loss.sum() / mask.sum()  # sum of all loss in batch
        loss_sum += (
            loss.item()
        )  # sum over all batches in one epoch, so we have loss per epoch
        iter_count += batch_size
        loss.backward()
        optimizer.step()
        if isinstance(scheduler, NoamLR):
            scheduler.step()  # adjust learning rate every batch
        # debug(f"Update of gradients and learning rate, new LR: {scheduler.get_lr()[0]:10.2e}")
        # if isinstance(scheduler, NoamLR):
        #    scheduler.step()

        n_iter += batch_size
    return n_iter, loss_sum


def run_training(inp: InputArguments, all_data: DatapointList, logger: Logger):
    """
    Loads data, trains a Chemprop model, and returns test scores for the model checkpoint with the highest validation
    score.

    :param inp: A Class object containing arguments for loading data and training the SolProp model.
    :param all_data: A Class object containing the data.
    :param logger: A logger to record output.
    """
    logger = logger.debug if logger is not None else print
    inp.num_mols = len(all_data[0].smiles)
    inp.f_mol_size = all_data[0].get_mol_encoder()[0].get_sizes()[2]
    inp.num_targets = len(all_data[0].targets)
    inp.num_features = len(all_data[0].features)
    logger(f"Total number of datapoints read is {len(all_data)}")
    logger(
        f"Number of molecules: {len(all_data[0].mol)}, features: {len(all_data[0].features)}, "
        f"targets: {len(all_data[0].targets)}"
    )
    initial_seed = inp.seed
    seed = inp.seed
    test_scores = dict()

    for fold in trange(inp.num_folds):
        seed = initial_seed + fold
        inp.seed = seed
        logger(f"Starting with fold number {fold} having seed {seed}")

        splitter = Splitter(seed=seed, split_ratio=inp.split_ratio)
        if inp.split == "random":
            train_data, val_data, test_data = splitter.split_random(all_data)
        elif inp.split == "scaffold":
            train_data, val_data, test_data = splitter.split_scaffold(all_data)
        else:
            raise ValueError("splitter not supported")
        logger(
            f"Splitting data with {len(train_data)} training data, {len(val_data)} validation data and "
            f"{len(test_data)} test data points."
        )
        train_data_size = len(train_data)
        train_data, val_data, test_data = (
            DatapointList(d) for d in (train_data, val_data, test_data)
        )

        scaler = Scaler(data=train_data, scale_features=inp.scale_features)
        if inp.scale == "standard":
            [scaler.transform_standard(d) for d in (train_data, val_data, test_data)]
            logger(f"Scaled data with {inp.scale} method")
            logger(f"mean: {scaler.mean[0]:5.2f}, std: {scaler.std[0]:5.2f}")
        else:
            raise ValueError("scaler not supported")

        for model_i in trange(inp.num_models):
            path = inp.output_dir + "/fold_" + str(fold) + "/model" + str(model_i)
            if path != "":
                os.makedirs(path, exist_ok=True)
            loss_func = get_loss_func(inp.loss_metric)
            logger(f"Initiate model {model_i}")
            model = Model(inp, logging)
            initialize_weights(model, seed=initial_seed + model_i)

            if inp.cuda:
                logger("Moving model to mps")
                device = torch.device('mps')
                model = model.to(device)

            logger(f"Save checkpoint file to {path}/model.pt")
            save_checkpoint(os.path.join(path, "model.pt"), model, inp, scaler)
            if inp.pretraining and inp.pretraining_path is not None:
                logger(f"Load pretraining parameters from {inp.pretraining_path}")
                model = load_checkpoint(inp.pretraining_path, current_inp=inp)
                logger(f"Load scaler from pretrained dataset")
                scaler = load_scaler(inp.pretraining_path)
                if inp.scale == "standard":
                    [
                        scaler.transform_standard(d)
                        for d in (train_data, val_data, test_data)
                    ]
                    logger(f"Scaled data with {inp.scale} method")
                    logger(f"mean: {scaler.mean[0]:5.2f}, std: {scaler.std[0]:5.2f}")
                else:
                    raise ValueError(
                        "minmax scaling not supported for pretraining and model loading"
                    )
                if inp.pretraining_fix == "mpn":
                    if inp.num_mols == 1:
                        for param in model.mpn.parameters():
                            param.requires_grad = False
                    else:
                        for param in model.mpn_1.parameters():
                            param.requires_grad = False
                        for param in model.mpn_2.parameters():
                            param.requires_grad = False
                elif inp.pretraining_fix == "onlysolute":
                    for param in model.mpn_2.parameters():
                        param.requires_grad = False
                elif inp.pretraining_fix == "ffn":
                    for param in model.ffn.parameters():
                        param.requires_grad = False

            optimizer = build_optimizer(model, inp.learning_rates[0])
            scheduler = build_lr_scheduler(optimizer, inp, train_data_size)

            best_score = float("inf")
            best_epoch = 0
            list_train_loss = []
            list_val_loss = []
            list_lr = []
            for epoch in trange(inp.epochs):
                logger(f"Starting epoch {epoch}/{inp.epochs}")
                (n_iter, loss_sum) = train(
                    model=model,
                    data=train_data,
                    loss_func=loss_func,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    inp=inp,
                )
                list_train_loss.append(loss_sum)
                if isinstance(scheduler, ExponentialLR) or isinstance(
                    scheduler, StepLR
                ):
                    # scheduler.step(epoch=epoch)
                    scheduler.step(epoch=epoch)
                list_lr.append(scheduler.get_lr()[0])
                logger(f"Evaluating validation set of epoch {epoch}/{inp.epochs}")
                val_scores = evaluate(
                    model=model,
                    data=val_data,
                    metric_func=inp.loss_metric,
                    scaler=scaler,
                )
                avg_val_score = np.nanmean(val_scores)
                list_val_loss.append(avg_val_score)
                # logger(f"Scores = {val_scores}")
                logger(f"Average validation score = {avg_val_score:5.2f}")
                # save model with lowest validation score for test
                if (inp.minimize_score and avg_val_score < best_score) or (
                    not inp.minimize_score and avg_val_score > best_score
                ):
                    logger(f"New best score")
                    best_score, best_epoch = avg_val_score, epoch
                    logger(f"Saving best model to {path}/model.pt")
                    save_checkpoint(os.path.join(path, "model.pt"), model, inp, scaler)

            logger(
                f"Loading best model, from epoch {best_epoch} with validation score {best_score:5.2f}"
            )
            model = load_checkpoint(path, inp)
            if inp.print_weigths:
                for param in model.mpn.W_i.parameters():
                    print(param)
            if inp.make_plots:
                test_rmse, test_mae = process_results(
                    path=path,
                    input=inp,
                    model=model,
                    data=[train_data, val_data, test_data],
                    scaler=scaler,
                    loss=[list_train_loss, list_val_loss],
                )
                test_scores[(fold, model_i)] = [test_rmse, test_mae]
        if inp.ensemble_variance:
            write_ensemble_summary(inp, fold)
        inp.seed = initial_seed

    if inp.make_plots:
        with open(os.path.join(inp.output_dir, "summary.csv"), "w+") as f:
            writer = csv.writer(f)
            writer.writerow(["fold", "model", "test_rmse", "test_mae"])
            for key in test_scores.keys():
                writer.writerow(
                    [key[0], key[1], test_scores[key][0], test_scores[key][1]]
                )


def write_ensemble_summary(inp: InputArguments, fold: int):
    names = ["train", "val", "test"]
    path = inp.output_dir + "/fold_" + str(fold)
    for i in names:
        df = pd.DataFrame()
        prediction_columns = []
        for folder_name in os.listdir(path):
            if os.path.isdir(os.path.join(path, folder_name)):
                with open(
                    path + "/" + folder_name + "/" + i + "_summary.csv", "r"
                ) as f:
                    df_temp = pd.read_csv(f)
                    n = "predictions_" + folder_name
                    if not "smiles" in df.columns:
                        df["smiles"] = df_temp.smiles
                        df["inchi"] = df_temp.inchi
                        df["target"] = df_temp.target
                    df[n] = df_temp.prediction
                    prediction_columns.append(n)

        df["average_prediction"] = df[prediction_columns].mean(axis=1)
        df["ensemble_variance"] = df[prediction_columns].var(axis=1)
        df.to_csv(path + "/" + i + "_ensemble_summary.csv")


def initialize_weights(model: nn.Module, seed=0):
    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        else:
            torch.manual_seed(seed)
            nn.init.xavier_normal_(param)


def build_optimizer(model: nn.Module, init_lr):
    params = [{"params": model.parameters(), "lr": init_lr, "weight_decay": 0}]
    return Adam(params)


def build_lr_scheduler(optimizer: Optimizer, input, train_data_size) -> _LRScheduler:
    """
    Builds a learning rate scheduler.

    :param optimizer: The Optimizer whose learning rate will be scheduled.
    :param args: Arguments.
    :param total_epochs: The total number of epochs for which the model will be run.
    :return: An initialized learning rate scheduler.
    """
    # Learning rate scheduler
    if input.lr_scheduler == "Noam":
        return NoamLR(
            optimizer=optimizer,
            warmup_epochs=[input.warm_up_epochs],
            total_epochs=[input.epochs],
            steps_per_epoch=train_data_size // input.batch_size,
            init_lr=[input.learning_rates[0]],
            max_lr=[input.learning_rates[2]],
            final_lr=[input.learning_rates[1]],
        )

    else:
        raise ValueError(
            f'Learning rate scheduler "{input.lr_scheduler}" not supported.'
        )


def save_checkpoint(
    path: str,
    model: Model,
    inp: InputArguments,
    scaler: Scaler = None,
):
    """
    Saves a model checkpoint.

    :param model: A MoleculeModel.
    :param scaler: A StandardScaler fitted on the data.
    :param features_scaler: A StandardScaler fitted on the features.
    :param args: Arguments namespace.
    :param path: Path where checkpoint will be saved.
    """
    state = {
        "input": inp,
        "state_dict": model.state_dict(),
        "data_scaler": {"means": scaler.mean, "stds": scaler.std}
        if scaler is not None
        else None,
        "features_scaler": {"means": scaler.mean_features, "stds": scaler.std_features}
        if scaler is not None
        else None,
        "scale_features": inp.scale_features,
        "use_same_scaler_for_features": inp.use_same_scaler_for_features,
    }
    torch.save(state, path)


def get_loss_func(metric):
    # todo check loss function that accounts as much for low values
    if metric == "rmse":
        return nn.MSELoss(reduction="none")
    if metric == "mae":
        return nn.L1Loss(reduction="none")
    if metric == "smooth":
        return nn.SmoothL1Loss()
    raise ValueError(f'Metric for loss function "{metric}" not supported.')


def load_checkpoint(
    path: str,
    current_inp: InputArguments,
    logger: logging.Logger = None,
    from_package=False,
) -> Model:
    """
    Loads a model checkpoint.

    :param path: Path where checkpoint is saved.
    :param current_args: The current arguments. Replaces the arguments loaded from the checkpoint if provided.
    :param cuda: Whether to move model to cuda.
    :param logger: A logger.
    :return: The loaded MoleculeModel.
    """

    debug = logger.debug if logger is not None else print

    # Load model and args
    if from_package:
        state = torch.load(
            io.BytesIO(pkgutil.get_data("solvation_predictor", path)),
            map_location=lambda storage, loc: storage,
        )
    else:
        if ".pt" not in path:
            path = path + "/model.pt"

        state = torch.load(path, map_location=lambda storage, loc: storage)
    inp, loaded_state_dict = state["input"], state["state_dict"]

    if current_inp is not None:
        args = current_inp
    # args.cuda = cuda if cuda is not None else args.cuda

    # Build model
    model = Model(args)
    model_state_dict = model.state_dict()

    # Skip missing parameters and parameters of mismatched size
    pretrained_state_dict = {}
    for param_name in loaded_state_dict.keys():
        if param_name not in model_state_dict:
            debug(
                f'Pretrained parameter "{param_name}" cannot be found in model parameters.'
            )
        elif model_state_dict[param_name].shape != loaded_state_dict[param_name].shape:
            debug(
                f'Pretrained parameter "{param_name}" '
                f"of shape {loaded_state_dict[param_name].shape} does not match corresponding "
                f"model parameter of shape {model_state_dict[param_name].shape}."
            )
        else:
            debug(f'Loading pretrained parameter "{param_name}".')
            pretrained_state_dict[param_name] = loaded_state_dict[param_name]

    # Load pretrained weights
    model_state_dict.update(pretrained_state_dict)
    model.load_state_dict(model_state_dict)

    if args.cuda:
        debug("Moving model to cuda")
        device = torch.device('mps')
        model = model.to(device)

    return model


class NoamLR(_LRScheduler):
    """
    Noam learning rate scheduler with piecewise linear increase and exponential decay.

    The learning rate increases linearly from init_lr to max_lr over the course of
    the first warmup_steps (where warmup_steps = warmup_epochs * steps_per_epoch).
    Then the learning rate decreases exponentially from max_lr to final_lr over the
    course of the remaining total_steps - warmup_steps (where total_steps =
    total_epochs * steps_per_epoch). This is roughly based on the learning rate
    schedule from Attention is All You Need, section 5.3 (https://arxiv.org/abs/1706.03762).
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: List[Union[float, int]],
        total_epochs: List[int],
        steps_per_epoch: int,
        init_lr: List[float],
        max_lr: List[float],
        final_lr: List[float],
    ):
        """
        Initializes the learning rate scheduler.

        :param optimizer: A PyTorch optimizer.
        :param warmup_epochs: The number of epochs during which to linearly increase the learning rate.
        :param total_epochs: The total number of epochs.
        :param steps_per_epoch: The number of steps (batches) per epoch.
        :param init_lr: The initial learning rate.
        :param max_lr: The maximum learning rate (achieved after warmup_epochs).
        :param final_lr: The final learning rate (achieved after total_epochs).
        """
        # assert len(optimizer.param_groups) == len(warmup_epochs) == len(total_epochs) == len(init_lr) == \
        #       len(max_lr) == len(final_lr)

        self.num_lrs = len(optimizer.param_groups)

        self.optimizer = optimizer
        self.warmup_epochs = np.array(warmup_epochs)
        self.total_epochs = np.array(total_epochs)
        self.steps_per_epoch = steps_per_epoch
        self.init_lr = np.array(init_lr)
        self.max_lr = np.array(max_lr)
        self.final_lr = np.array(final_lr)

        self.current_step = 0
        self.lr = init_lr
        self.warmup_steps = (self.warmup_epochs * self.steps_per_epoch).astype(int)
        self.total_steps = self.total_epochs * self.steps_per_epoch
        self.linear_increment = (self.max_lr - self.init_lr) / self.warmup_steps

        self.exponential_gamma = (self.final_lr / self.max_lr) ** (
            1 / (self.total_steps - self.warmup_steps)
        )

        super(NoamLR, self).__init__(optimizer)

    def get_lr(self) -> List[float]:
        """Gets a list of the current learning rates."""
        return list(self.lr)

    def step(self, current_step: int = None, epoch: int = None):
        """
        Updates the learning rate by taking a step.

        :param current_step: Optionally specify what step to set the learning rate to.
        If None, current_step = self.current_step + 1.
        """
        if current_step is not None:
            self.current_step = current_step
        else:
            self.current_step += 1

        for i in range(self.num_lrs):
            if self.current_step <= self.warmup_steps[i]:
                self.lr[i] = (
                    self.init_lr[i] + self.current_step * self.linear_increment[i]
                )
            elif self.current_step <= self.total_steps[i]:
                self.lr[i] = self.max_lr[i] * (
                    self.exponential_gamma[i]
                    ** (self.current_step - self.warmup_steps[i])
                )
            else:  # theoretically this case should never be reached since training should stop at total_steps
                self.lr[i] = self.final_lr[i]

            self.optimizer.param_groups[i]["lr"] = self.lr[i]


def create_logger(name: str, save_dir: str = None) -> logging.Logger:
    """
    Creates a logger with a stream handler and two file handlers.

    The stream handler prints to the screen depending on the value of `quiet`.
    One file handler (verbose.log) saves all logs, the other (quiet.log) only saves important info.

    :param name: The name of the logger.
    :param save_dir: The directory in which to save the logs.
    :param quiet: Whether the stream handler should be quiet (i.e. print only important info).
    :return: The logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # Set logger depending on desired verbosity
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        fh_v = logging.FileHandler(os.path.join(save_dir, "logger.log"))
        fh_v.setLevel(logging.DEBUG)
        logger.addHandler(fh_v)

    return logger


def process_results(
    path: str,
    input: InputArguments,
    model: nn.Module,
    data: [],
    scaler: Scaler,
    loss: [],
):
    train_preds = predict(model=model, data=data[0], scaler=scaler)
    val_preds = predict(model=model, data=data[1], scaler=scaler)
    test_preds = predict(model=model, data=data[2], scaler=scaler)

    fig, ax = plt.subplots()
    train_data = DatapointList(data[0].get_data()[0 : len(train_preds)])
    ax.plot([d.targets[0] for d in train_data.get_data()], train_preds, "b.")
    ax.set(xlabel="targets", ylabel="predictions")
    fig.savefig(os.path.join(path, "train_parity.png"))
    plt.close()
    fig, ax = plt.subplots()
    val_data = DatapointList(data[1].get_data()[0 : len(val_preds)])
    ax.plot([d.targets[0] for d in val_data.get_data()], val_preds, "b.")
    ax.set(xlabel="targets", ylabel="predictions")
    fig.savefig(os.path.join(path, "val_parity.png"))
    plt.close()
    fig, ax = plt.subplots()
    test_data = DatapointList(data[2].get_data()[0 : len(test_preds)])
    ax.plot([d.targets[0] for d in test_data.get_data()], test_preds, "b.")
    ax.set(xlabel="targets", ylabel="predictions")
    fig.savefig(os.path.join(path, "test_parity.png"))
    plt.close()

    data2 = [train_data, val_data, test_data]
    write_summary(path=path, input=input, data=data2[0], name="train")
    write_summary(path=path, input=input, data=data2[1], name="val")
    write_summary(path=path, input=input, data=data2[2], name="test")

    with open(os.path.join(path, "summary.csv"), "w+") as f:
        writer = csv.writer(f)
        row = ["name", "rmse", "mse", "mae", "max"]
        writer.writerow(row)
        names = ["train", "val", "test"]
        for i in range(0, len(names)):
            if len(data2[i].get_data()) > 0:
                row = [
                    names[i],
                    rmse(data=data2[i]),
                    mse(data=data2[i]),
                    mae(data=data2[i]),
                    max(data=data2[i]),
                ]
                writer.writerow(row)
    if len(data2[2].get_data()) > 0:
        return rmse(data=data2[2]), mae(data=data2[2])
    else:
        return None, None


def write_summary(path: str, input: InputArguments, data: DatapointList, name: str):
    with open(os.path.join(path, name + "_summary.csv"), "w+", encoding="utf-8") as f:
        writer = csv.writer(f)
        row = (
            ["inchi"] * input.num_mols
            + ["smiles"] * input.num_mols
            + ["target"] * input.num_targets
            + ["prediction"] * input.num_targets
        )
        if input.split == "scaffold":
            row += ["scaffold"]
        writer.writerow(row)
        for d in data.get_data():
            row = (
                [m for m in d.inchi]
                + [m for m in d.smiles]
                + [t for t in d.targets]
                + [p for p in d.predictions]
            )
            if input.split == "scaffold":
                row += [d.get_scaffold()[len(d.get_scaffold()) - 1]]
            writer.writerow(row)


def rmse(data: DatapointList):
    """
    Computes the root mean squared error.

    :param data:
    :param targets: A list of targets.
    :param preds: A list of predictions.
    :return: The computed rmse.
    """
    targets = [d.targets for d in data.get_data()]
    preds = [d.predictions for d in data.get_data()]
    return math.sqrt(mean_squared_error(targets, preds))


def mse(data: DatapointList) -> float:
    """
    Computes the mean squared error.

    :param data:
    :param targets: A list of targets.
    :param preds: A list of predictions.
    :return: The computed mse.
    """
    targets = [d.targets for d in data.get_data()]
    preds = [d.predictions for d in data.get_data()]
    return mean_squared_error(targets, preds)


def mae(data: DatapointList) -> float:
    """
    Computes the mean squared error.

    :param data:
    :param targets: A list of targets.
    :param preds: A list of predictions.
    :return: The computed mse.
    """
    targets = [d.targets for d in data.get_data()]
    preds = [d.predictions for d in data.get_data()]
    return mean_absolute_error(targets, preds)


def r2(data: DatapointList) -> float:
    """
    Computes the mean squared error.

    :param data:
    :param targets: A list of targets.
    :param preds: A list of predictions.
    :return: The computed mse.
    """
    targets = [d.targets for d in data.get_data()]
    preds = [d.predictions for d in data.get_data()]
    return r2_score(targets, preds)


def max(data: DatapointList) -> float:
    """
    Computes the mean squared error.

    :param data:
    :param targets: A list of targets.
    :param preds: A list of predictions.
    :return: The computed mse.
    """
    targets = [d.targets for d in data.get_data()]
    preds = [d.predictions for d in data.get_data()]
    return np.max(np.abs(np.subtract(targets, preds)))


def load_scaler(path: str, from_package=False) -> Scaler:
    if from_package:
        state = torch.load(
            io.BytesIO(pkgutil.get_data("solvation_predictor", path)),
            map_location=lambda storage, loc: storage,
        )
    else:
        if ".pt" not in path:
            path = path + "/model.pt"
        state = torch.load(path, map_location=lambda storage, loc: storage)
    sc_fe = state["scale_features"] if "scale_features" in state.keys() else True
    same_sc_fe = (
        state["use_same_scaler_for_features"]
        if "use_same_scaler_for_features" in state.keys()
        else True
    )
    scaler = Scaler(
        mean=state["data_scaler"]["means"],
        std=state["data_scaler"]["stds"],
        mean_f=state["features_scaler"]["means"],
        std_f=state["features_scaler"]["stds"],
        scale_features=sc_fe,
    )
    return scaler


def load_input(path: str) -> InputArguments:
    return torch.load(path, map_location=lambda storage, loc: storage)["input"]
