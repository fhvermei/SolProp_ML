from solvation_predictor import inp
from solvation_predictor.train.train import (
    create_logger,
    load_checkpoint,
    load_scaler,
    load_input,
)
from solvation_predictor.data.data import DatapointList, read_data
from solvation_predictor.train.evaluate import predict
import csv
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math

if __name__ == "__main__":
    inp = inp.InputArguments()

    logging = create_logger("predict", inp.output_dir)
    logger = logging.debug

    # load scalar and training input
    scaler = load_scaler(inp.model_path)
    train_inp = load_input(inp.model_path)

    all_data = read_data(inp)

    inp.num_mols = len(all_data[0].mol)
    inp.f_mol_size = all_data[0].get_mol_encoder()[0].get_sizes()[2]
    inp.num_targets = len(all_data[0].targets)
    inp.num_features = len(all_data[0].features)

    all_data = DatapointList(all_data)

    if inp.scale == "standard":
        scaler.transform_standard(all_data)
        logger(f"Scaled data with {inp.scale} method")
        logger(f"mean: {scaler.mean[0]:5.2f}, std: {scaler.std[0]:5.2f}")
    else:
        raise ValueError("scaler not supported")

    model = load_checkpoint(inp.model_path, inp, logger=logging)
    preds = predict(model=model, data=all_data, scaler=scaler)

    with open(os.path.join(inp.output_dir, "results_prediction.csv"), "w+") as f:
        writer = csv.writer(f)
        i = all_data.get_data()[0]
        row = (
            ["Smiles" for s in range(len(i.smiles))]
            + ["Inchi" for s in range(len(i.inchi))]
            + ["target" for s in range(len(i.targets))]
            + ["prediction" for s in range(len(i.targets))]
            + ["difference" for k in range(len(i.targets))]
        )

        writer.writerow(row)
        for i in all_data.get_data():
            diff = np.abs(i.targets - i.predictions)
            row = (
                [s for s in i.smiles]
                + [s for s in i.inchi]
                + [s for s in i.targets]
                + [s for s in i.predictions]
                + [
                    np.abs(i.targets[k] - i.predictions[k])
                    for k in range(len(i.targets))
                ]
            )
            writer.writerow(row)

    fig, ax = plt.subplots()
    s = [d.targets[0] for d in all_data.get_data()]
    p = [d.predictions[0] for d in all_data.get_data()]
    ax.plot(
        [d.targets[0] for d in all_data.get_data()],
        [d.predictions[0] for d in all_data.get_data()],
        "b.",
    )
    ax.set(xlabel="targets", ylabel="predictions")
    fig.savefig(os.path.join(inp.output_dir, "parity_predictions.png"))

    with open(os.path.join(inp.output_dir, "summary.csv"), "w+") as f:
        writer = csv.writer(f)
        writer.writerow(["rmse", "mse", "mae", "max", "r2"])
        t = [d.targets[0] for d in all_data.get_data()]
        p = [d.predictions[0] for d in all_data.get_data()]
        writer.writerow(
            [
                math.sqrt(mean_squared_error(t, p)),
                mean_squared_error(t, p),
                mean_absolute_error(t, p),
                np.max(np.abs(np.subtract(t, p))),
            ]
        )
