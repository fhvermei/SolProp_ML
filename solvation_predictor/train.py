from solvation_predictor import inp
from solvation_predictor.train.train import *
from solvation_predictor.data.data import read_data
import datetime

if __name__ == '__main__':
    inp = inp.InputArguments()

    logging = create_logger("train", inp.output_dir)
    logger = logging.debug
    logger(f"Start training on {datetime.time()}")
    logger(f"Doing {inp.num_folds} different folds")
    logger(f"Doing {inp.num_models} different models")

    all_data = read_data(inp)

    if not inp.optimization:

        run_training(inp, all_data, logging)
    else:
        run_training(inp, all_data, logging)
