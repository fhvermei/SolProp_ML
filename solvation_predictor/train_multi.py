from solvation_predictor import inp_multi
from solvation_predictor.train.train import *
from solvation_predictor.data.data import read_data
import datetime

if __name__ == "__main__":
    # Load the input arguments class, create the logger and read the training input data
    inp_multi = inp_multi.InputArguments()
    logging = create_logger("train", inp_multi.output_dir)
    logger = logging.debug
    logger(f"Start training on {datetime.time()}")
    logger(f"Doing {inp_multi.num_folds} different folds")
    logger(f"Doing {inp_multi.num_models} different models")
    all_data = read_data(inp_multi)

    run_training(inp_multi, all_data, logging)
