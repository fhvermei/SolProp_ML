import torch


class InputArguments:
    """
    Class that holds all input arguments for the prediction and training procedures. Paths, training parameters and
    neural network parameters are included.
    """
    def __init__(self):
        self.optimization = False

        # reading and processing data
        self.dir = "/Users/u0161682/Documents/GitHub/SolProp_ML/data/SolProp_v1.2/"
        self.input_file = self.dir + "dummy_multi_solvent.csv"
        self.split_ratio = (0.8, 0.1, 0.1)
        self.seed = 0
        self.model_path = "/Users/u0161682/Documents/GitHub/SolProp_ML/data/SolProp_v1.2/Model_predictions/" \
                          "fold_0/model0/model.pt"
        self.output_dir = self.dir + "Model_predictions"
        self.make_plots = True
        self.scale = "standard"  # standard or minmax
        self.scale_features = False
        self.use_same_scaler_for_features = False
        # random or scaffold or wo_solvents (the latter is based on random split) or kmeans or onecross
        self.split = "random"
        self.kmeans_split_base = "solvent"  # solvent or solute depending on if you want first or second molecule
        self.save_memory = False

        # for featurization
        self.property = "solvation"  # "solvation"  # alternatives are solvation, Tm and logS, None and nbv_f1, nbv_f2
        self.add_hydrogens_to_solvent = False  # adds hydrogens to first column smiles
        self.mix = False  # features are fractions of the different molecules in the same order

        # for active learning
        self.uncertainty = False  # calculate and output aleotoric uncertainties
        self.ensemble_variance = False  # calculate and output ensemble variance, epi
        # number or adaptpercent-n for n% of training set
        self.active_learning_batch_size = "adaptpercent-10"
        self.active_learning_iterations = 100
        # how to select data, options are: epistemic, total and random, epi_mol, epi_scaled
        # (for epi unc on scaled predictions)
        self.data_selection = "epistemic"
        self.restart_al = False
        self.active_learning_split_ratio = (
            0.3,
            0.4,
            0.3,
        )  # split between initial train data, experimental data and test set

        # for training
        self.num_folds = 1
        self.num_models = 1
        self.epochs = 10
        self.batch_size = 50
        self.loss_metric = "rmse"
        self.pretraining = False
        self.pretraining_path = [
            self.dir + f"examples/trained_models/model_Gsolv{i}0.pt"
            for i in range(0, 10)
        ]
        # mpn or ffn or none or onlylast or mpn1 or onlylast1 if you have only one molecule
        self.pretraining_fix = "none"
        self.learning_rates = (0.001, 0.0001, 0.001)  # initial, final, max
        self.warm_up_epochs = (
            2.0  # you need min 1 with adam optimizer and Noam learning rate scheduler
        )
        self.lr_scheduler = "Noam"  # Noam or Step or Exponential
        # in case of step
        self.step_size = 10
        self.step_decay = 0.2
        # in case of exponential
        self.exponential_decay = 0.1
        self.minimize_score = True
        self.cuda = True and torch.cuda.is_available()
        self.gpu = 4

        # for mpn
        self.depth = 4
        self.mpn_hidden = 200
        self.mpn_dropout = 0.00
        self.mpn_activation = "LeakyReLU"
        self.mpn_bias = False
        self.shared = False
        self.morgan_fingerprint = (
            "None"  # None, only_solvent or All #if you want morgan fingerprints
        )
        self.morgan_bits = 16
        self.morgan_radius = 2
        self.aggregation = "mean"
        # make sure your solvent is th
        # e first in the input file
        # self.dummy_atom_for_single_atoms = True

        # for attention
        self.attention = False  # True or false
        self.att_hidden = 200
        self.att_dropout = 0.0
        self.att_bias = False
        self.att_activation = "ReLU"
        self.att_normalize = "sigmoid"  # sigmoid or softmax or logsigmoid of logsoftmax or None
        self.att_first_normalize = False

        # for ffn
        self.ffn_hidden = 500
        self.ffn_num_layers = 4
        self.ffn_dropout = 0.00
        self.ffn_activation = "LeakyReLU"
        self.ffn_bias = True

        # results
        self.print_weigths = False
        self.postprocess = False

        # DONT CHANGE!
        self.num_mols = 2
        self.f_mol_size = 2
        self.num_targets = 1
        self.num_features = 0
