import os
from chemprop_solvation.solvation_estimator import load_SoluteML_estimator

from solvation_predictor import inp_logSaq, inp_Gsolv, inp_Hsolv
from solvation_predictor.train.train import load_checkpoint, load_scaler


class SolubilityModels:
    def __init__(self, reduced_number: bool = False,
                 load_g: bool = False,
                 load_h: bool = False,
                 load_saq: bool = False,
                 load_solute: bool = False,
                 logger=None,
                 verbose=True):
        """
        Loads the required models for solvation free energy, enthalpy, and aqueous solubility.
            :param reduced_number: if true, only 3 models are considered per property to make predictions faster
            :param load_g: load models for solvation free energy
            :param load_h: load models for solvation enthalpy
            :param load_saq: load models for aqueous solubility
            :param load_solute: load models for solute parameters
            :param logger: logger file
            :param verbose: whether to show logger info or not
        """
        self.g_models = None
        self.h_models = None
        self.saq_models = None
        self.solute_models = None
        self.logger = logger.info if logger is not None else print

        if load_g or load_h or load_saq:
            self.g_models = self.load_g_models(reduced_number=reduced_number, verbose=verbose) if load_g else None
            self.h_models = self.load_h_models(reduced_number=reduced_number, verbose=verbose) if load_h else None
            self.saq_models = self.load_saq_models(reduced_number=reduced_number, verbose=verbose) if load_saq else None
            self.solute_models = load_SoluteML_estimator() if load_solute else None

    def load_g_models(self, reduced_number=False, verbose=True):
        """
        Loads the solvation free energy models.
            :param reduced_number: if true, only 3 models are considered to make predictions faster
            :returns: model input, model scalers, and model parameters
        """
        number = 10 if not reduced_number else 3
        #path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        paths = [os.path.join('trained_models', 'Gsolv', 'model_Gsolv_' + str(i) + '.pt')
                       for i in range(number)]
        if verbose:
            self.logger(f'Loading {number} solvation free energy models')
        input = inp_Gsolv.InputArguments()
        input.add_hydrogens_to_solvent = False
        input.num_mols = 2
        input.f_mol_size = 2
        input.num_targets = 1
        input.num_features = 0
        scalers = []
        models = []
        for p in paths:
            input.model_path = p
            scaler = load_scaler(p, from_package=True)
            model = load_checkpoint(p, input, from_package=True)
            scalers.append(scaler)
            models.append(model)
        return input, scalers, models

    def load_h_models(self, reduced_number=False, verbose=True):
        """
        Loads the solvation enthalpy models.
            :param reduced_number: if true, only 3 models are considered to make predictions faster
            :returns: model input, model scalers, and model parameters
        """
        number = 12 if not reduced_number else 3
        #path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        paths = [os.path.join('trained_models', 'Hsolv', 'model_Hsolv_' + str(i) + '.pt')
                       for i in range(number)]
        if verbose:
            self.logger(f'Loading {number} solvation enthalpy models')
        input = inp_Hsolv.InputArguments()
        scalers = []
        models = []
        for p in paths:
            input.model_path = p
            scaler = load_scaler(p, from_package=True)
            model = load_checkpoint(p, input, from_package=True)
            scalers.append(scaler)
            models.append(model)
        return input, scalers, models

    def load_saq_models(self, reduced_number=False, verbose=True):
        """
        Loads the aqueous solubility models.
            :param reduced_number: if true, only 3 models are considered to make predictions faster
            :returns: model input, model scalers, and model parameters
        """
        number = 30 if not reduced_number else 3
        #path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        paths = [os.path.join('trained_models', 'Saq', 'model_' + str(i) + '.pt')
                       for i in range(number)]
        if verbose:
            self.logger(f'Loading {number} aqueous solubility models')
        input = inp_logSaq.InputArguments()
        scalers = []
        models = []
        for p in paths:
            input.model_path = p
            scaler = load_scaler(p, from_package=True)
            model = load_checkpoint(p, input, from_package=True)
            scalers.append(scaler)
            models.append(model)
        return input, scalers, models

