import os

from solvation_predictor import inp_logSaq, inp_Gsolv, inp_Hsolv
from solvation_predictor.train.train import load_checkpoint, load_scaler


class SolubilityModels:
    def __init__(self, reduced_number: bool = False,
                 load_g: bool = True,
                 load_h: bool = True,
                 load_saq: bool = True,
                 logger=None):
        """
        Loads the required models for solvation free energy, enthalpy, and aqueous solubility.
            :param reduced_number: if true, only 3 models are considered per property to make predictions faster
            :param load_g: load models for solvation free energy
            :param load_h: load models for solvation enthalpy
            :param load_saq: load models for aqueous solubility
        """
        self.g_models = self.load_g_models(reduced_number=reduced_number, logger=logger) if load_g else None
        self.h_models = self.load_h_models(reduced_number=reduced_number, logger=logger) if load_h else None
        self.saq_models = self.load_saq_models(reduced_number=reduced_number, logger=logger) if load_saq else None

    def load_g_models(self, reduced_number=False, logger=None):
        """
        Loads the solvation free energy models.
            :param reduced_number: if true, only 3 models are considered to make predictions faster
            :returns: model input, model scalers, and model parameters
        """
        number = 10 if not reduced_number else 3
        path = './../'
        paths = [os.path.join(path, 'trained_models', 'Gsolv', 'model_Gsolv_' + str(i) + '.pt')
                       for i in range(number)]
        logger.info(f'Loading {number} solvation free energy models from {path}')
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
            scaler = load_scaler(p, from_package=False)
            model = load_checkpoint(p, input, from_package=False)
            scalers.append(scaler)
            models.append(model)
        return input, scalers, models

    def load_h_models(self, reduced_number=False, logger=None):
        """
        Loads the solvation enthalpy models.
            :param reduced_number: if true, only 3 models are considered to make predictions faster
            :returns: model input, model scalers, and model parameters
        """
        number = 12 if not reduced_number else 3
        path = '//'

        paths = [os.path.join(path, 'trained_models', 'Hsolv', 'model_Hsolv_' + str(i) + '.pt')
                       for i in range(number)]
        logger.info(f'Loading {number} solvation enthalpy models from {path}')
        input = inp_Hsolv.InputArguments()
        scalers = []
        models = []
        for p in paths:
            input.model_path = p
            scaler = load_scaler(p, from_package=False)
            model = load_checkpoint(p, input, from_package=False)
            scalers.append(scaler)
            models.append(model)
        return input, scalers, models

    def load_saq_models(self, reduced_number=False, logger=None):
        """
        Loads the aqueous solubility models.
            :param reduced_number: if true, only 3 models are considered to make predictions faster
            :returns: model input, model scalers, and model parameters
        """
        number = 30 if not reduced_number else 3
        path = '//'

        paths = [os.path.join(path, 'trained_models', 'Saq', 'model_Saq_' + str(i) + '.pt')
                       for i in range(number)]
        logger.info(f'Loading {number} aqueous solubility models from {path}')
        input = inp_logSaq.InputArguments()
        scalers = []
        models = []
        for p in paths:
            input.model_path = p
            scaler = load_scaler(p, from_package=False)
            model = load_checkpoint(p, input, from_package=False)
            scalers.append(scaler)
            models.append(model)
        return input, scalers, models

