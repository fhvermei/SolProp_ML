import numpy as np

from solvation_predictor.data.data import DatapointList, DataPoint
from solvation_predictor.solubility.solubility_data import SolubilityData
from solvation_predictor.solubility.solubility_models import SolubilityModels
from solvation_predictor.train.evaluate import predict


class SolubilityPredictions:
    def __init__(self, data: SolubilityData = None,
                 models: SolubilityModels = None,
                 predict_aqueous: bool = False,
                 predict_reference_solvents: bool = False,
                 predict_t_dep: bool = False,
                 predict_solute_parameters: bool = False,
                 logger=None,
                 verbose=True):
        """
        Make the machine learning model predictions
            :param data: data of the type SolubilityData
            :param models: models of the type SolubilityModels
            :param predict_aqueous: if you want to calculate solubility using the model predicted aqueous solubility
            :param predict_reference_solvents: if you want to calculate solubility using reference solvents
            :param predict_t_dep: if you want to calculate temperature dependent solubility
            :param predict_solute_parameters: if you want to predict solute parameters
            :param logger: logger file
            :param verbose: whether to show logger info or not
        """
        self.data = data if data is not None else None
        self.models = models if models is not None else None
        self.logger = logger.info if logger is not None else print

        self.gsolv = None
        self.hsolv = None
        self.saq = None
        self.gsolv_aq = None
        self.gsolv_ref = None
        self.solute_parameters = None

        if self.data is not None and self.models is not None:
            self.make_all_model_predictions(predict_aqueous=predict_aqueous,
                                            predict_reference_solvents=predict_reference_solvents,
                                            predict_t_dep=predict_t_dep,
                                            predict_solute_parameters=predict_solute_parameters,
                                            verbose=verbose)

    def set_data(self, data: SolubilityData):
        self.data = data

    def set_models(self, models: SolubilityModels):
        self.models = models

    def make_all_model_predictions(self, predict_aqueous: bool = False, predict_reference_solvents: bool = False,
                                   predict_t_dep: bool = False, predict_solute_parameters: bool = False, verbose=False):

        self.gsolv = self.make_gsolv_predictions(verbose=verbose) if self.models.g_models is not None else None
        self.hsolv = self.make_hsolv_predictions(verbose=verbose) if self.models.h_models is not None else None
        self.saq = self.make_saq_predictions(verbose=verbose) if self.models.saq_models is not None else None

        self.gsolv_aq = self.make_gsolvaq_predictions(verbose=verbose) \
            if predict_aqueous and self.models.g_models is not None else None
        self.gsolv_ref = self.make_gsolvref_predictions(verbose=verbose) \
            if predict_reference_solvents and self.models.g_models is not None else None

        self.solute_parameters = self.make_soluteparameter_predictions(verbose=verbose) \
            if predict_t_dep or predict_solute_parameters else None

    def make_gsolv_predictions(self, verbose=False):
        if verbose:
            self.logger('Make Gsolv predictions')
        if self.models.g_models is None:
            raise ValueError('Gsolv models are not loaded, cannot make predictions')
        unique_smiles_pairs = set(self.data.smiles_pairs)
        results = self.make_predictions(unique_smiles_pairs, self.models.g_models)
        mean_predictions = [results[sm][0] for sm in self.data.smiles_pairs]
        variance_predictions = [results[sm][1] for sm in self.data.smiles_pairs]
        return mean_predictions, variance_predictions

    def make_hsolv_predictions(self, verbose=False):
        if verbose:
            self.logger('Make Hsolv predictions')
        if self.models.h_models is None:
            raise ValueError('Hsolv models are not loaded, cannot make predictions')
        unique_smiles_pairs = set(self.data.smiles_pairs)
        results = self.make_predictions(unique_smiles_pairs, self.models.h_models)
        mean_predictions = [results[sm][0] for sm in self.data.smiles_pairs]
        variance_predictions = [results[sm][1] for sm in self.data.smiles_pairs]
        return mean_predictions, variance_predictions

    def make_gsolvaq_predictions(self, verbose=False):
        if verbose:
            self.logger('Make Gsolv aqueous predictions')
        if self.models.g_models is None:
            raise ValueError('Gsolv models are not loaded, cannot make predictions')
        aq_smiles_pairs = [('O', sm[1]) for sm in self.data.smiles_pairs]
        results = self.make_predictions(set(aq_smiles_pairs), self.models.g_models)
        mean_predictions = [results[sm][0] for sm in aq_smiles_pairs]
        variance_predictions = [results[sm][1] for sm in aq_smiles_pairs]
        return mean_predictions, variance_predictions

    def make_gsolvref_predictions(self, verbose=False):
        if verbose:
            self.logger('Make Gsolv reference predictions')
        if self.models.g_models is None:
            raise ValueError('Gsolv models are not loaded, cannot make predictions')
        if self.data.reference_solvents is None:
            raise ValueError('Gsolv reference predictions cannot be made because no refrence solvents are provided')
        new_smiles_pairs = [(ref, sm[1]) for ref, sm in zip(self.data.reference_solvents, self.data.smiles_pairs)]
        results = self.make_predictions(set(new_smiles_pairs), self.models.g_models)
        mean_predictions = [results[sm][0] for sm in new_smiles_pairs]
        variance_predictions = [results[sm][1] for sm in new_smiles_pairs]
        return mean_predictions, variance_predictions

    def make_saq_predictions(self, verbose=False):
        if verbose:
            self.logger('Make logSaq predictions')
        if self.models.saq_models is None:
            raise ValueError('logSaq models are not loaded, cannot make predictions')
        solute_smiles = [(sm[1]) for sm in self.data.smiles_pairs]
        results = self.make_predictions(set(solute_smiles), self.models.saq_models)
        mean_predictions = [results[sm][0] for sm in solute_smiles]
        variance_predictions = [results[sm][1] for sm in solute_smiles]
        return mean_predictions, variance_predictions

    def make_predictions(self, smiles_set, models):
        results = dict()
        for sm in smiles_set:
            if not type(sm) is tuple:
                sm = [sm]
            result = self.run_model(inp=models[0],
                                    models=models[2],
                                    scalers=models[1],
                                    smiles=sm)
            if not type(sm) is tuple:
                sm = sm[0]
            results[sm] = result
        return results

    def run_model(self, inp=None, models=None, scalers=None, smiles=None):
        inp.num_targets = 1
        inp.num_features = 0
        preds = []
        for s, m in zip(scalers, models):
            data = DatapointList([DataPoint(smiles, None, None, inp)])
            pred = predict(model=m, data=data, scaler=s)
            preds.append(pred)
        return np.mean(preds), np.var(preds)

    def make_soluteparameter_predictions(self, verbose=False):
        if verbose:
            self.logger('Make solute parameter predictions')
        solute_smiles = [(sm[1]) for sm in self.data.smiles_pairs]
        unique_solute_smiles = [[sm] for sm in set(solute_smiles)]
        results = dict()
        uncertainties = dict()
        average_prediction, epistemic_uncertainty, valid_indices = self.models.solute_models(unique_solute_smiles)
        for i, sm in enumerate(unique_solute_smiles):
            results[sm[0]] = average_prediction[i]
            uncertainties[sm[0]] = epistemic_uncertainty[i]
        mean_predictions = [results[sm] for sm in solute_smiles]
        variance_predictions = [uncertainties[sm] for sm in solute_smiles]
        return mean_predictions, variance_predictions
