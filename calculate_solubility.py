import logging
import os

import numpy as np
import pandas as pd

from solvation_predictor.solubility.solubility_calculator import SolubilityCalculations
from solvation_predictor.solubility.solubility_data import SolubilityData
from solvation_predictor.solubility.solubility_models import SolubilityModels
from solvation_predictor.solubility.solubility_predictions import SolubilityPredictions


def create_logger(save_dir):
    logger = logging.getLogger('log_solubility_predictions.log')
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if save_dir is not None:
        fh_v = logging.FileHandler(os.path.join('.', 'log_solubility_predictions.log'))
        fh_v.setLevel(logging.INFO)
        logger.addHandler(fh_v)
    return logger

def predict_property(path: str = None,
                     df: pd.DataFrame = None,
                     gsolv: bool = False,
                     hsolv: bool = False,
                     saq: bool = False,
                     solute_parameters: bool = False,
                     reduced_number: bool = False,
                     validate_smiles: bool = False,
                     export_csv: str = False) -> SolubilityPredictions:
    """
    Function to only predict some properties. The desired ones can be specified with arguments.
    Data is read from a csv file or a pandas dataframe.
        :param path: specifies the path to the csv file
        :param df: direct import of pandas dataframe
        :param gsolv: predict solvation free energies
        :param hsolv: predict solvation enthalpies
        :param saq: predict aqueous solubility
        :param solute_parameters: predict solute parameters
        :param validate_smiles: validate the smiles inputs (also converts inchis)
        :param reduced_number: use a reduced number of models for faster but less accurate prediction (does not work for solute_parameters)
        :param export_csv: path if csv file with predictions needs to be exported

        :returns: predictions in the form of SolubilityPredictions
    """

    if not df:
        df = pd.read_csv(path)
    data = SolubilityData(df=df, validate_smiles=validate_smiles, logger=log)
    models = SolubilityModels(reduced_number=reduced_number, load_g=gsolv, load_h=hsolv, load_saq=saq)
    predictions = SolubilityPredictions(data, models, predict_solute_parameters=solute_parameters)
    if export_csv is not None:
        if gsolv:
            df['G_solv_298 [kcal/mol]'] = [i for i in predictions.gsolv[0]]
            df['uncertainty_G_solv_298 [kcal/mol]'] = [np.sqrt(i) for i in predictions.gsolv[1]]
        if hsolv:
            df['H_solv_298 [kcal/mol]'] = [i for i in predictions.hsolv[0]]
            df['uncertainty_H_solv_298 [kcal/mol]'] = [np.sqrt(i) for i in predictions.hsolv[1]]
        if saq:
            df['logS_aq_298 [log10(mol/L)]'] = [i for i in predictions.saq[0]]
            df['uncertainty_logS_aq_298 [log10(mol/L)]'] = [np.sqrt(i) for i in predictions.saq[1]]
        if solute_parameters:
            E, S, A, B, L = [], [], [], [], []
            for i in predictions.solute_parameters[0]:
                E.append(i[0])
                S.append(i[1])
                A.append(i[2])
                B.append(i[3])
                L.append(i[4])
            df['SoluParam_E'] = E
            df['SoluParam_S'] = S
            df['SoluParam_A'] = A
            df['SoluParam_B'] = B
            df['SoluParam_L'] = L

            E_unc, S_unc, A_unc, B_unc, L_unc = [], [], [], [], []
            for i in predictions.solute_parameter_predictions[1]:
                E_unc.append(i[0])
                S_unc.append(i[1])
                A_unc.append(i[2])
                B_unc.append(i[3])
                L_unc.append(i[4])
            df['uncertainty_SoluParam_E'] = E_unc
            df['uncertainty_SoluParam_S'] = S_unc
            df['uncertainty_SoluParam_A'] = A_unc
            df['uncertainty_SoluParam_B'] = B_unc
            df['uncertainty_SoluParam_L'] = L_unc
        df.to_csv(export_csv, index=False)

    return predictions


def calculate_solubility(path: str = None,
                        df: pd.DataFrame = None,
                        calculate_aqueous: bool = False,
                        validate_smiles: bool = False,
                        reduced_number: bool = False,
                        export_csv: str = False) -> SolubilityCalculations:
    """
    Predict relevant properties and make calculations.
    Data is read from a csv file or a pandas dataframe.
        :param path: specifies the path to the csv file
        :param df: direct import of pandas dataframe
        :param calculate_aqueous: also calculate aqueous solubility even if reference solubility is provided
        :param validate_smiles: validate the smiles inputs (also converts inchis)
        :param reduced_number: use a reduced number of models for faster but less accurate prediction (does not work for solute_parameters)
        :param export_csv: path if csv file with predictions needs to be exported

        :returns: calculations in the form of SolubilityCalculations
    """
    if not df:
        df = pd.read_csv(path)
    data = SolubilityData(df=df, validate_smiles=validate_smiles, logger=log)

    predict_reference_solvents = data.reference_solvents and not len([i for i in data.reference_solvents if i]) == 0
    predict_aqueous = calculate_aqueous or not predict_reference_solvents
    predict_t_dep = data.temperatures and (np.min(data.temperatures) < 297. or np.max(data.temperatures) > 299.)

    models = SolubilityModels(reduced_number=reduced_number,
                              load_g=True,
                              load_h=predict_t_dep,
                              load_saq=predict_aqueous)

    predictions = SolubilityPredictions(data,
                                        models,
                                        predict_aqueous=predict_aqueous,
                                        predict_reference_solvents=predict_reference_solvents,
                                        predict_t_dep=predict_t_dep)
    calculations = SolubilityCalculations(predictions,
                                          calculate_aqueous=predict_aqueous,
                                          calculate_reference_solvents=predict_reference_solvents,
                                          calculate_t_dep=predict_t_dep)
    #if export_csv:
        #todo
        #continue
    return calculations

log = create_logger('/home/fhvermei/Software/PycharmProjects/ml_solvation_v01/databases/test.log')
df = pd.read_csv('/home/fhvermei/Software/PycharmProjects/ml_solvation_v01/databases/test.csv')

