import logging
import os

import numpy as np
import pandas as pd

from solvation_predictor.solubility.solubility_calculator import SolubilityCalculations
from solvation_predictor.solubility.solubility_data import SolubilityData
from solvation_predictor.solubility.solubility_models import SolubilityModels
from solvation_predictor.solubility.solubility_predictions import SolubilityPredictions


def create_logger(save_dir=None):
    if save_dir is not None:
        path = save_dir
    else:
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'log_solubility_predictions.log')
    logger = logging.getLogger(path)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if save_dir is not None:
        fh_v = logging.FileHandler(path)
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
                     export_csv: str = False,
                     logger=None) -> SolubilityPredictions:
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
    logger = create_logger(save_dir=logger)
    if df is None:
        df = pd.read_csv(path)
    data = SolubilityData(df=df, validate_smiles=validate_smiles, logger=logger)
    models = SolubilityModels(reduced_number=reduced_number, load_g=gsolv, load_h=hsolv, load_saq=saq, logger=logger)
    predictions = SolubilityPredictions(data, models, predict_solute_parameters=solute_parameters, logger=logger)
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
            for i in predictions.solute_parameters[1]:
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
                        export_csv: str = False,
                        export_detailed_csv: str = False,
                        logger = None) -> SolubilityCalculations:
    """
    Predict relevant properties and make calculations.
    Data is read from a csv file or a pandas dataframe.
        :param path: specifies the path to the csv file
        :param df: direct import of pandas dataframe
        :param calculate_aqueous: also calculate aqueous solubility even if reference solubility is provided
        :param validate_smiles: validate the smiles inputs (also converts inchis)
        :param reduced_number: use a reduced number of models for faster but less accurate prediction (does not work for solute_parameters)
        :param export_csv: path if csv file with final logS calculations needs to be exported
        :param export_detailed_csv: path if csv file with all predictions and calculations needs to be exported

        :returns: calculations in the form of SolubilityCalculations
    """
    logger = create_logger(save_dir=logger)
    if df is None:
        df = pd.read_csv(path)
    data = SolubilityData(df=df, validate_smiles=validate_smiles, logger=logger)

    predict_reference_solvents = data.reference_solvents is not None and not len([i for i in data.reference_solvents if i]) == 0
    predict_aqueous = calculate_aqueous or not predict_reference_solvents
    predict_t_dep = data.temperatures is not None and (np.min(data.temperatures) < 297. or np.max(data.temperatures) > 299.)
    calculate_t_dep_with_t_dep_hsolu = predict_t_dep

    models = SolubilityModels(reduced_number=reduced_number,
                              load_g=True,
                              load_h=predict_t_dep,
                              load_saq=predict_aqueous,
                              logger=logger)

    predictions = SolubilityPredictions(data,
                                        models,
                                        predict_aqueous=predict_aqueous,
                                        predict_reference_solvents=predict_reference_solvents,
                                        predict_t_dep=predict_t_dep,
                                        logger=logger)

    calculations = SolubilityCalculations(predictions,
                                          calculate_aqueous=predict_aqueous,
                                          calculate_reference_solvents=predict_reference_solvents,
                                          calculate_t_dep=predict_t_dep,
                                          calculate_t_dep_with_t_dep_hsolu=calculate_t_dep_with_t_dep_hsolu,
                                          logger=logger)

    if export_csv is not None or export_detailed_csv is not None:
        if predict_aqueous:
            df['logS_298_from_aq [log10(mol/L)]'] = calculations.logs_298_from_aq
            df['uncertainty_logS_298_from_aq [log10(mol/L)]'] = calculations.unc_logs_298_from_aq
        if predict_reference_solvents:
            df['logS_298_from_ref [log10(mol/L)]'] = calculations.logs_298_from_ref
            df['uncertainty_logS_298_from_ref [log10(mol/L)]'] = calculations.unc_logs_298_from_ref
        if predict_t_dep:
            if predict_aqueous:
                df['logS_T_from_aq [log10(mol/L)]'] = calculations.logs_T_from_aq
            if predict_reference_solvents:
                df['logS_T_from_ref [log10(mol/L)]'] = calculations.logs_T_from_ref
        if calculate_t_dep_with_t_dep_hsolu:
            if predict_aqueous:
                df['logS_T_from_aq_with_T_dep_Hsolu [log10(mol/L)]'] = calculations.logs_T_with_t_dep_hsolu_from_aq
            if predict_reference_solvents:
                df['logS_T_from_ref_with_T_dep_Hsolu [log10(mol/L)]'] = calculations.logs_T_with_t_dep_hsolu_from_ref
            df['error_message_for_T_dep_Hsolu_prediction'] = calculations.logs_T_with_t_dep_hsolu_error_message
        if export_csv is not None:
            df.to_csv(export_csv, index=False)

        if export_detailed_csv is not None:
            df_details = df
            if calculations.gsolv_298 is not None:
                df_details['G_solv_298 [kcal/mol]'] = calculations.gsolv_298
                df_details['uncertainty_G_solv_298 [kcal/mol]'] = calculations.unc_gsolv_298
                df_details['logK_298 [log10(-)]'] = calculations.logk_298
                df_details['uncertainty_logK_298 [log10(-)]'] = calculations.unc_logk_298
            if predict_aqueous:
                df_details['G_solv_aq_298 [kcal/mol]'] = calculations.gsolv_aq_298
                df_details['uncertainty_G_solv_aq_298 [kcal/mol]'] = calculations.unc_gsolv_aq_298
                df_details['logK_aq_298 [log10(-)]'] = calculations.logk_aq_298
                df_details['uncertainty_logK_aq_298 [log10(-)]'] = calculations.unc_logk_aq_298
                df_details['logS_aq_298 [log10(mol/L)]'] = calculations.logs_aq_298
                df_details['uncertainty_logS_aq_298 [log10(mol/L)]'] = calculations.unc_logs_aq_298
                df_details['logS_298_from_aq [log10(mol/L)]'] = calculations.logs_298_from_aq
                df_details['uncertainty_logS_298_from_aq [log10(mol/L)]'] = calculations.unc_logs_298_from_aq
            if predict_reference_solvents:
                df_details['G_solv_ref_298 [kcal/mol]'] = calculations.gsolv_ref_298
                df_details['uncertainty_G_solv_ref_298 [kcal/mol]'] = calculations.unc_gsolv_ref_298
                df_details['logK_ref_298 [log10(-)]'] = calculations.logk_ref_298
                df_details['uncertainty_logK_ref_298 [log10(-)]'] = calculations.unc_logk_ref_298
                df_details['logS_298_from_ref [log10(mol/L)]'] = calculations.logs_298_from_ref
                df_details['uncertainty_logS_298_from_ref [log10(mol/L)]'] = calculations.unc_logs_298_from_ref
            if predict_t_dep:
                df_details['H_solv_298 [kcal/mol]'] = calculations.hsolv_298
                df_details['uncertainty_H_solv_298 [kcal/mol]'] = calculations.unc_hsolv_298
                df_details['E_solute_parameter'] = calculations.E
                df_details['uncertainty_E_solute_parameter'] = calculations.unc_E
                df_details['S_solute_parameter'] = calculations.S
                df_details['uncertainty_S_solute_parameter'] = calculations.unc_S
                df_details['A_solute_parameter'] = calculations.A
                df_details['uncertainty_A_solute_parameter'] = calculations.unc_A
                df_details['B_solute_parameter'] = calculations.B
                df_details['uncertainty_B_solute_parameter'] = calculations.unc_B
                df_details['V_solute_parameter'] = calculations.V
                df_details['L_solute_parameter'] = calculations.L
                df_details['uncertainty_L_solute_parameter'] = calculations.unc_L
                df_details['adjacent diol'] = calculations.I_OHadj
                df_details['non-adjacent diol'] = calculations.I_OHnonadj
                df_details['amine'] = calculations.I_NH
                df_details['H_subl_298 [kcal/mol]'] = calculations.hsubl_298
            if calculate_t_dep_with_t_dep_hsolu:
                df_details['Cp_gas [J/K/mol]'] = calculations.Cp_gas
                df_details['Cp_solid [J/K/mol]'] = calculations.Cp_solid
                df_details['H_solv_T [kcal/mol]'] = calculations.hsolv_T
            df_details.to_csv(export_detailed_csv, index=False)

    return calculations


df = pd.read_csv('/home/fhvermei/Software/PycharmProjects/ml_solvation_v01/databases/test.csv')
predictions = predict_property(path=None,
                               df=df,
                               gsolv=True,
                               hsolv=False,
                               saq=False,
                               solute_parameters=False,
                               reduced_number=False,
                               validate_smiles=False,
                               export_csv='./../results_predictions.csv',
                               logger='/home/fhvermei/Software/PycharmProjects/ml_solvation_v01/databases/test.log')

df = pd.read_csv('/home/fhvermei/Software/PycharmProjects/ml_solvation_v01/databases/test.csv')
results = calculate_solubility(path=None,
                               df=df,
                               validate_smiles=True,
                               calculate_aqueous=True,
                               reduced_number=False,
                               export_csv='./../results_calculations.csv',
                               export_detailed_csv='./../detailed_results_calculations.csv',
                               logger='/home/fhvermei/Software/PycharmProjects/ml_solvation_v01/databases/test.log')
