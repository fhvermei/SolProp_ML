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


def convert_arrays_to_df(solvent_smiles,
                         solute_smiles,
                         temperatures=None,
                         reference_solubility=None,
                         reference_solvent=None):
    """
    Function to convert arrays (e.g. for reading from website) to a pandas dataframe that is
    used further in the software. Note that all input arrays need the same size.
        :param solvent_smiles: array of solvent smiles
        :param solute_smiles: array of solute smiles
        :param temperatures: array of temperatures
        :param reference_solubility: array of the reference solubility in another solvent
        :param reference_solvent: array of the solvent smiles associated with the reference solubility

        :returns: pandas dataframe with all data
    """
    df = pd.DataFrame()
    df['solvent'] = solvent_smiles
    df['solute'] = solute_smiles
    if temperatures is not None:
        df['temperature'] = temperatures
    if reference_solubility is not None:
        df['reference_solubility'] = reference_solubility
        df['reference_solvent'] = reference_solvent
    return df


def predict_property(csv_path: str = None,
                     df: pd.DataFrame = None,
                     gsolv: bool = False,
                     hsolv: bool = False,
                     saq: bool = False,
                     solute_parameters: bool = False,
                     reduced_number: bool = False,
                     validate_data_list: list = [],
                     export_csv: str = False,
                     logger=None) -> SolubilityPredictions:
    """
    Function to only predict some properties. The desired ones can be specified with arguments.
    Data is read from a csv file or a pandas dataframe.
        :param csv_path: specifies the path to the csv file
        :param df: direct import of pandas dataframe
        :param gsolv: predict solvation free energies
        :param hsolv: predict solvation enthalpies
        :param saq: predict aqueous solubility
        :param solute_parameters: predict solute parameters
        :param validate_data_list: a list of data names to validate (also converts inchis to smiles)
        :param reduced_number: use a reduced number of models for faster but less accurate prediction (does not work for solute_parameters)
        :param export_csv: path if csv file with predictions needs to be exported

        :returns: predictions in the form of SolubilityPredictions
    """
    logger = create_logger(save_dir=logger)
    if df is None:
        df = pd.read_csv(csv_path)
    df_wrong_input = None
    data = SolubilityData(df=df, validate_data_list=validate_data_list, logger=logger)
    if len(validate_data_list) > 0:
        df = data.df
        df_wrong_input = data.df_wrong_input
    models = SolubilityModels(reduced_number=reduced_number, load_g=gsolv, load_h=hsolv, load_saq=saq, logger=logger)
    predictions = SolubilityPredictions(data, models, predict_solute_parameters=solute_parameters, logger=logger)

    if export_csv is not None:
        df = write_results(df,
                           export_csv,
                           df_wrong_input=df_wrong_input,
                           predictions=predictions)

    return predictions


def calculate_solubility(path: str = None,
                        df: pd.DataFrame = None,
                        calculate_aqueous: bool = False,
                        validate_data_list: list = [],
                        reduced_number: bool = False,
                        export_csv: str = None,
                        export_detailed_csv: bool = False,
                        solv_crit_prop_dict: dict = None,
                        logger = None) -> SolubilityCalculations:
    """
    Predict relevant properties and make calculations.
    Data is read from a csv file or a pandas dataframe.
        :param path: specifies the path to the csv file
        :param df: direct import of pandas dataframe
        :param calculate_aqueous: also calculate aqueous solubility even if reference solubility is provided
        :param validate_data_list: a list data names to validate (also converts inchis to smiles)
        inputs such as the reference solvent smiles inputs (also converts inchis), reference solubility inputs (must be
        number), and temperature inputs (must be number).
        :param reduced_number: use a reduced number of models for faster but less accurate prediction (does not work for solute_parameters)
        :param export_csv: path if csv file with final logS calculations needs to be exported
        :param export_detailed_csv: path if csv file with all predictions and calculations needs to be exported
        :param solv_crit_prop_dict: (optional input) dictionary containing the CoolProp name, critical temperature (in K),
        and critical density (in mol/m3) of solvents. If this is not provided, default dictionary is used. The inchi with
        fixed H option is used as a dictionary key.
            example format of solv_crit_prop_dict:
            {'InChI=1/CH4O/c1-2/h2H,1H3': {'name': 'methanol', 'smiles': 'CO', 'coolprop_name': 'Methanol',
                                            'Tc': 513.0, 'rho_c': 8510.0},
            'InChI=1/C3H8O/c1-3(2)4/h3-4H,1-2H3': {'name': 'propan-2-ol',  'smiles': 'CC(C)O', 'coolprop_name': None,
                                            'Tc': 509.0, 'rho_c': 4500.0}}

        :returns: calculations in the form of SolubilityCalculations
    """
    logger = create_logger(save_dir=logger)
    if df is None:
        df = pd.read_csv(path)

    df_wrong_input = None
    data = SolubilityData(df=df, validate_data_list=validate_data_list, logger=logger)
    if len(validate_data_list) > 0:
        df = data.df
        df_wrong_input = data.df_wrong_input

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
                                          solv_crit_prop_dict=solv_crit_prop_dict,
                                          logger=logger)
    if export_csv is not None:
        df = write_results(df,
                           export_csv,
                           df_wrong_input=df_wrong_input,
                           predictions=predictions,
                           calculations=calculations,
                           detail=export_detailed_csv)
    return calculations


def write_results(df,
                  export_path=None,
                  df_wrong_input=None,
                  predictions:SolubilityPredictions = None,
                  calculations:SolubilityCalculations = None,
                  detail=False):
    """
    Function to write the predictions and calculations to a pandas dataframe and export to csv
        :param df: initial pandas dataframe with validated input data
        :param export_path: path to export the csv file if export is required
        :param df_wrong_input: the pandas dataframe with wrong inputs and error messages, created after validation
        :param predictions: the predictions to write, of the class SolubilityPredictions
        :param calculations: the calculations to write, of the class SolubilityCalculations
        :param detail: boolean is detailed calculations are required

        :returns: pandas dataframe with predictions and calculations
    """
    if predictions:
        if predictions.gsolv is not None:
            df['G_solv_298 [kcal/mol]'] = [i for i in predictions.gsolv[0]]
            df['uncertainty_G_solv_298 [kcal/mol]'] = [np.sqrt(i) for i in predictions.gsolv[1]]
        if predictions.gsolv_aq is not None:
            df['G_solv_aq_298 [kcal/mol]'] = [i for i in predictions.gsolv_aq[0]]
            df['uncertainty_G_solv_aq_298 [kcal/mol]'] = [np.sqrt(i) for i in predictions.gsolv_aq[1]]
        if predictions.gsolv_ref is not None:
            df['G_solv_ref_298 [kcal/mol]'] = [i for i in predictions.gsolv_ref[0]]
            df['uncertainty_G_solv_ref_298 [kcal/mol]'] = [np.sqrt(i) for i in predictions.gsolv_ref[1]]
        if predictions.hsolv is not None:
            df['H_solv_298 [kcal/mol]'] = [i for i in predictions.hsolv[0]]
            df['uncertainty_H_solv_298 [kcal/mol]'] = [np.sqrt(i) for i in predictions.hsolv[1]]
        if predictions.saq is not None:
            df['logS_aq_298 [log10(mol/L)]'] = [i for i in predictions.saq[0]]
            df['uncertainty_logS_aq_298 [log10(mol/L)]'] = [np.sqrt(i) for i in predictions.saq[1]]
        if predictions.solute_parameters is not None:
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

        if calculations is not None:
            if calculations.logs_298_from_aq is not None:
                df['logS_298_from_aq [log10(mol/L)]'] = calculations.logs_298_from_aq
                df['uncertainty_logS_298_from_aq [log10(mol/L)]'] = calculations.unc_logs_298_from_aq
            if calculations.logs_298_from_ref is not None:
                df['logS_298_from_ref [log10(mol/L)]'] = calculations.logs_298_from_ref
                df['uncertainty_logS_298_from_ref [log10(mol/L)]'] = calculations.unc_logs_298_from_ref
            if calculations.logs_T_from_aq is not None:
                df['logS_T_from_aq [log10(mol/L)]'] = calculations.logs_T_from_aq
            if calculations.logs_T_from_ref is not None:
                df['logS_T_from_ref [log10(mol/L)]'] = calculations.logs_T_from_ref
            if calculations.logs_T_with_t_dep_hsolu_from_aq is not None:
                df['logS_T_from_aq_with_T_dep_Hsolu [log10(mol/L)]'] = calculations.logs_T_with_t_dep_hsolu_from_aq
                df['error_message_for_T_dep_Hsolu_prediction'] = calculations.logs_T_with_t_dep_hsolu_error_message
            if calculations.logs_T_with_t_dep_hsolu_from_ref is not None:
                df['logS_T_from_ref_with_T_dep_Hsolu [log10(mol/L)]'] = calculations.logs_T_with_t_dep_hsolu_from_ref
                df['error_message_for_T_dep_Hsolu_prediction'] = calculations.logs_T_with_t_dep_hsolu_error_message

            if detail:
                if calculations.logk_298 is not None:
                    df['logK_298 [log10(-)]'] = calculations.logk_298
                    df['uncertainty_logK_298 [log10(-)]'] = calculations.unc_logk_298
                if calculations.logk_aq_298 is not None:
                    df['logK_aq_298 [log10(-)]'] = calculations.logk_aq_298
                    df['uncertainty_logK_aq_298 [log10(-)]'] = calculations.unc_logk_aq_298
                if calculations.logs_298_from_aq is not None:
                    df['logS_298_from_aq [log10(mol/L)]'] = calculations.logs_298_from_aq
                    df['uncertainty_logS_298_from_aq [log10(mol/L)]'] = calculations.unc_logs_298_from_aq
                if calculations.logk_ref_298 is not None:
                    df['logK_ref_298 [log10(-)]'] = calculations.logk_ref_298
                    df['uncertainty_logK_ref_298 [log10(-)]'] = calculations.unc_logk_ref_298
                if calculations.logs_298_from_ref is not None:
                    df['logS_298_from_ref [log10(mol/L)]'] = calculations.logs_298_from_ref
                    df['uncertainty_logS_298_from_ref [log10(mol/L)]'] = calculations.unc_logs_298_from_ref
                if calculations.V is not None:
                    df['V_solute_parameter'] = calculations.V
                    df['adjacent diol parameter'] = calculations.I_OHadj
                    df['non-adjacent diol parameter'] = calculations.I_OHnonadj
                    df['amine parameter'] = calculations.I_NH
                    df['H_sub_298 [kcal/mol]'] = calculations.hsubl_298
                if calculations.Cp_gas is not None:
                    df['Cp_gas [J/K/mol]'] = calculations.Cp_gas
                    df['Cp_solid [J/K/mol]'] = calculations.Cp_solid
                if calculations.hsolv_T is not None:
                    df['G_solv_T [kcal/mol]'] = calculations.gsolv_T
                    df['H_solv_T [kcal/mol]'] = calculations.hsolv_T
                    df['S_solv_T [kcal/K/mol]'] = calculations.ssolv_T

    if df_wrong_input is not None:
        df = pd.concat([df, df_wrong_input], ignore_index=True)
    if export_path is not None:
        df.to_csv(export_path, index=False)

    return df


df = pd.read_csv('/home/fhvermei/Software/PycharmProjects/ml_solvation_v01/databases/test.csv')
predictions = predict_property(csv_path=None,
                               df=df,
                               gsolv=True,
                               hsolv=False,
                               saq=False,
                               solute_parameters=False,
                               reduced_number=False,
                               validate_data_list=['solute', 'solvent'],
                               export_csv='./../results_predictions.csv',
                               logger='/home/fhvermei/Software/PycharmProjects/ml_solvation_v01/databases/test.log')

df = pd.read_csv('/home/fhvermei/Software/PycharmProjects/ml_solvation_v01/databases/test.csv')
results = calculate_solubility(path=None,
                               df=df,
                               validate_data_list=['solute', 'solvent', 'reference_solvent',
                                                   'reference_solubility', 'temperature'],
                               calculate_aqueous=True,
                               reduced_number=False,
                               export_csv='./../results_calculations.csv',
                               export_detailed_csv=True,
                               solv_crit_prop_dict=None,
                               logger='/home/fhvermei/Software/PycharmProjects/ml_solvation_v01/databases/test.log')
