import pandas as pd
import numpy as np
from rdkit import Chem
import rdkit.Chem.rdmolops as rdmolops

from solvation_predictor.solubility.solubility_calculator import SolubilityCalculations
from solvation_predictor.solubility.solubility_predictions import SolubilityPredictions


class SolubilityData:
    """
    Class for storing the input data for solubility prediction
    """

    def __init__(self, solvent_smiles=None, solute_smiles=None, temp=None, ref_solub=None, ref_solv=None):
        self.smiles_pairs = [(solvent_smiles, solute_smiles)]
        self.temperatures = np.array([temp]) if temp is not None else None
        self.reference_solubility = np.array([ref_solub]) if ref_solub is not None else None
        self.reference_solvents = np.array([ref_solv]) if ref_solv is not None else None


def calc_solubility_no_ref(models=None, solvent_smiles=None, solute_smiles=None, temp=None, hsub298=None, cp_gas_298=None,
                           cp_solid_298=None):
    """
    Calculate solubility with no reference solvent and reference solubility
    """

    hsubl_298 = np.array([hsub298]) if hsub298 is not None else None
    Cp_solid = np.array([cp_solid_298]) if cp_solid_298 is not None else None
    Cp_gas = np.array([cp_gas_298]) if cp_gas_298 is not None else None

    solub_data = SolubilityData(solvent_smiles=solvent_smiles, solute_smiles=solute_smiles, temp=temp)
    predictions = SolubilityPredictions(solub_data, models, predict_aqueous=True,
                                        predict_reference_solvents=False, predict_t_dep=True,
                                        predict_solute_parameters=True, verbose=False)
    calculations = SolubilityCalculations(predictions, calculate_aqueous=True,
                                          calculate_reference_solvents=False, calculate_t_dep=True,
                                          calculate_t_dep_with_t_dep_hdiss=True, verbose=False,
                                          hsubl_298=hsubl_298, Cp_solid=Cp_solid, Cp_gas=Cp_gas)
    return calculations


def calc_solubility_with_ref(models=None, solvent_smiles=None, solute_smiles=None, temp=None, ref_solvent_smiles=None,
                             ref_solubility298=None, hsub298=None, cp_gas_298=None, cp_solid_298=None):
    """
    Calculate solubility with a reference solvent and reference solubility
    """

    hsubl_298 = np.array([hsub298]) if hsub298 is not None else None
    Cp_solid = np.array([cp_solid_298]) if cp_solid_298 is not None else None
    Cp_gas = np.array([cp_gas_298]) if cp_gas_298 is not None else None

    solub_data = SolubilityData(solvent_smiles=solvent_smiles, solute_smiles=solute_smiles, temp=temp,
                                ref_solub=ref_solubility298, ref_solv=ref_solvent_smiles)
    predictions = SolubilityPredictions(solub_data, models, predict_aqueous=False,
                                        predict_reference_solvents=True, predict_t_dep=True,
                                        predict_solute_parameters=True, verbose=False)
    calculations = SolubilityCalculations(predictions, calculate_aqueous=False,
                                          calculate_reference_solvents=True, calculate_t_dep=True,
                                          calculate_t_dep_with_t_dep_hdiss=True, verbose=False,
                                          hsubl_298=hsubl_298, Cp_solid=Cp_solid, Cp_gas=Cp_gas)
    return calculations


def get_ref_solubility298(calculations_ref=None, ref_solubility=None):
    """
    Estimate the reference solubility at 298 K based on the reference solvent calculation results and input
    reference solubility value at reference temperature.
    """
    # Use the prediction from method 2 if available. If not, use the prediction from method 1 as the ref value
    logST_ref_pred = calculations_ref.logs_T_with_T_dep_hdiss_from_aq[0]
    if logST_ref_pred is None:
        logST_ref_pred = calculations_ref.logs_T_with_const_hdiss_from_aq[0]
    # Get ref_solubility value at 298 K from ref_solubility at T
    logS298_ref_from_aq = calculations_ref.logs_298_from_aq[0]
    logS_diff = logST_ref_pred - logS298_ref_from_aq
    ref_solubility298 = ref_solubility - logS_diff
    return ref_solubility298


def format_T_dep_hdiss_error_mesg(error_msg):
    """
    Turn the error_msg for the T-dep Hdiss prediction into appropriate error and warning messages
    """
    calc_error_msg, warning_msg = None, None
    if error_msg is not None:
        if 'above the critical temperature' in error_msg:
            calc_error_msg = error_msg
        elif 'The given solvent is not supported' in error_msg:
            calc_error_msg = 'The input solvent is not supported by method 2'
        elif 'Unable to predict dHdissT for T' in error_msg or 'is below the minimum limit' in error_msg:
            warning_msg = 'The input temperature is too low. The prediction may not be reliable'
    return calc_error_msg, warning_msg


def update_error_msg(error_msg, new_msg, overwrite=False):
    if overwrite:
        return new_msg
    else:
        if error_msg is None:
            return new_msg
        else:
            return error_msg + ', ' + new_msg


def validate_smiles(mol_input, error_msg, mol_type):
    """
    Convert the mol_input into SMILES using RDKit and returns an error message if it fails
    """
    mol_smiles = None
    mol_charge = 0
    if mol_input is not None and not pd.isnull(mol_input):
        if 'InChI' in mol_input:
            try:
                mol = Chem.MolFromInchi(mol_input)
                mol_smiles = Chem.MolToSmiles(mol)
                mol_charge = rdmolops.GetFormalCharge(mol)
            except:
                error_msg = update_error_msg(
                    error_msg, f'{mol_type} is invalid')
        else:
            try:
                mol = Chem.MolFromSmiles(mol_input)
                mol_smiles = Chem.MolToSmiles(mol)
                mol_charge = rdmolops.GetFormalCharge(mol)
            except:
                error_msg = update_error_msg(
                    error_msg, f'{mol_type} is invalid')
        if mol_charge != 0:
            error_msg = update_error_msg(
                error_msg, f'{mol_type} has charge {mol_charge} but only neutral molecules are allowed')
    else:
        error_msg = update_error_msg(error_msg, f'{mol_type} input is not empty')
    return mol_smiles, error_msg


def clean_up_value(value, sigfigs=3, lower_limit=1e-1, upper_limit=1000):
    """
    Round the given value to the specified number of significant figures.

    If lower_limit <= abs(value) <= upper_limit, return fixed notation.
    Otherwise, return exponential notation.

    Args:
        value (float): number to format
        sigfigs (int, optional): number of significant figures
        lower_limit (float, optional): lower threshold for exponential notation
        upper_limit (float, optional): upper threshold for exponential notation

    Returns:
        str: rounded value
    """
    if value is None:
        return value
    if (lower_limit and abs(value) < lower_limit) or (upper_limit and abs(value) > upper_limit):
        return np.format_float_scientific(value, precision=sigfigs - 1)  # precision only counts digits after decimal
    else:
        return np.format_float_positional(value, precision=sigfigs, fractional=False)  # precision counts sig digits


def get_solubility_pred(solvent_smiles, solute_smiles, temp, ref_solvent_smiles, ref_solubility, ref_temp,
                        hsub298, cp_gas_298, cp_solid_298, models):
    """
    Get solubility prediction for the input solvent and solute pair at the input temperature
    """

    # Case 1: reference values are not provided
    if ref_solvent_smiles is None:
        calculations = calc_solubility_no_ref(models=models, solvent_smiles=solvent_smiles, solute_smiles=solute_smiles, temp=temp,
                                              hsub298=hsub298, cp_gas_298=cp_gas_298, cp_solid_298=cp_solid_298)
        # Extract the solubility prediction results using from_aq keys
        logST_method1 = calculations.logs_T_with_const_hdiss_from_aq[0]
        logST_method2 = calculations.logs_T_with_T_dep_hdiss_from_aq[0]
        logS298 = calculations.logs_298_from_aq[0]
        logS298_unc = calculations.unc_logs_298_from_aq[0]

    # Case 2: reference values are provided
    else:
        calculations_ref = calc_solubility_no_ref(models=models, solvent_smiles=ref_solvent_smiles, solute_smiles=solute_smiles,
                                                  temp=ref_temp, hsub298=hsub298, cp_gas_298=cp_gas_298,
                                                  cp_solid_298=cp_solid_298)
        ref_solubility298 = get_ref_solubility298(calculations_ref=calculations_ref, ref_solubility=ref_solubility)
        calculations = calc_solubility_with_ref(models=models, solvent_smiles=solvent_smiles, solute_smiles=solute_smiles, temp=temp,
                                                ref_solvent_smiles=ref_solvent_smiles,
                                                ref_solubility298=ref_solubility298,
                                                hsub298=hsub298, cp_gas_298=cp_gas_298, cp_solid_298=cp_solid_298)
        # Extract the solubility prediction results using from_ref keys
        logST_method1 = calculations.logs_T_with_const_hdiss_from_ref[0]
        logST_method2 = calculations.logs_T_with_T_dep_hdiss_from_ref[0]
        logS298 = calculations.logs_298_from_ref[0]
        logS298_unc = calculations.unc_logs_298_from_ref[0]

    # Extract other results
    dGsolvT, dHsolvT, dSsolvT = calculations.gsolv_T[0], calculations.hsolv_T[0], calculations.ssolv_T[0]
    Hsub298_pred = calculations.hsubl_298[0] if hsub298 is None else None
    Cpg298_pred = calculations.Cp_gas[0] if cp_gas_298 is None else None
    Cps298_pred = calculations.Cp_solid[0] if cp_solid_298 is None else None
    dGsolv298, dGsolv_unc = calculations.gsolv_298[0], calculations.unc_gsolv_298[0]
    dHsolv298, dHsolv_unc = calculations.hsolv_298[0], calculations.unc_hsolv_298[0]
    E, S, A, B, L, V = calculations.E[0], calculations.S[0], calculations.A[0], calculations.B[0], \
                       calculations.L[0], calculations.V[0]
    T_dep_hdiss_error_mesg = calculations.logs_T_with_T_dep_hdiss_error_message[0]
    calc_error_msg, warning_msg = format_T_dep_hdiss_error_mesg(T_dep_hdiss_error_mesg)

    pred_val_list = [logST_method1, logST_method2, dGsolvT, dHsolvT, dSsolvT,
                     Hsub298_pred, Cpg298_pred, Cps298_pred,
                     logS298, logS298_unc, dGsolv298, dGsolv_unc, dHsolv298, dHsolv_unc,
                     E, S, A, B, L, V]

    return pred_val_list, calc_error_msg, warning_msg


def make_solubility_prediction(models=None, solvent_list=None, solute_list=None, temp_list=None,
                               ref_solvent_list=None, ref_solubility_list=None, ref_temp_list=None,
                               hsub298_list=None, cp_gas_298_list=None, cp_solid_298_list=None):
    """
    Returns a pandas dataframe with the given solubility data results.
    """

    # Prepare an empty result dictionary
    solubility_results = {}
    input_col_name_list = [
        'Solvent', 'Solute', 'Temp', 'Ref. Solv', 'Ref. Solub', 'Ref. Temp',
        'Input Hsub298', 'Input Cpg298', 'Input Cps298',
        'Error Message', 'Warning Message',
    ]
    output_col_name_list = [
        'logST (method1) [log10(mol/L)]', 'logST (method2) [log10(mol/L)]',
        'dGsolvT [kcal/mol]', 'dHsolvT [kcal/mol]', 'dSsolvT [cal/K/mol]',
        'Pred. Hsub298 [kcal/mol]', 'Pred. Cpg298 [cal/K/mol]', 'Pred. Cps298 [cal/K/mol]',
        'logS298 [log10(mol/L)]', 'uncertainty logS298 [log10(mol/L)]',
        'dGsolv298 [kcal/mol]', 'uncertainty dGsolv298 [kcal/mol]',
        'dHsolv298 [kcal/mol]', 'uncertainty dHsolv298 [kcal/mol]',
        'E', 'S', 'A', 'B', 'L', 'V'
    ]
    col_name_list = input_col_name_list + output_col_name_list

    for col_name in col_name_list:
        solubility_results[col_name] = []

    for solvent, solute, temp, ref_solvent, ref_solubility, ref_temp, hsub298, cp_gas_298, cp_solid_298 \
            in zip(solvent_list, solute_list, temp_list, ref_solvent_list, ref_solubility_list, ref_temp_list,
                   hsub298_list, cp_gas_298_list, cp_solid_298_list):
        # Initialize the outputs
        error_msg, warning_msg = None, None

        # First check whether solvent and solute have valid SMILES
        solvent_smiles, error_msg = validate_smiles(solvent, error_msg, 'Solvent SMILES')
        solute_smiles, error_msg = validate_smiles(solute, error_msg, 'Solute SMILES')
        if ref_solvent is not None:
            ref_solvent_smiles, error_msg = validate_smiles(ref_solvent, error_msg, 'Ref. solvent SMILES')
        else:
            ref_solvent_smiles = None

        # Get the predictions if there is no error with input
        if error_msg is None:
            pred_val_list, error_msg, warning_msg = get_solubility_pred(solvent_smiles, solute_smiles, temp,
                                                                        ref_solvent_smiles, ref_solubility, ref_temp,
                                                                        hsub298, cp_gas_298, cp_solid_298, models)
        else:
            pred_val_list = [None for _ in output_col_name_list]

        # Append the results
        input_val_list = [solvent, solute, temp, ref_solvent, ref_solubility, ref_temp,
                          hsub298, cp_gas_298, cp_solid_298, error_msg, warning_msg]
        result_val_list = input_val_list + pred_val_list

        for key, val in zip(col_name_list, result_val_list):
            if key in output_col_name_list:
                if key in ['E', 'S', 'A', 'B', 'L', 'V']:
                    val = clean_up_value(val, sigfigs=4)
                else:
                    val = clean_up_value(val, sigfigs=3, lower_limit=0)
            solubility_results[key].append(val)

    return solubility_results
