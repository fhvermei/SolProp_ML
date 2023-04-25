import numpy as np
import pandas as pd
import rdkit.Chem as Chem
import rdkit.Chem.rdmolops as rdmolops


class SolubilityData:
    def __init__(
        self, df: pd.DataFrame, validate_data_list: list = [], logger=None, verbose=True
    ):
        """Reads in data used to calculate the solubility of neutral solutes in organic solvents
        :param df: pandas dataframe with columns 'solvent', 'solute', 'temperature' 'reference_solubility', and 'reference_solvent'.
        In general only the 'solute' column is mandatory, for solubility predictions at room temperature 'solute'
        and 'solvent', for other temperatures 'temperature'.
        If calculations use experimental reference solubility 'reference_solubility' and 'reference_solvent' have to be specified
        :param validate_data_list: a list of data names to validate. e.g. solvent-solute smiles inputs (also converts inchis)
        and optional inputs such as the reference solvent smiles inputs (also converts inchis), reference solubility inputs (must
        be number), and temperature inputs (must be number).
        """
        self.df = df
        self.smiles_pairs = []
        self.temperatures = None
        self.reference_solubility = None
        self.reference_solvents = None
        self.validate = True if len(validate_data_list) > 0 else False
        self.validate_data_list = validate_data_list
        self.validate_error_messages = None
        self.df_wrongsmiles = None
        self.logger = logger.info if logger is not None else print

        if df is not None:
            self.get_data_from_df(verbose=verbose)
        else:
            raise ValueError(
                "A dataframe needs to be provided with columns 'solvent', 'solute', "
                "'temperature' 'logS_ref', and 'solvent_ref', "
                "currently there is only one possibility to reading data"
            )

    def get_data_from_df(self, verbose=False):
        self.df.columns = self.df.columns.str.strip().str.lower()
        if verbose:
            self.logger("Reading dataframe")
        if not "solute" in self.df.columns:
            raise ValueError("CSV column names must have at least 'solute'")
        elif "solute" in self.df.columns and not "solvent" in self.df.columns:
            if verbose:
                self.logger(
                    "Reading only solute smiles, no solvent smiles are provided"
                )
            self.smiles_pairs = [(None, row.solute) for i, row in self.df.iterrows()]
        else:
            if verbose:
                self.logger("Reading solute and solvent smiles")
            self.smiles_pairs = [
                (row.solvent, row.solute) for i, row in self.df.iterrows()
            ]

        if self.validate:
            if verbose:
                self.logger(
                    "Validating smiles (or converting inchis) and other input values"
                )
            self.validate_data(verbose=verbose)
            # update the smiles pairs
            self.smiles_pairs = [
                (row.solvent, row.solute) for i, row in self.df.iterrows()
            ]

        if "temperature" in self.df.columns:
            if verbose:
                self.logger("Reading temperature")
            self.temperatures = self.df.temperature.values

        if (
            "reference_solubility" in self.df.columns
            and "reference_solvent" in self.df.columns
        ):
            if verbose:
                self.logger("Reading reference solubility")
            self.reference_solubility = self.df.reference_solubility.values
            self.reference_solvents = self.df.reference_solvent.values

        if verbose:
            self.logger("Done reading data")

    def validate_data(self, verbose=False):
        # store the original input SMILES as separate columns and initialize the result lists
        validate_solute, validate_solvent, validate_ref_solvent, validate_temp = (
            False,
            False,
            False,
            False,
        )
        new_value_dict = {}
        cols_to_move, wrong_idx_list, error_msg_list = [], [], []
        if "solute" in self.validate_data_list:
            self.df["input_solute"] = self.df["solute"]
            validate_solute, new_value_dict["solute"] = True, []
            cols_to_move.append("input_solute")
        if "solvent" in self.validate_data_list:
            self.df["input_solvent"] = self.df["solvent"]
            validate_solvent, new_value_dict["solvent"] = True, []
            cols_to_move.append("input_solvent")
        if (
            "reference_solvent" in self.validate_data_list
            and "reference_solvent" in self.df.columns
        ):
            self.df["input_reference_solvent"] = self.df["reference_solvent"]
            cols_to_move.append("input_reference_solvent")
            (
                validate_ref_solvent,
                new_value_dict["reference_solvent"],
                new_value_dict["reference_solubility"],
            ) = (True, [], [])
        if (
            "temperature" in self.validate_data_list
            and "temperature" in self.df.columns
        ):
            validate_temp, new_value_dict["temperature"] = True, []
        # reorder columns
        self.df = self.df[
            cols_to_move + [col for col in self.df.columns if col not in cols_to_move]
        ]
        for index, row in self.df.iterrows():
            error_msg = None
            # validate the solvent, solute, and reference solvent smiles and reference solubility and temperature inputs
            if validate_solute:
                solute_smiles, error_msg = self.validate_smiles(
                    row.solute, error_msg, "solute"
                )
                new_value_dict["solute"].append(solute_smiles)
            if validate_solvent:
                solvent_smiles, error_msg = self.validate_smiles(
                    row.solvent, error_msg, "solvent"
                )
                new_value_dict["solvent"].append(solvent_smiles)
            if "reference_solvent" in self.df.columns and validate_ref_solvent:
                ref_solvent_smiles, error_msg = self.validate_smiles(
                    row.reference_solvent, error_msg, "reference solvent"
                )
                ref_solubility, error_msg = self.check_is_number(
                    row.reference_solubility, error_msg, "reference solubility"
                )
                new_value_dict["reference_solvent"].append(ref_solvent_smiles)
                new_value_dict["reference_solubility"].append(ref_solubility)
            if "temperature" in self.df.columns and validate_temp:
                temperature, error_msg = self.check_is_number(
                    row.temperature, error_msg, "temperature"
                )
                new_value_dict["temperature"].append(temperature)
            # append the results
            if not error_msg is None:
                wrong_idx_list.append(index)
                error_msg_list.append(error_msg)

        # update the df columns with new values
        for key in new_value_dict.keys():
            self.df[key] = new_value_dict[key]
        # split df into to valid df and wrong df
        if len(wrong_idx_list) > 0:
            df_correct = self.df.drop(wrong_idx_list)
            df_wrong_input = self.df.loc[wrong_idx_list]
            df_wrong_input["error"] = error_msg_list
            self.df = df_correct
            self.df_wrong_input = df_wrong_input
            if verbose:
                self.logger(
                    f"Validation done, {len(self.df_wrong_input)} inputs are not correct"
                )
        else:
            self.df_wrong_input = None
            if verbose:
                self.logger(f"Validation done, all inputs are correct")

    def validate_smiles(self, mol_input, error_msg, mol_type):
        mol_smiles = None
        mol_charge = 0
        if mol_input is not None and not pd.isnull(mol_input):
            if "InChI" in mol_input:
                try:
                    mol = Chem.MolFromInchi(mol_input)
                    mol_smiles = Chem.MolToSmiles(mol)
                    mol_charge = rdmolops.GetFormalCharge(mol)
                except:
                    error_msg = self.update_error_msg(
                        error_msg,
                        f"{mol_type} id {mol_input} cannot be converted by RDKit",
                    )
            else:
                try:
                    mol = Chem.MolFromSmiles(mol_input)
                    mol_smiles = Chem.MolToSmiles(mol)
                    mol_charge = rdmolops.GetFormalCharge(mol)
                except:
                    error_msg = self.update_error_msg(
                        error_msg,
                        f"{mol_type} id {mol_input} cannot be converted by RDKit",
                    )
            if mol_charge != 0:
                error_msg = self.update_error_msg(
                    error_msg,
                    f"{mol_type} id {mol_input} has charge {mol_charge} calculated by RDKit, "
                    f"only neutral molecules are allowed",
                )
        else:
            error_msg = self.update_error_msg(
                error_msg, f"{mol_type} input is not provided"
            )
        return mol_smiles, error_msg

    def check_is_number(self, input_value, error_msg, input_type):
        if input_value is None or pd.isnull(input_value):
            return input_value, self.update_error_msg(
                error_msg, f"{input_type} value is not provided"
            )
        elif isinstance(input_value, str):
            try:
                input_value = float(input_value.strip())
            except:
                error_msg = self.update_error_msg(
                    error_msg, f"{input_type} value {input_value} is not numeric"
                )
            return input_value, error_msg
        elif isinstance(input_value, (int, float)):
            return input_value, error_msg
        else:
            return input_value, self.update_error_msg(
                error_msg, f"{input_type} value {input_value} is an unknown type"
            )

    def update_error_msg(self, error_msg, new_msg, overwrite=False):
        if overwrite == True:
            return new_msg
        else:
            if error_msg is None:
                return new_msg
            else:
                return error_msg + ", " + new_msg
